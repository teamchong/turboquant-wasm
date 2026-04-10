/** TQ GPU KV Cache — compress K/V vectors on GPU, all data stays compressed. */

import encodeShaderSrc from "./shaders/tq-encode.wgsl?raw";

// LCG + Box-Muller gaussian + Gram-Schmidt, matching Zig reference.
function generateRotationMatrix(dim: number, seed: number): Float32Array {
  const mat = new Float32Array(dim * dim);

  function nextRng(state: { v: bigint }): bigint {
    state.v = BigInt.asUintN(64, state.v * 1103515245n + 12345n);
    return state.v;
  }

  function randF32(state: { v: bigint }): number {
    const val = Number(nextRng(state) % (1n << 31n)) / (1 << 31);
    return val === 0 ? 0.00001 : val;
  }

  function randGaussian(state: { v: bigint }): number {
    const v1 = randF32(state);
    const v2 = randF32(state);
    return Math.sqrt(-2 * Math.log(v1)) * Math.cos(2 * Math.PI * v2);
  }

  for (let row = 0; row < dim; row++) {
    for (let col = 0; col < dim; col++) {
      const rngState = { v: BigInt(seed) + BigInt(row * 31 + col) };
      mat[row * dim + col] = randGaussian(rngState);
    }
  }

  // Gram-Schmidt orthogonalization (in-place, row-wise)
  for (let i = 0; i < dim; i++) {
    for (let j = 0; j < i; j++) {
      let dot = 0;
      for (let k = 0; k < dim; k++) {
        dot += mat[i * dim + k] * mat[j * dim + k];
      }
      for (let k = 0; k < dim; k++) {
        mat[i * dim + k] -= dot * mat[j * dim + k];
      }
    }
    let norm = 0;
    for (let k = 0; k < dim; k++) {
      norm += mat[i * dim + k] * mat[i * dim + k];
    }
    norm = Math.sqrt(norm);
    if (norm > 1e-10) {
      for (let k = 0; k < dim; k++) {
        mat[i * dim + k] /= norm;
      }
    }
  }

  return mat;
}

interface CacheBuffers {
  polar: GPUBuffer;
  qjl: GPUBuffer;
  maxR: GPUBuffer;
  gamma: GPUBuffer;
  length: number;
}

export class TQGpuCache {
  private device: GPUDevice;
  private dim: number;
  private maxPositions: number;
  private polarWordsPerPos: number;
  private qjlWordsPerPos: number;
  private rotationBuf: GPUBuffer;

  private caches: Map<string, CacheBuffers> = new Map();
  private encodePipeline: GPUComputePipeline | null = null;

  constructor(device: GPUDevice, dim: number, maxPositions: number, seed: number) {
    this.device = device;
    this.dim = dim;
    this.maxPositions = maxPositions;
    this.polarWordsPerPos = Math.ceil((dim / 2) * 7 / 32);
    this.qjlWordsPerPos = dim / 32;

    const rotMat = generateRotationMatrix(dim, seed);
    this.rotationBuf = device.createBuffer({
      size: rotMat.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.rotationBuf, 0, rotMat);
  }

  async initPipelines(): Promise<void> {
    const encodeModule = this.device.createShaderModule({ code: encodeShaderSrc });
    this.encodePipeline = await this.device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: encodeModule, entryPoint: "encode" },
    });
  }

  private getOrCreateCache(key: string): CacheBuffers {
    let cache = this.caches.get(key);
    if (cache) return cache;

    const d = this.device;
    const maxPos = this.maxPositions;
    cache = {
      polar: d.createBuffer({
        size: maxPos * this.polarWordsPerPos * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
      qjl: d.createBuffer({
        size: maxPos * this.qjlWordsPerPos * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
      maxR: d.createBuffer({
        size: maxPos * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
      gamma: d.createBuffer({
        size: maxPos * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
      length: 0,
    };
    this.caches.set(key, cache);
    return cache;
  }

  private stagingBuf: GPUBuffer | null = null;
  private paramStagingBuf: GPUBuffer | null = null;
  private pendingEncodes: Array<{ cacheKey: string; data: Float32Array }> = [];

  /** Queue a raw (unrotated) vector for compression. Call flush() to submit. */
  encodeAndAppend(cacheKey: string, rawData: Float32Array): void {
    this.pendingEncodes.push({ cacheKey, data: rawData });
  }

  /** Submit all pending encodes as a single GPU command buffer. */
  flush(): void {
    if (this.pendingEncodes.length === 0) return;
    const d = this.device;
    const dim = this.dim;
    const encoder = d.createCommandEncoder();

    for (const { cacheKey, data } of this.pendingEncodes) {
      const cache = this.getOrCreateCache(cacheKey);
      const pos = cache.length;
      if (pos >= this.maxPositions) continue;

      if (!this.stagingBuf || this.stagingBuf.size < data.byteLength) {
        this.stagingBuf?.destroy();
        this.stagingBuf = d.createBuffer({
          size: dim * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
      }
      d.queue.writeBuffer(this.stagingBuf, 0, data);

      if (!this.paramStagingBuf) {
        this.paramStagingBuf = d.createBuffer({
          size: 32,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
      }
      d.queue.writeBuffer(this.paramStagingBuf, 0, new Uint32Array([
        dim, 1, this.polarWordsPerPos, this.qjlWordsPerPos, pos, 0, 0, 0,
      ]));

      const pass = encoder.beginComputePass();
      pass.setPipeline(this.encodePipeline!);
      pass.setBindGroup(0, d.createBindGroup({
        layout: this.encodePipeline!.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.paramStagingBuf } },
          { binding: 1, resource: { buffer: this.stagingBuf } },
          { binding: 2, resource: { buffer: this.rotationBuf } },
          { binding: 3, resource: { buffer: cache.polar } },
          { binding: 4, resource: { buffer: cache.qjl } },
          { binding: 5, resource: { buffer: cache.maxR } },
          { binding: 6, resource: { buffer: cache.gamma } },
        ],
      }));
      pass.dispatchWorkgroups(1);
      pass.end();
      cache.length = pos + 1;
    }

    d.queue.submit([encoder.finish()]);
    this.pendingEncodes.length = 0;
  }

  getCacheLength(key: string): number {
    return this.caches.get(key)?.length ?? 0;
  }

  resetCache(key: string): void {
    const cache = this.caches.get(key);
    if (cache) cache.length = 0;
  }

  resetAll(): void {
    for (const cache of this.caches.values()) {
      cache.length = 0;
    }
  }

  destroy(): void {
    this.rotationBuf.destroy();
    this.stagingBuf?.destroy();
    this.paramStagingBuf?.destroy();
    for (const cache of this.caches.values()) {
      cache.polar.destroy();
      cache.qjl.destroy();
      cache.maxR.destroy();
      cache.gamma.destroy();
    }
    this.caches.clear();
  }
}
