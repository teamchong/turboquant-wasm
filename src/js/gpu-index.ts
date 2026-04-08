/**
 * TQGpuIndex — WebGPU accelerated vector search on TurboQuant compressed data.
 *
 * Uploads compressed vectors to GPU storage buffers once at init.
 * Each search dispatches a compute shader that scans all vectors in parallel.
 * CPU handles query rotation (via TurboQuant WASM); GPU handles the dot product scan.
 */

import type TurboQuant from "./index.js";

// Inline the WGSL shader source. Bun's bundler handles text imports.
import SHADER_SRC from "./shaders/tq-dot-batch.wgsl" with { type: "text" };

const HEADER_SIZE = 22;
const SQRT_PI_OVER_2 = 1.2533141;

/** Precompute the 128-entry polar decode LUT (matches Zig's pair_lut). */
function buildPolarLut(): Float32Array {
  const lut = new Float32Array(256); // 128 × vec2<f32>
  const TWO_PI = 2 * Math.PI;
  for (let c = 0; c < 128; c++) {
    const rNorm = ((c >> 3) & 0xf) / 15;
    const bucket = c & 0x7;
    const theta = (bucket / 8) * TWO_PI - Math.PI;
    lut[c * 2] = rNorm * Math.cos(theta);
    lut[c * 2 + 1] = rNorm * Math.sin(theta);
  }
  return lut;
}

/** Parse one LE u32 from a Uint8Array at the given byte offset. */
function readU32(buf: Uint8Array, off: number): number {
  return buf[off] | (buf[off + 1] << 8) | (buf[off + 2] << 16) | (buf[off + 3] << 24);
}

/** Parse one LE f32 from a Uint8Array at the given byte offset. */
function readF32(buf: Uint8Array, off: number): number {
  const tmp = new Uint8Array(4);
  tmp[0] = buf[off]; tmp[1] = buf[off + 1]; tmp[2] = buf[off + 2]; tmp[3] = buf[off + 3];
  return new Float32Array(tmp.buffer)[0];
}

/** Pad byte count up to next multiple of 4 (u32 alignment for GPU). */
function alignU32(bytes: number): number {
  return Math.ceil(bytes / 4) * 4;
}

export class TQGpuIndex {
  #device: GPUDevice;
  #pipeline: GPUComputePipeline;
  #dbBindGroup: GPUBindGroup;
  #lutBindGroup: GPUBindGroup;
  #queryBuf: GPUBuffer;
  #configBuf: GPUBuffer;
  #scoresBuf: GPUBuffer;
  #stagingBuf: GPUBuffer;
  #queryBindGroupLayout: GPUBindGroupLayout;
  #numVectors: number;
  #dim: number;
  #polarBytes: number;
  #qjlBytes: number;
  #tq: TurboQuant;

  private constructor(
    device: GPUDevice,
    pipeline: GPUComputePipeline,
    dbBindGroup: GPUBindGroup,
    lutBindGroup: GPUBindGroup,
    queryBindGroupLayout: GPUBindGroupLayout,
    queryBuf: GPUBuffer,
    configBuf: GPUBuffer,
    scoresBuf: GPUBuffer,
    stagingBuf: GPUBuffer,
    numVectors: number,
    dim: number,
    polarBytes: number,
    qjlBytes: number,
    tq: TurboQuant,
  ) {
    this.#device = device;
    this.#pipeline = pipeline;
    this.#dbBindGroup = dbBindGroup;
    this.#lutBindGroup = lutBindGroup;
    this.#queryBindGroupLayout = queryBindGroupLayout;
    this.#queryBuf = queryBuf;
    this.#configBuf = configBuf;
    this.#scoresBuf = scoresBuf;
    this.#stagingBuf = stagingBuf;
    this.#numVectors = numVectors;
    this.#dim = dim;
    this.#polarBytes = polarBytes;
    this.#qjlBytes = qjlBytes;
    this.#tq = tq;
  }

  /** Number of vectors in the index. */
  get numVectors(): number { return this.#numVectors; }

  /** Dimension of the original vectors. */
  get dim(): number { return this.#dim; }

  /**
   * Create a GPU index from concatenated TQ compressed vectors.
   * Returns null if WebGPU is unavailable.
   *
   * @param tq - TurboQuant instance (used for query rotation on CPU)
   * @param compressedConcat - All compressed vectors concatenated (same format as dotBatch)
   * @param bytesPerVector - Size of each compressed vector
   */
  static async create(
    tq: TurboQuant,
    compressedConcat: Uint8Array,
    bytesPerVector: number,
  ): Promise<TQGpuIndex | null> {
    if (typeof navigator === "undefined" || !navigator.gpu) return null;
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return null;
    const device = await adapter.requestDevice();

    const numVectors = Math.floor(compressedConcat.length / bytesPerVector);
    if (numVectors === 0) return null;

    // Parse first header for shared dimensions
    const dim = readU32(compressedConcat, 1);
    const polarBytes = readU32(compressedConcat, 6);
    const qjlBytes = readU32(compressedConcat, 10);

    const polarU32sPerVec = Math.ceil(polarBytes / 4);
    const qjlU32sPerVec = Math.ceil(qjlBytes / 4);

    // Strip headers, pack polar/qjl/params into separate contiguous arrays
    const polarPacked = new Uint8Array(numVectors * alignU32(polarBytes));
    const qjlPacked = new Uint8Array(numVectors * alignU32(qjlBytes));
    const paramsPacked = new Float32Array(numVectors * 2);

    for (let i = 0; i < numVectors; i++) {
      const off = i * bytesPerVector;
      const polarOff = off + HEADER_SIZE;
      const qjlOff = polarOff + polarBytes;

      polarPacked.set(
        compressedConcat.subarray(polarOff, polarOff + polarBytes),
        i * alignU32(polarBytes),
      );
      qjlPacked.set(
        compressedConcat.subarray(qjlOff, qjlOff + qjlBytes),
        i * alignU32(qjlBytes),
      );
      paramsPacked[i * 2] = readF32(compressedConcat, off + 14);     // max_r
      paramsPacked[i * 2 + 1] = readF32(compressedConcat, off + 18); // gamma
    }

    // Upload database to GPU (persistent storage buffers)
    const polarBuf = device.createBuffer({
      size: polarPacked.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(polarBuf, 0, polarPacked);

    const qjlBuf = device.createBuffer({
      size: qjlPacked.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(qjlBuf, 0, qjlPacked);

    const paramsBuf = device.createBuffer({
      size: paramsPacked.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(paramsBuf, 0, paramsPacked.buffer);

    // Static polar decode LUT
    const lutData = buildPolarLut();
    const lutBuf = device.createBuffer({
      size: lutData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(lutBuf, 0, lutData.buffer);

    // Per-query buffers
    const queryBuf = device.createBuffer({
      size: dim * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const configBuf = device.createBuffer({
      size: 32, // Config struct: 8 u32s
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const scoresBuf = device.createBuffer({
      size: numVectors * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const stagingBuf = device.createBuffer({
      size: numVectors * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // Shader + pipeline
    const shaderModule = device.createShaderModule({ code: SHADER_SRC });

    const dbBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      ],
    });
    const queryBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    const lutBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      ],
    });

    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [dbBindGroupLayout, queryBindGroupLayout, lutBindGroupLayout],
    });
    const pipeline = device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module: shaderModule, entryPoint: "main" },
    });

    const dbBindGroup = device.createBindGroup({
      layout: dbBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: polarBuf } },
        { binding: 1, resource: { buffer: qjlBuf } },
        { binding: 2, resource: { buffer: paramsBuf } },
      ],
    });
    const lutBindGroup = device.createBindGroup({
      layout: lutBindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: lutBuf } }],
    });

    // Write static config fields
    const configData = new ArrayBuffer(32);
    const configView = new DataView(configData);
    configView.setUint32(0, numVectors, true);
    configView.setUint32(4, dim, true);
    configView.setUint32(8, polarU32sPerVec, true);
    configView.setUint32(12, qjlU32sPerVec, true);
    // q_sum written per-query at offset 16
    device.queue.writeBuffer(configBuf, 0, configData);

    return new TQGpuIndex(
      device, pipeline, dbBindGroup, lutBindGroup, queryBindGroupLayout,
      queryBuf, configBuf, scoresBuf, stagingBuf,
      numVectors, dim, polarBytes, qjlBytes, tq,
    );
  }

  /**
   * Compute dot(query, vec[i]) for all vectors using the GPU.
   * Returns Float32Array of scores (length = numVectors).
   */
  async dotBatchGpu(query: Float32Array): Promise<Float32Array> {
    // Rotate query on CPU
    const rotated = this.#tq.rotateQuery(query);

    // Compute q_sum (sum of all rotated query components)
    let qSum = 0;
    for (let i = 0; i < rotated.length; i++) qSum += rotated[i];

    // Upload rotated query
    this.#device.queue.writeBuffer(this.#queryBuf, 0, rotated.buffer);

    // Write q_sum into config at offset 16
    const qSumBuf = new Float32Array([qSum]);
    this.#device.queue.writeBuffer(this.#configBuf, 16, qSumBuf.buffer);

    // Create per-query bind group
    const queryBindGroup = this.#device.createBindGroup({
      layout: this.#queryBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.#queryBuf } },
        { binding: 1, resource: { buffer: this.#configBuf } },
        { binding: 2, resource: { buffer: this.#scoresBuf } },
      ],
    });

    // Dispatch
    const encoder = this.#device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.#pipeline);
    pass.setBindGroup(0, this.#dbBindGroup);
    pass.setBindGroup(1, queryBindGroup);
    pass.setBindGroup(2, this.#lutBindGroup);
    pass.dispatchWorkgroups(this.#numVectors);
    pass.end();

    // Copy scores to staging for readback
    encoder.copyBufferToBuffer(
      this.#scoresBuf, 0,
      this.#stagingBuf, 0,
      this.#numVectors * 4,
    );
    this.#device.queue.submit([encoder.finish()]);

    // Read back scores
    await this.#stagingBuf.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(this.#stagingBuf.getMappedRange().slice(0));
    this.#stagingBuf.unmap();

    return result;
  }

  /** Release all GPU resources. */
  destroy(): void {
    this.#queryBuf.destroy();
    this.#configBuf.destroy();
    this.#scoresBuf.destroy();
    this.#stagingBuf.destroy();
  }
}

export default TQGpuIndex;
