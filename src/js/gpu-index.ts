/**
 * TQGpuIndex — WebGPU accelerated vector search on TurboQuant compressed data.
 *
 * Uploads compressed vectors to GPU storage buffers once at init.
 * Each search dispatches a compute shader that scans all vectors in parallel.
 * CPU handles query rotation (via TurboQuant WASM); GPU handles the dot product scan.
 */

import type TurboQuant from "./index.js";
import SHADER_SRC from "./shaders/tq-dot-batch.wgsl" with { type: "text" };
import BRUTE_SHADER_SRC from "./shaders/brute-dot-batch.wgsl" with { type: "text" };

// Zig's @sizeOf(packed Header) = 32 due to alignment padding.
// The serialized header uses bytes 0-21, payload starts at byte 32.
const HEADER_SIZE = 32;

/** Precompute the 128-entry polar decode LUT (matches Zig's pair_lut). */
function buildPolarLut(): Float32Array {
  const lut = new Float32Array(256);
  const TWO_PI = 2 * Math.PI;
  const THETA_LEVELS = 7.0;
  for (let c = 0; c < 128; c++) {
    const rNorm = ((c >> 3) & 0xf) / 15;
    const bucket = c & 0x7;
    const theta = (bucket / THETA_LEVELS) * TWO_PI - Math.PI;
    lut[c * 2] = rNorm * Math.cos(theta);
    lut[c * 2 + 1] = rNorm * Math.sin(theta);
  }
  return lut;
}

function readU32(buf: Uint8Array, off: number): number {
  return (buf[off] | (buf[off + 1] << 8) | (buf[off + 2] << 16) | (buf[off + 3] << 24)) >>> 0;
}

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
  #blobBuf: GPUBuffer;
  #paramsBuf: GPUBuffer;
  #lutBuf: GPUBuffer;
  #queryBindGroup: GPUBindGroup;
  #numVectors: number;
  #dim: number;
  #tq: TurboQuant;
  #qSumBuf: Float32Array;
  #resultBuf: Float32Array;

  private constructor(
    device: GPUDevice, pipeline: GPUComputePipeline,
    dbBindGroup: GPUBindGroup, lutBindGroup: GPUBindGroup,
    queryBindGroup: GPUBindGroup,
    queryBuf: GPUBuffer, configBuf: GPUBuffer,
    scoresBuf: GPUBuffer, stagingBuf: GPUBuffer,
    blobBuf: GPUBuffer, paramsBuf: GPUBuffer, lutBuf: GPUBuffer,
    numVectors: number, dim: number, tq: TurboQuant,
  ) {
    this.#device = device;
    this.#pipeline = pipeline;
    this.#dbBindGroup = dbBindGroup;
    this.#lutBindGroup = lutBindGroup;
    this.#queryBindGroup = queryBindGroup;
    this.#queryBuf = queryBuf;
    this.#configBuf = configBuf;
    this.#scoresBuf = scoresBuf;
    this.#stagingBuf = stagingBuf;
    this.#blobBuf = blobBuf;
    this.#paramsBuf = paramsBuf;
    this.#lutBuf = lutBuf;
    this.#numVectors = numVectors;
    this.#dim = dim;
    this.#tq = tq;
    this.#qSumBuf = new Float32Array(1);
    this.#resultBuf = new Float32Array(numVectors);
  }

  get numVectors(): number { return this.#numVectors; }
  get dim(): number { return this.#dim; }

  /**
   * Stream compressed vectors to GPU. Accepts:
   * - Response: streams .tqv from fetch (17-byte TQV header + vectors)
   * - Uint8Array: raw concatenated compressed vectors (no TQV header, needs bytesPerVector)
   *
   * Never holds the full dataset in JS memory — one vector at a time.
   */
  static async create(
    tq: TurboQuant,
    source: Response | Uint8Array,
    bytesPerVector?: number,
  ): Promise<TQGpuIndex | null> {
    if (typeof navigator === "undefined" || !navigator.gpu) return null;
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return null;
    const device = await adapter.requestDevice();

    let reader: ReadableStreamDefaultReader<Uint8Array>;
    let numVectors: number;
    let dim: number;
    let bpv: number;
    let leftover = new Uint8Array(0);

    if (source instanceof Response) {
      // Stream from fetch — parse 17-byte TQV header first
      reader = source.body!.getReader();
      const headerBuf = new Uint8Array(17);
      let filled = 0;
      while (filled < 17) {
        const { done, value } = await reader.read();
        if (done) return null;
        const take = Math.min(17 - filled, value.length);
        headerBuf.set(value.subarray(0, take), filled);
        filled += take;
        if (value.length > take) leftover = new Uint8Array(value.subarray(take));
      }
      const hv = new DataView(headerBuf.buffer);
      if (headerBuf[0] !== 0x54 || headerBuf[1] !== 0x51 || headerBuf[2] !== 0x56) return null;
      numVectors = hv.getUint32(5, true);
      dim = hv.getUint16(9, true);
      bpv = hv.getUint16(15, true);
    } else {
      // Uint8Array — raw compressed vectors, no TQV header
      if (!bytesPerVector) return null;
      bpv = bytesPerVector;
      numVectors = Math.floor(source.length / bpv);
      dim = readU32(source, 1); // from first vector's format header
      reader = new ReadableStream<Uint8Array>({
        start(c) { c.enqueue(source); c.close(); },
      }).getReader();
    }

    if (numVectors === 0) return null;

    const blobU32sPerVec = Math.ceil(bpv / 4);
    const alignedBpv = alignU32(bpv);

    const blobBuf = device.createBuffer({
      size: numVectors * alignedBpv,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const paramsBuf = device.createBuffer({
      size: numVectors * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Batch writes: accumulate up to BATCH vectors, write to GPU in one call.
    // ~64KB batch = good balance between memory and writeBuffer overhead.
    const BATCH = Math.max(1, Math.floor(65536 / alignedBpv));
    const batchBlob = new Uint8Array(BATCH * alignedBpv);
    const batchParams = new Float32Array(BATCH * 2);
    let pending = leftover;
    let vecIdx = 0;
    let batchCount = 0;
    let polarBytes = 0;

    function flushBatch() {
      if (batchCount === 0) return;
      const blobSlice = batchBlob.subarray(0, batchCount * alignedBpv);
      device.queue.writeBuffer(blobBuf, (vecIdx - batchCount) * alignedBpv, blobSlice);
      const paramSlice = batchParams.subarray(0, batchCount * 2);
      device.queue.writeBuffer(paramsBuf, (vecIdx - batchCount) * 8, paramSlice.buffer, 0, batchCount * 8);
      batchCount = 0;
    }

    while (vecIdx < numVectors) {
      while (pending.length < bpv) {
        const { done, value } = await reader.read();
        if (done) break;
        const merged = new Uint8Array(pending.length + value.length);
        merged.set(pending);
        merged.set(value, pending.length);
        pending = merged;
      }
      if (pending.length < bpv) break;

      // Pack into batch buffer
      const bOff = batchCount * alignedBpv;
      batchBlob.fill(0, bOff, bOff + alignedBpv);
      batchBlob.set(pending.subarray(0, bpv), bOff);
      batchParams[batchCount * 2] = new DataView(pending.buffer, pending.byteOffset + 14, 4).getFloat32(0, true);
      batchParams[batchCount * 2 + 1] = new DataView(pending.buffer, pending.byteOffset + 18, 4).getFloat32(0, true);

      if (vecIdx === 0) polarBytes = readU32(pending, 6);

      pending = pending.subarray(bpv);
      vecIdx++;
      batchCount++;

      if (batchCount >= BATCH) flushBatch();
    }
    flushBatch();

    reader.releaseLock();
    if (vecIdx === 0) {
      blobBuf.destroy();
      paramsBuf.destroy();
      device.destroy();
      return null;
    }

    return TQGpuIndex.#finishCreate(
      device, tq, blobBuf, paramsBuf,
      blobU32sPerVec, polarBytes, bpv,
      vecIdx, dim,
    );
  }


  static #finishCreate(
    device: GPUDevice, tq: TurboQuant,
    blobBuf: GPUBuffer, paramsBuf: GPUBuffer,
    blobU32sPerVec: number, polarBytes: number, bytesPerVector: number,
    numVectors: number, dim: number,
  ): TQGpuIndex {
    const lutData = buildPolarLut();
    const lutBuf = device.createBuffer({
      size: lutData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(lutBuf, 0, lutData.buffer);

    const queryBuf = device.createBuffer({
      size: dim * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const configBuf = device.createBuffer({
      size: 32,
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

    const shaderModule = device.createShaderModule({ code: SHADER_SRC });

    const dbBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      ],
    });
    const queryBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    const lutBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      ],
    });

    const pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [dbBGL, queryBGL, lutBGL] }),
      compute: { module: shaderModule, entryPoint: "main" },
    });

    const dbBindGroup = device.createBindGroup({
      layout: dbBGL,
      entries: [
        { binding: 0, resource: { buffer: blobBuf } },
        { binding: 1, resource: { buffer: paramsBuf } },
      ],
    });
    const lutBindGroup = device.createBindGroup({
      layout: lutBGL,
      entries: [{ binding: 0, resource: { buffer: lutBuf } }],
    });

    const configData = new ArrayBuffer(32);
    const cv = new DataView(configData);
    cv.setUint32(0, numVectors, true);
    cv.setUint32(4, dim, true);
    cv.setUint32(8, blobU32sPerVec, true);
    cv.setUint32(12, HEADER_SIZE, true);
    cv.setUint32(16, HEADER_SIZE + polarBytes, true);
    device.queue.writeBuffer(configBuf, 0, configData);

    const queryBindGroup = device.createBindGroup({
      layout: queryBGL,
      entries: [
        { binding: 0, resource: { buffer: queryBuf } },
        { binding: 1, resource: { buffer: configBuf } },
        { binding: 2, resource: { buffer: scoresBuf } },
      ],
    });

    return new TQGpuIndex(
      device, pipeline, dbBindGroup, lutBindGroup, queryBindGroup,
      queryBuf, configBuf, scoresBuf, stagingBuf,
      blobBuf, paramsBuf, lutBuf,
      numVectors, dim, tq,
    );
  }

  async dotBatchGpu(query: Float32Array): Promise<Float32Array> {
    const rotated = this.#tq.rotateQuery(query);

    let qSum = 0;
    for (let i = 0; i < rotated.length; i++) qSum += rotated[i];

    this.#device.queue.writeBuffer(this.#queryBuf, 0, rotated.buffer);
    this.#qSumBuf[0] = qSum;
    this.#device.queue.writeBuffer(this.#configBuf, 20, this.#qSumBuf.buffer);

    const encoder = this.#device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.#pipeline);
    pass.setBindGroup(0, this.#dbBindGroup);
    pass.setBindGroup(1, this.#queryBindGroup);
    pass.setBindGroup(2, this.#lutBindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.#numVectors / 256));
    pass.end();

    encoder.copyBufferToBuffer(this.#scoresBuf, 0, this.#stagingBuf, 0, this.#numVectors * 4);
    this.#device.queue.submit([encoder.finish()]);

    await this.#stagingBuf.mapAsync(GPUMapMode.READ);
    this.#resultBuf.set(new Float32Array(this.#stagingBuf.getMappedRange()));
    this.#stagingBuf.unmap();

    return this.#resultBuf;
  }

  destroy(): void {
    this.#queryBuf.destroy();
    this.#configBuf.destroy();
    this.#scoresBuf.destroy();
    this.#stagingBuf.destroy();
    this.#blobBuf.destroy();
    this.#paramsBuf.destroy();
    this.#lutBuf.destroy();
    this.#device.destroy();
  }
}

/**
 * BruteGpuIndex — WebGPU brute-force dot product on raw f32 vectors.
 * Fair baseline comparison: same GPU dispatch overhead as TQGpuIndex.
 */
export class BruteGpuIndex {
  #device: GPUDevice;
  #pipeline: GPUComputePipeline;
  #bindGroup: GPUBindGroup;
  #queryBuf: GPUBuffer;
  #configBuf: GPUBuffer;
  #scoresBuf: GPUBuffer;
  #stagingBuf: GPUBuffer;
  #vecBuf: GPUBuffer;
  #numVectors: number;
  #resultBuf: Float32Array;

  private constructor(
    device: GPUDevice, pipeline: GPUComputePipeline, bindGroup: GPUBindGroup,
    queryBuf: GPUBuffer, configBuf: GPUBuffer,
    scoresBuf: GPUBuffer, stagingBuf: GPUBuffer,
    vecBuf: GPUBuffer,
    numVectors: number,
  ) {
    this.#device = device;
    this.#pipeline = pipeline;
    this.#bindGroup = bindGroup;
    this.#queryBuf = queryBuf;
    this.#configBuf = configBuf;
    this.#scoresBuf = scoresBuf;
    this.#stagingBuf = stagingBuf;
    this.#vecBuf = vecBuf;
    this.#numVectors = numVectors;
    this.#resultBuf = new Float32Array(numVectors);
  }

  static async create(
    device: GPUDevice, rawVectors: Float32Array, dim: number,
  ): Promise<BruteGpuIndex> {
    if (dim > 1024) throw new Error(`BruteGpuIndex: dim ${dim} exceeds shader MAX_DIM (1024)`);
    const numVectors = rawVectors.length / dim;

    const vecBuf = device.createBuffer({
      size: rawVectors.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(vecBuf, 0, rawVectors.buffer);

    const queryBuf = device.createBuffer({
      size: dim * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const configBuf = device.createBuffer({
      size: 16,
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

    const shaderModule = device.createShaderModule({ code: BRUTE_SHADER_SRC });

    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });

    const pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module: shaderModule, entryPoint: "main" },
    });

    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: vecBuf } },
        { binding: 1, resource: { buffer: queryBuf } },
        { binding: 2, resource: { buffer: configBuf } },
        { binding: 3, resource: { buffer: scoresBuf } },
      ],
    });

    const configData = new Uint32Array([numVectors, dim, 0, 0]);
    device.queue.writeBuffer(configBuf, 0, configData.buffer);

    return new BruteGpuIndex(
      device, pipeline, bindGroup, queryBuf, configBuf, scoresBuf, stagingBuf, vecBuf, numVectors,
    );
  }

  async dotBatch(query: Float32Array): Promise<Float32Array> {
    this.#device.queue.writeBuffer(this.#queryBuf, 0, query.buffer);

    const encoder = this.#device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.#pipeline);
    pass.setBindGroup(0, this.#bindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.#numVectors / 256));
    pass.end();

    encoder.copyBufferToBuffer(this.#scoresBuf, 0, this.#stagingBuf, 0, this.#numVectors * 4);
    this.#device.queue.submit([encoder.finish()]);

    await this.#stagingBuf.mapAsync(GPUMapMode.READ);
    this.#resultBuf.set(new Float32Array(this.#stagingBuf.getMappedRange()));
    this.#stagingBuf.unmap();

    return this.#resultBuf;
  }

  destroy(): void {
    this.#queryBuf.destroy();
    this.#configBuf.destroy();
    this.#scoresBuf.destroy();
    this.#stagingBuf.destroy();
    this.#vecBuf.destroy();
    this.#device.destroy();
  }
}

export default TQGpuIndex;
