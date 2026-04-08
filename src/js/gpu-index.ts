/**
 * TQGpuIndex — WebGPU accelerated vector search on TurboQuant compressed data.
 *
 * Uploads compressed vectors to GPU storage buffers once at init.
 * Each search dispatches a compute shader that scans all vectors in parallel.
 * CPU handles query rotation (via TurboQuant WASM); GPU handles the dot product scan.
 */

import type TurboQuant from "./index.js";
import SHADER_SRC from "./shaders/tq-dot-batch.wgsl" with { type: "text" };

// Zig's @sizeOf(packed Header) = 32 due to alignment padding.
// The serialized header uses bytes 0-21, payload starts at byte 32.
const HEADER_SIZE = 32;

const SQRT_PI_OVER_2 = 1.2533141;

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
  return buf[off] | (buf[off + 1] << 8) | (buf[off + 2] << 16) | (buf[off + 3] << 24);
}

function readF32(buf: Uint8Array, off: number): number {
  const tmp = new Uint8Array(4);
  tmp[0] = buf[off]; tmp[1] = buf[off + 1]; tmp[2] = buf[off + 2]; tmp[3] = buf[off + 3];
  return new Float32Array(tmp.buffer)[0];
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
  #queryBindGroupLayout: GPUBindGroupLayout;
  #numVectors: number;
  #dim: number;
  #tq: TurboQuant;

  private constructor(
    device: GPUDevice, pipeline: GPUComputePipeline,
    dbBindGroup: GPUBindGroup, lutBindGroup: GPUBindGroup,
    queryBindGroupLayout: GPUBindGroupLayout,
    queryBuf: GPUBuffer, configBuf: GPUBuffer,
    scoresBuf: GPUBuffer, stagingBuf: GPUBuffer,
    numVectors: number, dim: number, tq: TurboQuant,
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
    this.#tq = tq;
  }

  get numVectors(): number { return this.#numVectors; }
  get dim(): number { return this.#dim; }

  static async create(
    tq: TurboQuant, compressedConcat: Uint8Array, bytesPerVector: number,
  ): Promise<TQGpuIndex | null> {
    if (typeof navigator === "undefined" || !navigator.gpu) return null;
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return null;
    const device = await adapter.requestDevice();

    const numVectors = Math.floor(compressedConcat.length / bytesPerVector);
    if (numVectors === 0) return null;

    const dim = readU32(compressedConcat, 1);
    const polarBytes = readU32(compressedConcat, 6);

    const blobU32sPerVec = Math.ceil(bytesPerVector / 4);
    const blobPacked = new Uint8Array(numVectors * alignU32(bytesPerVector));
    const paramsPacked = new Float32Array(numVectors * 2);

    for (let i = 0; i < numVectors; i++) {
      const off = i * bytesPerVector;
      blobPacked.set(
        compressedConcat.subarray(off, off + bytesPerVector),
        i * alignU32(bytesPerVector),
      );
      paramsPacked[i * 2] = readF32(compressedConcat, off + 14);
      paramsPacked[i * 2 + 1] = readF32(compressedConcat, off + 18);
    }

    const blobBuf = device.createBuffer({
      size: blobPacked.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(blobBuf, 0, blobPacked);

    const paramsBuf = device.createBuffer({
      size: paramsPacked.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(paramsBuf, 0, paramsPacked.buffer);

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

    const dbBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
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

    const pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [dbBindGroupLayout, queryBindGroupLayout, lutBindGroupLayout],
      }),
      compute: { module: shaderModule, entryPoint: "main" },
    });

    const dbBindGroup = device.createBindGroup({
      layout: dbBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: blobBuf } },
        { binding: 1, resource: { buffer: paramsBuf } },
      ],
    });
    const lutBindGroup = device.createBindGroup({
      layout: lutBindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: lutBuf } }],
    });

    const configData = new ArrayBuffer(32);
    const configView = new DataView(configData);
    configView.setUint32(0, numVectors, true);
    configView.setUint32(4, dim, true);
    configView.setUint32(8, blobU32sPerVec, true);
    configView.setUint32(12, HEADER_SIZE, true);
    configView.setUint32(16, HEADER_SIZE + polarBytes, true);
    device.queue.writeBuffer(configBuf, 0, configData);

    return new TQGpuIndex(
      device, pipeline, dbBindGroup, lutBindGroup, queryBindGroupLayout,
      queryBuf, configBuf, scoresBuf, stagingBuf,
      numVectors, dim, tq,
    );
  }

  async dotBatchGpu(query: Float32Array): Promise<Float32Array> {
    const rotated = this.#tq.rotateQuery(query);

    let qSum = 0;
    for (let i = 0; i < rotated.length; i++) qSum += rotated[i];

    this.#device.queue.writeBuffer(this.#queryBuf, 0, rotated.buffer);
    this.#device.queue.writeBuffer(this.#configBuf, 20, new Float32Array([qSum]).buffer);

    const queryBindGroup = this.#device.createBindGroup({
      layout: this.#queryBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.#queryBuf } },
        { binding: 1, resource: { buffer: this.#configBuf } },
        { binding: 2, resource: { buffer: this.#scoresBuf } },
      ],
    });

    const encoder = this.#device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.#pipeline);
    pass.setBindGroup(0, this.#dbBindGroup);
    pass.setBindGroup(1, queryBindGroup);
    pass.setBindGroup(2, this.#lutBindGroup);
    pass.dispatchWorkgroups(this.#numVectors);
    pass.end();

    encoder.copyBufferToBuffer(this.#scoresBuf, 0, this.#stagingBuf, 0, this.#numVectors * 4);
    this.#device.queue.submit([encoder.finish()]);

    await this.#stagingBuf.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(this.#stagingBuf.getMappedRange().slice(0));
    this.#stagingBuf.unmap();

    return result;
  }

  destroy(): void {
    this.#queryBuf.destroy();
    this.#configBuf.destroy();
    this.#scoresBuf.destroy();
    this.#stagingBuf.destroy();
  }
}

export default TQGpuIndex;
