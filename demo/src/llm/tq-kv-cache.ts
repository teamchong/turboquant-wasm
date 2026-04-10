/**
 * TQ Compressed KV Cache — WebGPU compute shader integration.
 *
 * Replaces Transformers.js DynamicCache with compressed storage:
 * - K/V vectors compressed to ~3 bits/dim via TQ (polar + QJL encoding)
 * - Compressed data stays on GPU as GPUBuffer
 * - Attention computed directly on compressed data (never decompress K)
 * - V decoded per-dimension during weighted sum (online softmax)
 *
 * Memory: float16 K/V cache = 2 bytes/dim → TQ = ~0.44 bytes/dim = ~4.5x savings
 * Context: 8K tokens → ~36K tokens in same GPU memory budget
 */

import encodeShaderSource from "../../../src/js/shaders/tq-kv-encode.wgsl?raw";
import attentionShaderSource from "../../../src/js/shaders/tq-kv-attention.wgsl?raw";

// Precomputed polar decode LUT (128 entries of [cos, sin])
function buildPolarLUT(): Float32Array {
  const lut = new Float32Array(256); // 128 * 2
  const THETA_LEVELS = 7.0;
  const TWO_PI = 2.0 * Math.PI;
  const PI = Math.PI;
  for (let c = 0; c < 128; c++) {
    const r_norm = ((c >> 3) & 0xF) / 15.0;
    const bucket = c & 0x7;
    const theta = (bucket / THETA_LEVELS) * TWO_PI - PI;
    lut[c * 2] = r_norm * Math.cos(theta);
    lut[c * 2 + 1] = r_norm * Math.sin(theta);
  }
  return lut;
}

interface TQLayerCache {
  kBuffer: GPUBuffer;   // compressed K: [capacity * blob_u32s_per_vec * 4] bytes
  vBuffer: GPUBuffer;   // compressed V: same layout
  length: number;       // current number of cached positions
  capacity: number;     // max positions allocated
}

export class TQKVCache {
  private device: GPUDevice;
  private encodePipeline: GPUComputePipeline;
  private attentionPipeline: GPUComputePipeline;
  private polarLutBuffer: GPUBuffer;
  private layers: Map<number, TQLayerCache> = new Map();

  // TQ encoding parameters
  private dim: number;
  private numPairs: number;
  private polarBytes: number;
  private qjlBytes: number;
  private headerBytes: number = 8; // max_r(f32) + gamma(f32)
  private blobBytes: number;
  private blobU32s: number;

  static async create(device: GPUDevice, headDim: number, maxPositions: number): Promise<TQKVCache> {
    const cache = new TQKVCache();
    cache.device = device;
    cache.dim = headDim;
    cache.numPairs = headDim / 2;
    cache.polarBytes = Math.ceil((cache.numPairs * 7) / 8) + 1; // +1 padding
    cache.qjlBytes = Math.ceil(headDim / 8);
    cache.blobBytes = cache.headerBytes + cache.polarBytes + cache.qjlBytes;
    cache.blobU32s = Math.ceil(cache.blobBytes / 4);
    cache.maxPositions = maxPositions;

    // Create polar LUT buffer
    const lutData = buildPolarLUT();
    cache.polarLutBuffer = device.createBuffer({
      size: lutData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(cache.polarLutBuffer, 0, lutData);

    // Create shader pipelines
    const encodeModule = device.createShaderModule({ code: encodeShaderSource });
    cache.encodePipeline = await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: encodeModule, entryPoint: "main" },
    });

    const attentionModule = device.createShaderModule({ code: attentionShaderSource });
    cache.attentionPipeline = await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: attentionModule, entryPoint: "main" },
    });

    return cache;
  }

  /**
   * Encode new K/V vectors and append to the compressed cache for a layer.
   * Input: float32 GPU buffer [num_new_positions, head_dim]
   * The data stays compressed on GPU.
   */
  private encodeCount = 0;
  private maxPositions: number;
  onContextLimitReached: ((used: number, max: number) => void) | null = null;

  encodeAndAppend(layerIdx: number, kData: GPUBuffer, vData: GPUBuffer, numNewPositions: number) {
    if (this.encodeCount < 3) {
      console.log(`[TQ] encode layer=${layerIdx} positions=${numNewPositions} dim=${this.dim} blobBytes=${this.blobBytes}`);
    }
    this.encodeCount++;

    let layer = this.layers.get(layerIdx);
    const currentLen = layer ? layer.length : 0;
    if (currentLen + numNewPositions > this.maxPositions) {
      this.onContextLimitReached?.(currentLen + numNewPositions, this.maxPositions);
      return;
    }
    if (!layer) {
      const capacity = 4096;
      const bufSize = capacity * this.blobU32s * 4;
      layer = {
        kBuffer: this.device.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC }),
        vBuffer: this.device.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC }),
        length: 0,
        capacity,
      };
      this.layers.set(layerIdx, layer);
    }

    // Grow buffers if needed (double capacity)
    if (layer.length + numNewPositions > layer.capacity) {
      const newCapacity = Math.max(layer.capacity * 2, layer.length + numNewPositions);
      const newBufSize = newCapacity * this.blobU32s * 4;
      const oldBufSize = layer.length * this.blobU32s * 4;
      const newK = this.device.createBuffer({ size: newBufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
      const newV = this.device.createBuffer({ size: newBufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
      if (oldBufSize > 0) {
        const copyEncoder = this.device.createCommandEncoder();
        copyEncoder.copyBufferToBuffer(layer.kBuffer, 0, newK, 0, oldBufSize);
        copyEncoder.copyBufferToBuffer(layer.vBuffer, 0, newV, 0, oldBufSize);
        this.device.queue.submit([copyEncoder.finish()]);
      }
      layer.kBuffer.destroy();
      layer.vBuffer.destroy();
      layer.kBuffer = newK;
      layer.vBuffer = newV;
      layer.capacity = newCapacity;
    }

    // Encode config
    const configData = new Uint32Array([
      this.dim,
      numNewPositions,
      this.polarBytes,
      this.qjlBytes,
      this.blobU32s,
      2, // header_u32s (8 bytes = 2 u32s)
      0, 0, // padding
    ]);
    const configBuffer = this.device.createBuffer({
      size: configData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(configBuffer, 0, configData);

    // Temporary buffer for newly encoded vectors
    const encBufSize = numNewPositions * this.blobU32s * 4;
    const kEncoded = this.device.createBuffer({ size: encBufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const vEncoded = this.device.createBuffer({ size: encBufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

    const encoder = this.device.createCommandEncoder();

    // Encode K vectors
    const kBindGroup = this.device.createBindGroup({
      layout: this.encodePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: kData } },
        { binding: 1, resource: { buffer: kEncoded } },
        { binding: 2, resource: { buffer: configBuffer } },
      ],
    });
    const kPass = encoder.beginComputePass();
    kPass.setPipeline(this.encodePipeline);
    kPass.setBindGroup(0, kBindGroup);
    kPass.dispatchWorkgroups(numNewPositions);
    kPass.end();

    // Encode V vectors
    const vBindGroup = this.device.createBindGroup({
      layout: this.encodePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: vData } },
        { binding: 1, resource: { buffer: vEncoded } },
        { binding: 2, resource: { buffer: configBuffer } },
      ],
    });
    const vPass = encoder.beginComputePass();
    vPass.setPipeline(this.encodePipeline);
    vPass.setBindGroup(0, vBindGroup);
    vPass.dispatchWorkgroups(numNewPositions);
    vPass.end();

    // Copy encoded vectors into the cache at the current position
    const dstOffset = layer.length * this.blobU32s * 4;
    encoder.copyBufferToBuffer(kEncoded, 0, layer.kBuffer, dstOffset, encBufSize);
    encoder.copyBufferToBuffer(vEncoded, 0, layer.vBuffer, dstOffset, encBufSize);

    this.device.queue.submit([encoder.finish()]);

    layer.length += numNewPositions;

    // Cleanup temp buffers
    kEncoded.destroy();
    vEncoded.destroy();
    configBuffer.destroy();
  }

  /**
   * Run TQ attention: Q @ compressed_K^T + softmax @ compressed_V
   * All on GPU. Returns a GPUBuffer with the attention output [num_q_heads, head_dim].
   */
  runAttention(
    layerIdx: number,
    queryBuffer: GPUBuffer,
    maskBuffer: GPUBuffer,
    numQHeads: number,
    numKVHeads: number,
    scale: number,
  ): GPUBuffer {
    const layer = this.layers.get(layerIdx);
    if (!layer || layer.length === 0) {
      throw new Error(`No cached KV for layer ${layerIdx}`);
    }

    const outputSize = numQHeads * this.dim * 4;
    const outputBuffer = this.device.createBuffer({
      size: outputSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const configData = new Uint32Array(12);
    const configView = new DataView(configData.buffer);
    configView.setUint32(0, numQHeads, true);
    configView.setUint32(4, numKVHeads, true);
    configView.setUint32(8, layer.length, true);
    configView.setUint32(12, this.dim, true);
    configView.setUint32(16, this.blobU32s, true);
    configView.setUint32(20, 2, true); // header_u32s
    configView.setUint32(24, this.headerBytes, true); // polar_byte_offset
    configView.setUint32(28, this.headerBytes + this.polarBytes, true); // qjl_byte_offset
    configView.setFloat32(32, scale, true);

    const configBuffer = this.device.createBuffer({
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(configBuffer, 0, configData);

    const bindGroup = this.device.createBindGroup({
      layout: this.attentionPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: queryBuffer } },
        { binding: 1, resource: { buffer: layer.kBuffer } },
        { binding: 2, resource: { buffer: layer.vBuffer } },
        { binding: 3, resource: { buffer: maskBuffer } },
        { binding: 4, resource: { buffer: outputBuffer } },
        { binding: 5, resource: { buffer: configBuffer } },
        { binding: 6, resource: { buffer: this.polarLutBuffer } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.attentionPipeline);
    pass.setBindGroup(0, bindGroup);
    // Dispatch: (ceil(dim/256), num_q_heads, 1)
    pass.dispatchWorkgroups(Math.ceil(this.dim / 256), numQHeads);
    pass.end();

    this.device.queue.submit([encoder.finish()]);
    configBuffer.destroy();

    return outputBuffer;
  }

  /**
   * Get compression stats for display.
   */
  getStats(): { layers: number; positions: number; compressedBytes: number; uncompressedBytes: number } {
    let positions = 0;
    let compressedBytes = 0;
    for (const [, layer] of this.layers) {
      positions = Math.max(positions, layer.length);
      compressedBytes += layer.length * this.blobBytes * 2; // K + V
    }
    const uncompressedBytes = positions * this.dim * 4 * 2 * this.layers.size; // float32, K+V, all layers
    return { layers: this.layers.size, positions, compressedBytes, uncompressedBytes };
  }

  destroy() {
    for (const [, layer] of this.layers) {
      layer.kBuffer.destroy();
      layer.vBuffer.destroy();
    }
    this.polarLutBuffer.destroy();
    this.layers.clear();
  }
}
