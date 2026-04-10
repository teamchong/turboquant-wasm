/**
 * TQ attention replacement for ORT Web's applyAttention.
 *
 * Dispatched from inside ORT's GroupQueryAttention kernel.
 * Reads Q from ORT's GPU buffers. Compresses K/V into TQ GPU cache.
 * Computes attention scores and weighted sum directly on compressed data.
 * Writes result back to ORT's output buffer. Zero decompression.
 */

import attentionShaderSrc from "./shaders/tq-attention.wgsl?raw";
import weightedSumShaderSrc from "./shaders/tq-weighted-sum.wgsl?raw";
import encodeShaderSrc from "./shaders/tq-encode.wgsl?raw";
import rotateShaderSrc from "./shaders/tq-rotate.wgsl?raw";

interface LayerCache {
  kPolar: GPUBuffer;
  kQjl: GPUBuffer;
  kMaxR: GPUBuffer;
  kGamma: GPUBuffer;
  vPolar: GPUBuffer;
  vQjl: GPUBuffer;
  vMaxR: GPUBuffer;
  vGamma: GPUBuffer;
  length: number;
}

interface TQState {
  device: GPUDevice;
  rotationBuf: GPUBuffer;
  rotatePipeline: GPUComputePipeline;
  encodePipeline: GPUComputePipeline;
  attentionPipeline: GPUComputePipeline;
  softmaxPipeline: GPUComputePipeline;
  weightedSumPipeline: GPUComputePipeline;
  layers: Map<string, LayerCache>;
  dim: number;
  maxPositions: number;
  polarWordsPerPos: number;
  qjlWordsPerPos: number;
}

let state: TQState | null = null;

function generateRotationMatrix(dim: number, seed: number): Float32Array {
  const mat = new Float32Array(dim * dim);
  function nextRng(s: { v: bigint }): bigint {
    s.v = BigInt.asUintN(64, s.v * 1103515245n + 12345n);
    return s.v;
  }
  function randF32(s: { v: bigint }): number {
    const val = Number(nextRng(s) % (1n << 31n)) / (1 << 31);
    return val === 0 ? 0.00001 : val;
  }
  function randGaussian(s: { v: bigint }): number {
    const v1 = randF32(s);
    const v2 = randF32(s);
    return Math.sqrt(-2 * Math.log(v1)) * Math.cos(2 * Math.PI * v2);
  }
  for (let row = 0; row < dim; row++) {
    for (let col = 0; col < dim; col++) {
      mat[row * dim + col] = randGaussian({ v: BigInt(seed) + BigInt(row * 31 + col) });
    }
  }
  for (let i = 0; i < dim; i++) {
    for (let j = 0; j < i; j++) {
      let dot = 0;
      for (let k = 0; k < dim; k++) dot += mat[i * dim + k] * mat[j * dim + k];
      for (let k = 0; k < dim; k++) mat[i * dim + k] -= dot * mat[j * dim + k];
    }
    let norm = 0;
    for (let k = 0; k < dim; k++) norm += mat[i * dim + k] * mat[i * dim + k];
    norm = Math.sqrt(norm);
    if (norm > 1e-10) for (let k = 0; k < dim; k++) mat[i * dim + k] /= norm;
  }
  return mat;
}

const SOFTMAX_WGSL = `
struct Params { len: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(1)
fn softmax(@builtin(global_invocation_id) gid: vec3u) {
  let n = params.len;
  var max_val = data[0];
  for (var i = 1u; i < n; i++) { max_val = max(max_val, data[i]); }
  var sum = 0.0f;
  for (var i = 0u; i < n; i++) {
    data[i] = exp(data[i] - max_val);
    sum += data[i];
  }
  for (var i = 0u; i < n; i++) { data[i] /= sum; }
}
`;

async function initState(device: GPUDevice, dim: number, maxPositions: number): Promise<TQState> {
  const polarWordsPerPos = Math.ceil((dim / 2) * 7 / 32);
  const qjlWordsPerPos = dim / 32;

  const rotMat = generateRotationMatrix(dim, 42);
  const rotationBuf = device.createBuffer({
    size: rotMat.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(rotationBuf, 0, rotMat);

  const mkPipeline = async (code: string, entry: string) =>
    device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: device.createShaderModule({ code }), entryPoint: entry },
    });

  const [rotatePipeline, encodePipeline, attentionPipeline, softmaxPipeline, weightedSumPipeline] = await Promise.all([
    mkPipeline(rotateShaderSrc, "rotate"),
    mkPipeline(encodeShaderSrc, "encode"),
    mkPipeline(attentionShaderSrc, "compute_scores"),
    mkPipeline(SOFTMAX_WGSL, "softmax"),
    mkPipeline(weightedSumShaderSrc, "weighted_sum"),
  ]);

  return {
    device, rotationBuf, rotatePipeline, encodePipeline, attentionPipeline, softmaxPipeline, weightedSumPipeline,
    layers: new Map(), dim, maxPositions, polarWordsPerPos, qjlWordsPerPos,
  };
}

function getOrCreateLayer(s: TQState, key: string): LayerCache {
  let lc = s.layers.get(key);
  if (lc) return lc;
  const d = s.device;
  const mp = s.maxPositions;
  const buf = (size: number) => d.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  lc = {
    kPolar: buf(mp * s.polarWordsPerPos * 4), kQjl: buf(mp * s.qjlWordsPerPos * 4),
    kMaxR: buf(mp * 4), kGamma: buf(mp * 4),
    vPolar: buf(mp * s.polarWordsPerPos * 4), vQjl: buf(mp * s.qjlWordsPerPos * 4),
    vMaxR: buf(mp * 4), vGamma: buf(mp * 4),
    length: 0,
  };
  s.layers.set(key, lc);
  return lc;
}

function dispatchEncode(
  encoder: GPUCommandEncoder, s: TQState, inputBuf: GPUBuffer,
  polar: GPUBuffer, qjl: GPUBuffer, maxR: GPUBuffer, gamma: GPUBuffer, writePos: number,
) {
  const paramsBuf = s.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  s.device.queue.writeBuffer(paramsBuf, 0, new Uint32Array([s.dim, 1, s.polarWordsPerPos, s.qjlWordsPerPos, writePos, 0, 0, 0]));
  const pass = encoder.beginComputePass();
  pass.setPipeline(s.encodePipeline);
  pass.setBindGroup(0, s.device.createBindGroup({
    layout: s.encodePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 2, resource: { buffer: s.rotationBuf } },
      { binding: 3, resource: { buffer: polar } },
      { binding: 4, resource: { buffer: qjl } },
      { binding: 5, resource: { buffer: maxR } },
      { binding: 6, resource: { buffer: gamma } },
    ],
  }));
  pass.dispatchWorkgroups(1);
  pass.end();
  return paramsBuf;
}

/**
 * Replaces ORT's applyAttention. Called from the patched groupQueryAttention.
 * All KV data stored and operated on in TQ compressed format.
 */
export function tqApplyAttention(
  backend: any,
  Q: any, K: any, V: any,
  context: any,
  params: any,
  layerIndex: number,
): void {
  const device: GPUDevice = backend.device;

  // Synchronous initialization on first call — pipelines are created async but
  // we block on them via the device's queue. Subsequent calls are fast.
  if (!state) {
    // Create pipelines synchronously by pre-creating shader modules
    const dim = params.headSize;
    const polarWordsPerPos = Math.ceil((dim / 2) * 7 / 32);
    const qjlWordsPerPos = dim / 32;
    const rotMat = generateRotationMatrix(dim, 42);
    const rotationBuf = device.createBuffer({ size: rotMat.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(rotationBuf, 0, rotMat);

    const mkPipeline = (code: string, entry: string) =>
      device.createComputePipeline({
        layout: "auto",
        compute: { module: device.createShaderModule({ code }), entryPoint: entry },
      });

    state = {
      device, rotationBuf,
      rotatePipeline: mkPipeline(rotateShaderSrc, "rotate"),
      encodePipeline: mkPipeline(encodeShaderSrc, "encode"),
      attentionPipeline: mkPipeline(attentionShaderSrc, "compute_scores"),
      softmaxPipeline: mkPipeline(SOFTMAX_WGSL, "softmax"),
      weightedSumPipeline: mkPipeline(weightedSumShaderSrc, "weighted_sum"),
      layers: new Map(), dim, maxPositions: 8192, polarWordsPerPos, qjlWordsPerPos,
    };
    console.log("[TQ] GPU attention initialized, dim=" + dim);
  }

  const s = state;
  const lc = getOrCreateLayer(s, `layer_${layerIndex}`);
  const pos = lc.length;
  const dim = s.dim;
  const numPositions = pos + 1;

  const getBuffer = (tv: any): GPUBuffer => backend.gpuDataManager.get(tv.data).buffer;
  const qBuf = getBuffer(Q);
  const kBuf = getBuffer(K);
  const vBuf = getBuffer(V);

  const encoder = device.createCommandEncoder();
  const tempBuffers: GPUBuffer[] = [];

  // === ROTATE Q: rotated_q = R * Q ===
  const rotatedQBuf = device.createBuffer({ size: dim * 4, usage: GPUBufferUsage.STORAGE });
  tempBuffers.push(rotatedQBuf);
  const rotParamsBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  tempBuffers.push(rotParamsBuf);
  device.queue.writeBuffer(rotParamsBuf, 0, new Uint32Array([dim, 0, 0, 0]));
  {
    const pass = encoder.beginComputePass();
    pass.setPipeline(s.rotatePipeline);
    pass.setBindGroup(0, device.createBindGroup({
      layout: s.rotatePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: rotParamsBuf } },
        { binding: 1, resource: { buffer: qBuf } },
        { binding: 2, resource: { buffer: s.rotationBuf } },
        { binding: 3, resource: { buffer: rotatedQBuf } },
      ],
    }));
    pass.dispatchWorkgroups(Math.ceil(dim / 256));
    pass.end();
  }

  // === ENCODE K_new and V_new into compressed cache ===
  tempBuffers.push(dispatchEncode(encoder, s, kBuf, lc.kPolar, lc.kQjl, lc.kMaxR, lc.kGamma, pos));
  tempBuffers.push(dispatchEncode(encoder, s, vBuf, lc.vPolar, lc.vQjl, lc.vMaxR, lc.vGamma, pos));
  lc.length = numPositions;

  // === TQ ATTENTION SCORES: rotated_Q @ compressed_K ===
  const scoresBuf = device.createBuffer({ size: numPositions * 4, usage: GPUBufferUsage.STORAGE });
  tempBuffers.push(scoresBuf);
  const attnParamsBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  tempBuffers.push(attnParamsBuf);
  device.queue.writeBuffer(attnParamsBuf, 0, new Uint32Array([dim, numPositions, s.polarWordsPerPos, s.qjlWordsPerPos]));
  {
    const pass = encoder.beginComputePass();
    pass.setPipeline(s.attentionPipeline);
    pass.setBindGroup(0, device.createBindGroup({
      layout: s.attentionPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: attnParamsBuf } },
        { binding: 1, resource: { buffer: rotatedQBuf } },
        { binding: 2, resource: { buffer: lc.kPolar } },
        { binding: 3, resource: { buffer: lc.kQjl } },
        { binding: 4, resource: { buffer: lc.kMaxR } },
        { binding: 5, resource: { buffer: lc.kGamma } },
        { binding: 6, resource: { buffer: scoresBuf } },
      ],
    }));
    pass.dispatchWorkgroups(Math.ceil(numPositions / 256));
    pass.end();
  }

  // === SOFTMAX ===
  const softmaxParamsBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  tempBuffers.push(softmaxParamsBuf);
  device.queue.writeBuffer(softmaxParamsBuf, 0, new Uint32Array([numPositions, 0, 0, 0]));
  {
    const pass = encoder.beginComputePass();
    pass.setPipeline(s.softmaxPipeline);
    pass.setBindGroup(0, device.createBindGroup({
      layout: s.softmaxPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: softmaxParamsBuf } },
        { binding: 1, resource: { buffer: scoresBuf } },
      ],
    }));
    pass.dispatchWorkgroups(1);
    pass.end();
  }

  // === TQ WEIGHTED SUM: softmax @ compressed_V ===
  const outputBuf = device.createBuffer({ size: dim * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  tempBuffers.push(outputBuf);
  {
    const pass = encoder.beginComputePass();
    pass.setPipeline(s.weightedSumPipeline);
    pass.setBindGroup(0, device.createBindGroup({
      layout: s.weightedSumPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: attnParamsBuf } },
        { binding: 1, resource: { buffer: scoresBuf } },
        { binding: 2, resource: { buffer: lc.vPolar } },
        { binding: 3, resource: { buffer: lc.vQjl } },
        { binding: 4, resource: { buffer: lc.vMaxR } },
        { binding: 5, resource: { buffer: lc.vGamma } },
        { binding: 6, resource: { buffer: s.rotationBuf } },
        { binding: 7, resource: { buffer: outputBuf } },
      ],
    }));
    pass.dispatchWorkgroups(dim);
    pass.end();
  }

  // === COPY to ORT's output tensor ===
  const outputDataId = context.output(0, [params.batchSize, params.numHeads, params.sequenceLength, params.headSize]);
  const ortOutputBuf = getBuffer({ data: outputDataId });
  encoder.copyBufferToBuffer(outputBuf, 0, ortOutputBuf, 0, Math.min(dim * 4, ortOutputBuf.size));

  device.queue.submit([encoder.finish()]);

  // Dummy present_key/present_value — TQ manages the cache externally
  if (context.outputCount > 1) {
    context.output(1, [params.batchSize, params.kvNumHeads ?? 1, 1, params.headSize]);
  }
  if (context.outputCount > 2) {
    context.output(2, [params.batchSize, params.kvNumHeads ?? 1, 1, params.headSize]);
  }

  for (const buf of tempBuffers) buf.destroy();
}

/** Reset all layer caches (call at start of each generation). */
export function resetTqCaches(): void {
  if (state) {
    for (const lc of state.layers.values()) lc.length = 0;
  }
}
