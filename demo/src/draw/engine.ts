/**
 * WebGPU Inference Engine for Gemma 4 E2B with TQ KV Cache.
 *
 * Foundation: pre-allocated buffers, uniform pool, explicit hidden ping-pong.
 * All attention uses TQ compressed KV cache — no decompression.
 */

import rmsNormSrc from "./shaders/rms-norm.wgsl?raw";
import geluSrc from "./shaders/gelu.wgsl?raw";
import ropeSrc from "./shaders/rope.wgsl?raw";
import softcapSrc from "./shaders/softcap.wgsl?raw";
import residualAddSrc from "./shaders/residual-add.wgsl?raw";
import residualAddInplaceSrc from "./shaders/residual-add-inplace.wgsl?raw";
import residualAddScaleInplaceSrc from "./shaders/residual-add-scale-inplace.wgsl?raw";
import matmulQ4kSrc from "./shaders/matmul-q4k.wgsl?raw";
import matmulQ4kGeluSrc from "./shaders/matmul-q4k-gelu.wgsl?raw";
import matmulQ4kBatchedSrc from "./shaders/matmul-q4k-batched.wgsl?raw";
import matmulQ4kBatchedFixedSrc from "./shaders/matmul-q4k-batched-fixed.wgsl?raw";
import matmulQ4kBatchedGeluSrc from "./shaders/matmul-q4k-batched-gelu.wgsl?raw";
import tqEncodeSrc from "./shaders/tq-encode.wgsl?raw";
import tqAttentionSrc from "./shaders/tq-attention.wgsl?raw";
import tqWeightedSumP1Src from "./shaders/tq-weighted-sum-p1.wgsl?raw";
import tqWeightedSumP2Src from "./shaders/tq-weighted-sum-p2.wgsl?raw";
import tqRotateSrc from "./shaders/tq-rotate.wgsl?raw";
import tqInverseRotateSrc from "./shaders/tq-inverse-rotate.wgsl?raw";
import tqDecodeSrc from "./shaders/tq-decode.wgsl?raw";
import getRowsQ4kSrc from "./shaders/get-rows-q4k.wgsl?raw";
import matmulQ6kSrc from "./shaders/matmul-q6k.wgsl?raw";
import matmulQ6kBatchedSrc from "./shaders/matmul-q6k-batched.wgsl?raw";
import mhSoftmaxSrc from "./shaders/mh-softmax.wgsl?raw";
import argmaxSrc from "./shaders/argmax.wgsl?raw";
import rmsNormResidualSrc from "./shaders/rms-norm-residual.wgsl?raw";
import scaleInplaceSrc from "./shaders/scale-inplace.wgsl?raw";
import getRowsQ5kSrc from "./shaders/get-rows-q5k.wgsl?raw";
import matmulBf16Src from "./shaders/matmul-bf16.wgsl?raw";
import matmulF32Src from "./shaders/matmul-f32.wgsl?raw";
import matmulF32GeluSrc from "./shaders/matmul-f32-gelu.wgsl?raw";
import scaleByBufSrc from "./shaders/scale-by-buf.wgsl?raw";
import logitMaskSrc from "./shaders/logit-mask.wgsl?raw";
import { injectPolarConfig, polarWordsPerPos, qjlWordsPerPos, K_POLAR_CONFIG, V_POLAR_CONFIG } from "./polar-config.js";
import type { LoadedModel } from "./model-loader.js";

// Gemma 4 E2B constants (gemma3 architecture in GGUF)
const HIDDEN_SIZE = 1536;
const NUM_HEADS = 8;
const NUM_KV_HEADS = 1;
const HEAD_DIM_SLIDING = 256;
const HEAD_DIM_GLOBAL = 512;
const NUM_LAYERS = 35;
const RMS_NORM_EPS = 1e-6;
const SLIDING_WINDOW = 512;
const ROPE_THETA_SLIDING = 10000.0;
const ROPE_THETA_GLOBAL = 1000000.0;
const LOGIT_SOFTCAP = 30.0;
const N_EMBD_PER_LAYER = 256;
const PER_LAYER_TOTAL = N_EMBD_PER_LAYER * NUM_LAYERS; // 8960 = size of per_layer_token_embd row
const MAX_FFN = 12288;
const N_LAYER_KV_FROM_START = 15; // Gemma 4 E2B: first 15 layers compute KV, rest reuse

// Returns the layer index whose KV cache should be reused by layer `il`, or -1 if `il`
// computes its own. Matches llama.cpp's Gemma4 layer_reuse_cb:
//   if il >= N_LAYER_KV_FROM_START: return N_LAYER_KV_FROM_START - (is_swa(il) ? 2 : 1)
// which gives 13 for sliding layers ≥15 and 14 for full layers ≥15.
function kvReuseLayer(il: number): number {
  if (il < N_LAYER_KV_FROM_START) return -1;
  return isGlobalLayer(il) ? (N_LAYER_KV_FROM_START - 1) : (N_LAYER_KV_FROM_START - 2);
}
const MAX_POSITIONS = 8192;
const VOCAB_SIZE = 262144;
const MAX_Q = NUM_HEADS * HEAD_DIM_GLOBAL;   // 4096
const MAX_KV = NUM_KV_HEADS * HEAD_DIM_GLOBAL; // 512

// GGML tensor type IDs for dynamic quantization routing
const GGML_Q6_K = 14;

function isGlobalLayer(layer: number): boolean {
  return (layer + 1) % 5 === 0;
}

// =============================================================================
// GPU Profiler — timestamp queries for per-pass GPU timing
// =============================================================================

class GPUProfiler {
  private querySet: GPUQuerySet;
  private resolveBuf: GPUBuffer;
  private stagingBuf: GPUBuffer;
  private slots: { cat: string; begin: number; end: number }[] = [];
  private nextIdx = 0;
  private maxCount: number;

  constructor(private device: GPUDevice, maxPasses = 1024) {
    this.maxCount = maxPasses * 2;
    this.querySet = device.createQuerySet({ type: "timestamp", count: this.maxCount });
    this.resolveBuf = device.createBuffer({
      size: this.maxCount * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    this.stagingBuf = device.createBuffer({
      size: this.maxCount * 8,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  }

  reset(): void { this.slots.length = 0; this.nextIdx = 0; }

  /** Get a compute pass descriptor with timestamp writes for the given category. */
  pass(cat: string): GPUComputePassDescriptor | undefined {
    if (this.nextIdx + 2 > this.maxCount) return undefined;
    const begin = this.nextIdx++;
    const end = this.nextIdx++;
    this.slots.push({ cat, begin, end });
    return {
      timestampWrites: {
        querySet: this.querySet,
        beginningOfPassWriteIndex: begin,
        endOfPassWriteIndex: end,
      },
    };
  }

  /** Append resolve + copy commands to the encoder (call before submit). */
  resolve(enc: GPUCommandEncoder): void {
    if (this.nextIdx === 0) return;
    enc.resolveQuerySet(this.querySet, 0, this.nextIdx, this.resolveBuf, 0);
    enc.copyBufferToBuffer(this.resolveBuf, 0, this.stagingBuf, 0, this.nextIdx * 8);
  }

  /** Start mapping the staging buffer (call after submit). */
  mapAsync(): Promise<void> {
    return this.nextIdx > 0 ? this.stagingBuf.mapAsync(GPUMapMode.READ) : Promise.resolve();
  }

  /** Read timing results after mapAsync resolves. Returns per-category totals. */
  read(): Record<string, { ms: number; count: number }> {
    if (this.nextIdx === 0) return {};
    const ts = new BigUint64Array(this.stagingBuf.getMappedRange());
    const r: Record<string, { ms: number; count: number }> = {};
    for (const s of this.slots) {
      const ms = Number(ts[s.end] - ts[s.begin]) / 1e6;
      if (!r[s.cat]) r[s.cat] = { ms: 0, count: 0 };
      r[s.cat].ms += ms;
      r[s.cat].count++;
    }
    this.stagingBuf.unmap();
    return r;
  }

  destroy(): void {
    this.querySet.destroy();
    this.resolveBuf.destroy();
    this.stagingBuf.destroy();
  }
}

// =============================================================================
// Uniform Megabuffer — one GPU buffer, one writeBuffer per token
// =============================================================================

/** All uniforms packed into a single GPU buffer. One writeBuffer per token instead of ~1000. */
class UniformMega {
  private buf: GPUBuffer;
  private cpu: Uint8Array;
  private nextOffset = 0;
  private align: number;
  private device: GPUDevice;
  // Deduplication cache: exact-content string key → existing offset. Per
  // WeInfer (ICLR '26 WebGPU paper), many dispatches within a single
  // decode step post identical uniform blocks — rms-norm's [dim, eps, 0, 0]
  // is identical across all 35 layers on the same width; matmul's
  // [n_rows, n_cols] repeats for every Q/K/V/attn_out in a layer and
  // across layers with matching shapes. Keying by exact byte content
  // (not a hash — collisions would silently return another dispatch's
  // slot) lets repeats share one slot + one writeBuffer byte range.
  private cache: Map<string, number> = new Map();

  constructor(device: GPUDevice, maxSlots: number) {
    this.device = device;
    this.align = 256; // minUniformBufferOffsetAlignment
    const size = maxSlots * this.align;
    this.buf = device.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.cpu = new Uint8Array(size);
  }

  reset(): void { this.nextOffset = 0; this.cache.clear(); }

  /** Write 16 bytes, return binding 0 entry. Deduplicates by exact value. */
  get16(data: Uint32Array): GPUBindGroupEntry {
    const key = `16|${data[0]}|${data[1]}|${data[2]}|${data[3]}`;
    const cached = this.cache.get(key);
    if (cached !== undefined) {
      return { binding: 0, resource: { buffer: this.buf, offset: cached, size: 16 } };
    }
    const offset = this.nextOffset;
    this.nextOffset += this.align;
    new Uint32Array(this.cpu.buffer, offset, 4).set(data);
    this.cache.set(key, offset);
    return { binding: 0, resource: { buffer: this.buf, offset, size: 16 } };
  }

  /** Write 32 bytes, return binding 0 entry. Deduplicates by exact value. */
  get32(data: ArrayBuffer): GPUBindGroupEntry {
    const u = new Uint32Array(data, 0, 8);
    const key = `32|${u[0]}|${u[1]}|${u[2]}|${u[3]}|${u[4]}|${u[5]}|${u[6]}|${u[7]}`;
    const cached = this.cache.get(key);
    if (cached !== undefined) {
      return { binding: 0, resource: { buffer: this.buf, offset: cached, size: 32 } };
    }
    const offset = this.nextOffset;
    this.nextOffset += this.align;
    new Uint8Array(this.cpu.buffer, offset, 32).set(new Uint8Array(data, 0, 32));
    this.cache.set(key, offset);
    return { binding: 0, resource: { buffer: this.buf, offset, size: 32 } };
  }

  /** Upload all staged data to GPU in one call. Call before queue.submit(). */
  flush(): void {
    if (this.nextOffset > 0) {
      this.device.queue.writeBuffer(this.buf, 0, this.cpu as Uint8Array<ArrayBuffer>, 0, this.nextOffset);
    }
  }
}

// =============================================================================
// Scratch Buffers
// =============================================================================

interface Scratch {
  // Layer-level (reused across layers)
  normed: GPUBuffer;         // HIDDEN_SIZE
  q: GPUBuffer;              // MAX_Q
  k: GPUBuffer;              // MAX_KV
  v: GPUBuffer;              // MAX_KV
  qNormed: GPUBuffer;        // MAX_Q
  kNormed: GPUBuffer;        // MAX_KV
  vNormed: GPUBuffer;        // MAX_KV — V after unweighted RMS norm (Gemma 4 applies this)
  onesWeight: GPUBuffer;     // MAX_KV filled with 1.0, used as weight for unweighted RMS norm
  // Per-Layer Embedding (Gemma 4)
  perLayerRaw: GPUBuffer;    // PER_LAYER_TOTAL — row from per_layer_token_embd, scaled by sqrt(256)
  perLayerProj: GPUBuffer;   // PER_LAYER_TOTAL — per_layer_model_proj @ inp_scaled, normed
  perLayerFinal: GPUBuffer;  // PER_LAYER_TOTAL — (perLayerRaw + perLayerProj) / sqrt(2)
  pleGated: GPUBuffer;       // N_EMBD_PER_LAYER — PLE inp_gate matmul output
  pleLayerSlice: GPUBuffer;  // N_EMBD_PER_LAYER — slice of inp_per_layer for this layer
  pleActivated: GPUBuffer;   // N_EMBD_PER_LAYER — GELU(gated) * inp_per_layer slice
  pleProjected: GPUBuffer;   // HIDDEN_SIZE — PLE proj matmul output
  attnOut: GPUBuffer;        // MAX_Q
  attnProj: GPUBuffer;       // HIDDEN_SIZE
  attnPostNormed: GPUBuffer; // HIDDEN_SIZE
  hidden1: GPUBuffer;        // HIDDEN_SIZE — attention residual result (dedicated, no aliasing)
  mlpNormed: GPUBuffer;      // HIDDEN_SIZE
  gate: GPUBuffer;           // MAX_FFN
  up: GPUBuffer;             // MAX_FFN
  geluOut: GPUBuffer;        // MAX_FFN
  mlpOut: GPUBuffer;         // HIDDEN_SIZE
  mlpPostNormed: GPUBuffer;  // HIDDEN_SIZE
  // TQ attention scratch (populated per layer, consumed by all NUM_HEADS at once)
  rotQ: GPUBuffer;           // NUM_HEADS × HEAD_DIM_GLOBAL — Q in TQ rotation space
  scores: GPUBuffer;         // NUM_HEADS × MAX_POSITIONS — attention scores/softmax
  rotOut: GPUBuffer;         // NUM_HEADS × HEAD_DIM_GLOBAL — weighted-sum output in rotated space
  // TQ weighted-sum staging: Phase 1 writes per-(head, dim) polar and qjl
  // sums into these, Phase 2 reads them to compute R^T @ qjl and add polar.
  wsumPolar: GPUBuffer;      // NUM_HEADS × HEAD_DIM_GLOBAL — polar sum per dim per head
  wsumQjl: GPUBuffer;        // NUM_HEADS × HEAD_DIM_GLOBAL — qjl sum per dim per head
  // Hidden ping-pong pair for layer alternation
  hiddenA: GPUBuffer;        // HIDDEN_SIZE
  hiddenB: GPUBuffer;        // HIDDEN_SIZE
  // TQ round-trip debug probes (populated only when debugDumps is active)
  kDecoded: GPUBuffer;       // MAX_KV — K reconstructed from cache for conformance probe
  vDecoded: GPUBuffer;       // MAX_KV — V reconstructed from cache for conformance probe
}

// =============================================================================
// Pipelines
// =============================================================================

interface Pipelines {
  rmsNorm: GPUComputePipeline;
  gelu: GPUComputePipeline;
  rope: GPUComputePipeline;
  softcap: GPUComputePipeline;
  matmulQ4k: GPUComputePipeline;
  matmulQ4kGelu: GPUComputePipeline;
  matmulQ4kBatched: GPUComputePipeline;
  matmulQ4kBatchedN2: GPUComputePipeline;
  matmulQ4kBatchedN4: GPUComputePipeline;
  matmulQ4kBatchedN8: GPUComputePipeline;
  matmulQ4kBatchedGeluN2: GPUComputePipeline;
  matmulQ4kBatchedGeluN4: GPUComputePipeline;
  matmulQ4kBatchedGeluN8: GPUComputePipeline;
  tqEncodeK: GPUComputePipeline;
  tqEncodeV: GPUComputePipeline;
  tqDecodeK: GPUComputePipeline;
  tqDecodeV: GPUComputePipeline;
  tqAttention: GPUComputePipeline;
  tqWeightedSumP1: GPUComputePipeline;
  tqWeightedSumP2: GPUComputePipeline;
  tqRotate: GPUComputePipeline;
  tqInverseRotate: GPUComputePipeline;
  mhSoftmax: GPUComputePipeline;
  residualAdd: GPUComputePipeline;
  residualAddInplace: GPUComputePipeline;
  residualAddScaleInplace: GPUComputePipeline;
  getRowsQ4k: GPUComputePipeline;
  matmulQ6k: GPUComputePipeline;
  matmulQ6kBatched: GPUComputePipeline;
  argmaxPass1: GPUComputePipeline;
  argmaxPass2: GPUComputePipeline;
  rmsNormResidual: GPUComputePipeline;
  scaleInplace: GPUComputePipeline;
  getRowsQ5k: GPUComputePipeline;
  matmulBf16: GPUComputePipeline;
  matmulF32: GPUComputePipeline;
  matmulF32Gelu: GPUComputePipeline;
  scaleByBuf: GPUComputePipeline;
  logitMask: GPUComputePipeline;
}

// =============================================================================
// TQ KV Cache
// =============================================================================

interface TQLayerCache {
  kPolar: GPUBuffer; kQjl: GPUBuffer; kMaxR: GPUBuffer; kGamma: GPUBuffer;
  vPolar: GPUBuffer; vQjl: GPUBuffer; vMaxR: GPUBuffer; vGamma: GPUBuffer;
  length: number;
}

// =============================================================================
// Inference Engine
// =============================================================================

export class InferenceEngine {
  device: GPUDevice;
  pipelines: Pipelines;
  model: LoadedModel;
  uniforms: UniformMega;
  scratch: Scratch;
  kvCache: Map<number, TQLayerCache> = new Map();
  tqRotBuf256: GPUBuffer;
  tqRotBuf512: GPUBuffer;
  // Transposed rotation matrices (R^T). Stored alongside R so tq-weighted-sum
  // can read R^T in stride-1 order for its R^T @ shared_qjl matmul in Phase 2
  // (otherwise it would be reading R by columns, which is stride-dim and
  // trashes the cache).
  tqRotBufT256: GPUBuffer;
  tqRotBufT512: GPUBuffer;
  tqSigns256: GPUBuffer;
  tqSigns512: GPUBuffer;
  position: number = 0;
  private profiler: GPUProfiler | null = null;
  private catMap = new Map<GPUComputePipeline, string>();
  private catOverride: string | null = null;
  lastProfile: Record<string, { ms: number; count: number }> | null = null;

  private constructor(
    device: GPUDevice, pipelines: Pipelines, model: LoadedModel,
    uniforms: UniformMega, scratch: Scratch,
    tqRotBuf256: GPUBuffer, tqRotBuf512: GPUBuffer,
    tqRotBufT256: GPUBuffer, tqRotBufT512: GPUBuffer,
    tqSigns256: GPUBuffer, tqSigns512: GPUBuffer,
    dummyFreqFactors: GPUBuffer,
  ) {
    this.device = device;
    this.pipelines = pipelines;
    this.model = model;
    this.uniforms = uniforms;
    this.scratch = scratch;
    this.tqRotBuf256 = tqRotBuf256;
    this.tqRotBuf512 = tqRotBuf512;
    this.tqRotBufT256 = tqRotBufT256;
    this.tqRotBufT512 = tqRotBufT512;
    this.tqSigns256 = tqSigns256;
    this.tqSigns512 = tqSigns512;
    this.dummyFreqFactors = dummyFreqFactors;
  }
  private dummyFreqFactors: GPUBuffer;

  /** Create engine. Device MUST have requiredFeatures: ["shader-f16"]. */
  static async create(device: GPUDevice, model: LoadedModel): Promise<InferenceEngine> {
    const mk = (src: string, entry: string) =>
      device.createComputePipeline({
        layout: "auto",
        compute: { module: device.createShaderModule({ code: src }), entryPoint: entry },
      });

    const pipelines: Pipelines = {
      rmsNorm: mk(rmsNormSrc, "rms_norm"),
      gelu: mk(geluSrc, "gelu_gate"),
      rope: mk(ropeSrc, "rope"),
      softcap: mk(softcapSrc, "softcap"),
      matmulQ4k: mk(matmulQ4kSrc, "matmul_q4k"),
      matmulQ4kGelu: mk(matmulQ4kGeluSrc, "matmul_q4k_gelu"),
      matmulQ4kBatched: mk(matmulQ4kBatchedSrc, "matmul_q4k_batched"),
      matmulQ4kBatchedN2: mk(matmulQ4kBatchedFixedSrc.replace("/*@BATCH_N@*/", "2u"), "matmul_q4k_batched_fixed"),
      matmulQ4kBatchedN4: mk(matmulQ4kBatchedFixedSrc.replace("/*@BATCH_N@*/", "4u"), "matmul_q4k_batched_fixed"),
      matmulQ4kBatchedN8: mk(matmulQ4kBatchedFixedSrc.replace("/*@BATCH_N@*/", "8u"), "matmul_q4k_batched_fixed"),
      matmulQ4kBatchedGeluN2: mk(matmulQ4kBatchedGeluSrc.replace("/*@BATCH_N@*/", "2u"), "matmul_q4k_batched_gelu"),
      matmulQ4kBatchedGeluN4: mk(matmulQ4kBatchedGeluSrc.replace("/*@BATCH_N@*/", "4u"), "matmul_q4k_batched_gelu"),
      matmulQ4kBatchedGeluN8: mk(matmulQ4kBatchedGeluSrc.replace("/*@BATCH_N@*/", "8u"), "matmul_q4k_batched_gelu"),
      tqEncodeK: mk(injectPolarConfig(tqEncodeSrc, K_POLAR_CONFIG), "encode"),
      tqEncodeV: mk(injectPolarConfig(tqEncodeSrc, V_POLAR_CONFIG), "encode"),
      tqDecodeK: mk(injectPolarConfig(tqDecodeSrc, K_POLAR_CONFIG), "decode"),
      tqDecodeV: mk(injectPolarConfig(tqDecodeSrc, V_POLAR_CONFIG), "decode"),
      tqAttention: mk(injectPolarConfig(tqAttentionSrc, K_POLAR_CONFIG), "compute_scores"),
      tqWeightedSumP1: mk(injectPolarConfig(tqWeightedSumP1Src, V_POLAR_CONFIG), "weighted_sum_p1"),
      tqWeightedSumP2: mk(tqWeightedSumP2Src, "weighted_sum_p2"),
      tqRotate: mk(tqRotateSrc, "rotate"),
      tqInverseRotate: mk(tqInverseRotateSrc, "inverse_rotate"),
      mhSoftmax: mk(mhSoftmaxSrc, "mh_softmax"),
      residualAdd: mk(residualAddSrc, "residual_add"),
      residualAddInplace: mk(residualAddInplaceSrc, "residual_add_inplace"),
      residualAddScaleInplace: mk(residualAddScaleInplaceSrc, "residual_add_scale_inplace"),
      getRowsQ4k: mk(getRowsQ4kSrc, "get_rows_q4k"),
      matmulQ6k: mk(matmulQ6kSrc, "matmul_q6k"),
      matmulQ6kBatched: mk(matmulQ6kBatchedSrc, "matmul_q6k_batched"),
      argmaxPass1: mk(argmaxSrc, "argmax_pass1"),
      argmaxPass2: mk(argmaxSrc, "argmax_pass2"),
      rmsNormResidual: mk(rmsNormResidualSrc, "rms_norm_residual"),
      scaleInplace: mk(scaleInplaceSrc, "scale_inplace"),
      getRowsQ5k: mk(getRowsQ5kSrc, "get_rows_q5k"),
      matmulBf16: mk(matmulBf16Src, "matmul_bf16"),
      matmulF32: mk(matmulF32Src, "matmul_f32"),
      matmulF32Gelu: mk(matmulF32GeluSrc, "matmul_f32_gelu"),
      scaleByBuf: mk(scaleByBufSrc, "scale_by_buf"),
      logitMask: mk(logitMaskSrc, "mask_logits"),
    };

    // TQ rotation matrices (R and R^T, both row-major).
    const mkRotBufs = (dim: number): [GPUBuffer, GPUBuffer, GPUBuffer] => {
      const mat = generateRotationMatrix(dim, 42);
      const matT = new Float32Array(dim * dim);
      for (let i = 0; i < dim; i++) {
        for (let j = 0; j < dim; j++) {
          matT[i * dim + j] = mat[j * dim + i];
        }
      }
      const buf = device.createBuffer({ size: mat.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const bufT = device.createBuffer({ size: matT.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(buf, 0, mat as Float32Array<ArrayBuffer>);
      device.queue.writeBuffer(bufT, 0, matT as Float32Array<ArrayBuffer>);
      const signs = generateHadamardSigns(dim, 42);
      const signsBuf = device.createBuffer({ size: signs.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(signsBuf, 0, signs as Int32Array<ArrayBuffer>);
      return [buf, bufT, signsBuf];
    };

    // Scratch buffers (all STORAGE | COPY_DST | COPY_SRC)
    const sbuf = (n: number) => device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const onesWeightBuf = device.createBuffer({
      size: MAX_KV * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(onesWeightBuf, 0, new Float32Array(MAX_KV).fill(1.0));
    // Sliding layers bind this buffer as freq_factors; a read is harmless because
    // the shader gates usage with `use_freq_factors` = 0 and uses the raw theta.
    const dummyFreqFactorsBuf = device.createBuffer({
      size: (HEAD_DIM_GLOBAL / 2) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(dummyFreqFactorsBuf, 0, new Float32Array(HEAD_DIM_GLOBAL / 2).fill(1.0));
    // Scratch sized for up to 8 batched slots — runLayer uses only slot 0,
    // runLayerBatched uses slots 0..N-1. 8 matches the batched matmul's
    // MAX_BATCH and the prefill CHUNK size.
    const B = 8;
    const scratch: Scratch = {
      normed: sbuf(HIDDEN_SIZE * B), q: sbuf(MAX_Q * B), k: sbuf(MAX_KV * B), v: sbuf(MAX_KV * B),
      qNormed: sbuf(MAX_Q * B), kNormed: sbuf(MAX_KV * B),
      vNormed: sbuf(MAX_KV * B),
      onesWeight: onesWeightBuf,
      perLayerRaw: sbuf(PER_LAYER_TOTAL * B),
      perLayerProj: sbuf(PER_LAYER_TOTAL * B),
      perLayerFinal: sbuf(PER_LAYER_TOTAL * B),
      pleGated: sbuf(N_EMBD_PER_LAYER * B),
      pleLayerSlice: sbuf(N_EMBD_PER_LAYER * B),
      pleActivated: sbuf(N_EMBD_PER_LAYER * B),
      pleProjected: sbuf(HIDDEN_SIZE * B),
      attnOut: sbuf(MAX_Q * B), attnProj: sbuf(HIDDEN_SIZE * B),
      attnPostNormed: sbuf(HIDDEN_SIZE * B), hidden1: sbuf(HIDDEN_SIZE * B),
      mlpNormed: sbuf(HIDDEN_SIZE * B), gate: sbuf(MAX_FFN * B), up: sbuf(MAX_FFN * B),
      geluOut: sbuf(MAX_FFN * B), mlpOut: sbuf(HIDDEN_SIZE * B), mlpPostNormed: sbuf(HIDDEN_SIZE * B),
      rotQ: sbuf(NUM_HEADS * HEAD_DIM_GLOBAL * B),
      scores: sbuf(NUM_HEADS * MAX_POSITIONS),
      rotOut: sbuf(NUM_HEADS * HEAD_DIM_GLOBAL * B),
      wsumPolar: sbuf(NUM_HEADS * HEAD_DIM_GLOBAL * B),
      wsumQjl: sbuf(NUM_HEADS * HEAD_DIM_GLOBAL * B),
      hiddenA: sbuf(HIDDEN_SIZE * B), hiddenB: sbuf(HIDDEN_SIZE * B),
      kDecoded: sbuf(MAX_KV), vDecoded: sbuf(MAX_KV),
    };

    // Decode: ~2200 dispatches per token (35 layers × ~63 each). Batch prefill packs
    // multiple tokens per submit without calling reset() in-between, so size the arena
    // for up to 16 tokens per submit = ~35200 slots. 65536 gives comfortable headroom.
    const uniforms = new UniformMega(device, 65536);

    const [rot256, rotT256, signs256] = mkRotBufs(HEAD_DIM_SLIDING);
    const [rot512, rotT512, signs512] = mkRotBufs(HEAD_DIM_GLOBAL);
    console.log("[engine] Initialized: 25 pipelines, 23 scratch buffers, 16MB uniform megabuffer");
    return new InferenceEngine(
      device, pipelines, model, uniforms, scratch,
      rot256, rot512, rotT256, rotT512,
      signs256, signs512,
      dummyFreqFactorsBuf,
    );
  }

  rotBufT(layer: number): GPUBuffer { return isGlobalLayer(layer) ? this.tqRotBufT512 : this.tqRotBufT256; }
  signsBuf(layer: number): GPUBuffer { return isGlobalLayer(layer) ? this.tqSigns512 : this.tqSigns256; }

  // ===========================================================================
  // Profiling
  // ===========================================================================

  /** Enable GPU timestamp profiling. Device must have "timestamp-query" feature. */
  enableProfiling(): void {
    if (this.profiler) return;
    this.profiler = new GPUProfiler(this.device, 1024);
    const p = this.pipelines;
    const m = this.catMap;
    m.set(p.rmsNorm, "norm");
    m.set(p.matmulQ4k, "matmul"); m.set(p.matmulQ6k, "matmul");
    m.set(p.matmulQ4kGelu, "matmul");
    m.set(p.matmulQ4kBatchedGeluN2, "matmul"); m.set(p.matmulQ4kBatchedGeluN4, "matmul"); m.set(p.matmulQ4kBatchedGeluN8, "matmul");
    m.set(p.rope, "rope");
    m.set(p.tqEncodeK, "tq_encode"); m.set(p.tqEncodeV, "tq_encode");
    m.set(p.tqRotate, "tq_rotate"); m.set(p.tqInverseRotate, "tq_invrot");
    m.set(p.tqAttention, "tq_scores");
    // Both phases roll up into the same "tq_wsum" bucket so the perf
    // baseline number stays comparable across the split.
    m.set(p.tqWeightedSumP1, "tq_wsum_p1"); m.set(p.tqWeightedSumP2, "tq_wsum_p2");
    m.set(p.mhSoftmax, "tq_softmax");
    m.set(p.gelu, "gelu");
    m.set(p.residualAdd, "residual");
    m.set(p.residualAddInplace, "residual");
    m.set(p.residualAddScaleInplace, "residual");
    m.set(p.rmsNormResidual, "norm_residual");
    m.set(p.getRowsQ4k, "embed"); m.set(p.getRowsQ5k, "embed");
    m.set(p.argmaxPass1, "argmax"); m.set(p.argmaxPass2, "argmax");
    m.set(p.matmulBf16, "ple_matmul"); m.set(p.matmulF32, "ple_matmul"); m.set(p.matmulF32Gelu, "ple_matmul");
    m.set(p.softcap, "logits"); m.set(p.scaleInplace, "scale");
    m.set(p.scaleByBuf, "scale");
    m.set(p.logitMask, "logit_mask");
    console.log("[engine] GPU profiling enabled");
  }

  disableProfiling(): void {
    this.profiler?.destroy();
    this.profiler = null;
    this.catMap.clear();
    console.log("[engine] GPU profiling disabled");
  }

  // ===========================================================================
  // Helpers
  // ===========================================================================

  weight(name: string): GPUBuffer {
    const t = this.model.tensors.get(name);
    if (!t?.gpuBuffer) throw new Error("Tensor not on GPU: " + name);
    return t.gpuBuffer;
  }

  /** Allocate 16-byte uniform, return binding 0 entry. */
  u16(data: Uint32Array): GPUBindGroupEntry { return this.uniforms.get16(data); }

  /** Allocate 32-byte uniform, return binding 0 entry. */
  u32(data: ArrayBuffer): GPUBindGroupEntry { return this.uniforms.get32(data); }

  // Bind group cache keyed by (pipeline, entry shape). Uniform offsets
  // repeat across tokens because uniforms.reset() winds the mega buffer
  // back to 0 each token, and scratch buffers are persistent at engine init.
  // So every dispatch the second token onward hits this cache — no
  // createBindGroup call, no GPU driver allocation, just a map lookup.
  private _bindGroupCache = new Map<string, GPUBindGroup>();

  private bindGroupKey(pl: GPUComputePipeline, entries: GPUBindGroupEntry[]): string {
    let key = this.pipelineIds.get(pl) ?? -1;
    let out = key.toString();
    for (const e of entries) {
      const r = e.resource as GPUBufferBinding;
      const id = this.bufferIds.get(r.buffer) ?? -1;
      out += "|" + e.binding + ":" + id + "," + (r.offset ?? 0) + "," + (r.size ?? 0);
    }
    return out;
  }

  private pipelineIds = new Map<GPUComputePipeline, number>();
  private bufferIds = new WeakMap<GPUBuffer, number>();
  private nextBufferId = 1;

  private bufferIdOf(buf: GPUBuffer): number {
    let id = this.bufferIds.get(buf);
    if (id === undefined) {
      id = this.nextBufferId++;
      this.bufferIds.set(buf, id);
    }
    return id;
  }

  // Active compute-pass tracking. Batching consecutive dispatches into one
  // pass removes ~10μs of beginComputePass/end overhead per dispatch. At
  // ~900 dispatches per decode this recovers ~9ms of the ~13ms wall-clock
  // overhead that sits on top of summed GPU kernel time. `endActivePass`
  // must be called before any non-compute encoder work (copyBufferToBuffer,
  // queue.submit, resolveQuerySet) or the encoder state goes invalid.
  //
  // When the profiler is attached we keep the old one-pass-per-category
  // behaviour so per-category timestamp accounting stays intact. In
  // production (profiler disabled) we batch aggressively.
  private _activePass: GPUComputePassEncoder | null = null;
  private _activePassEnc: GPUCommandEncoder | null = null;

  endActivePass(): void {
    if (this._activePass) {
      this._activePass.end();
      this._activePass = null;
      this._activePassEnc = null;
    }
  }

  private ensureComputePass(enc: GPUCommandEncoder, cat?: string): GPUComputePassEncoder {
    // Profiler mode: one pass per category so GPUProfiler's timestamp
    // query pair lands on a pass boundary. Lose some perf, keep measurements.
    if (this.profiler) {
      this.endActivePass();
      const desc = cat ? this.profiler.pass(cat) : undefined;
      this._activePass = enc.beginComputePass(desc);
      this._activePassEnc = enc;
      return this._activePass;
    }
    // Non-profiler: reuse the pass if it's on the same encoder. Encoder
    // changes across the worker's command-encoder boundaries (e.g. between
    // forward passes) — `_activePassEnc` tracks this so we don't accidentally
    // reuse an ended pass after a new encoder is created.
    if (this._activePass && this._activePassEnc === enc) {
      return this._activePass;
    }
    this.endActivePass();
    this._activePass = enc.beginComputePass();
    this._activePassEnc = enc;
    return this._activePass;
  }

  dispatch(enc: GPUCommandEncoder, pl: GPUComputePipeline, entries: GPUBindGroupEntry[], wgX: number, wgY = 1): void {
    // Assign ids lazily so we don't need to enumerate every pipeline at init.
    if (!this.pipelineIds.has(pl)) this.pipelineIds.set(pl, this.pipelineIds.size);
    for (const e of entries) {
      const r = e.resource as GPUBufferBinding;
      if (r.buffer) this.bufferIdOf(r.buffer);
    }

    const key = this.bindGroupKey(pl, entries);
    let bg = this._bindGroupCache.get(key);
    if (!bg) {
      bg = this.device.createBindGroup({ layout: pl.getBindGroupLayout(0), entries });
      this._bindGroupCache.set(key, bg);
    }

    const cat = this.catOverride || this.catMap.get(pl);
    const pass = this.ensureComputePass(enc, cat);
    pass.setPipeline(pl);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
  }

  /** Begin a compute pass with optional profiling timestamp. */
  /** Dispatch multiple independent operations in a single compute pass.
   *  Caller MUST ensure ops have no data dependencies on each other. */
  private multiDispatch(
    enc: GPUCommandEncoder, cat: string,
    ops: Array<{ pl: GPUComputePipeline; entries: GPUBindGroupEntry[]; wgX: number; wgY?: number }>,
  ): void {
    const pass = this.ensureComputePass(enc, cat);
    for (const op of ops) {
      if (!this.pipelineIds.has(op.pl)) this.pipelineIds.set(op.pl, this.pipelineIds.size);
      for (const e of op.entries) {
        const r = e.resource as GPUBufferBinding;
        if (r.buffer) this.bufferIdOf(r.buffer);
      }
      const key = this.bindGroupKey(op.pl, op.entries);
      let bg = this._bindGroupCache.get(key);
      if (!bg) {
        bg = this.device.createBindGroup({ layout: op.pl.getBindGroupLayout(0), entries: op.entries });
        this._bindGroupCache.set(key, bg);
      }
      pass.setPipeline(op.pl);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(op.wgX, op.wgY ?? 1);
    }
    // Don't end — next dispatch/multiDispatch will reuse this pass.
  }

  /** Build matmul dispatch parameters without dispatching. */
  private matmulOp(w: GPUBuffer, inp: GPUBuffer, out: GPUBuffer, nRows: number, nCols: number, isQ6K = false) {
    const pl = isQ6K ? this.pipelines.matmulQ6k : this.pipelines.matmulQ4k;
    const maxWG = 65535;
    // Both Q4K and Q6K kernels use 8 rows per workgroup (64 threads, 8
    // threads per row — one lane per Q4_K sub-block / Q6_K 32-weight
    // group, MLC's small-workgroup layout).
    const rowsPerWg = 8;
    const wgRows = Math.ceil(nRows / rowsPerWg);
    return {
      pl, wgX: Math.min(wgRows, maxWG), wgY: Math.ceil(wgRows / maxWG),
      entries: [
        this.u16(new Uint32Array([nRows, nCols, 0, 0])),
        { binding: 1, resource: { buffer: w } }, { binding: 2, resource: { buffer: inp } },
        { binding: 3, resource: { buffer: out } },
      ] as GPUBindGroupEntry[],
    };
  }

  rotBuf(layer: number): GPUBuffer { return isGlobalLayer(layer) ? this.tqRotBuf512 : this.tqRotBuf256; }
  headDim(layer: number): number { return isGlobalLayer(layer) ? HEAD_DIM_GLOBAL : HEAD_DIM_SLIDING; }

  /** Route matmul through Q4K or Q6K pipeline based on tensor's actual quantization type. */
  matmulAuto(enc: GPUCommandEncoder, tensorName: string, inp: GPUBuffer, out: GPUBuffer, nRows: number, nCols: number): void {
    const t = this.model.tensors.get(tensorName)!;
    if (t.type === GGML_Q6_K) {
      this.matmulQ6k(enc, t.gpuBuffer!, inp, out, nRows, nCols);
    } else {
      this.matmul(enc, t.gpuBuffer!, inp, out, nRows, nCols);
    }
  }

  // ===========================================================================
  // Shader dispatch wrappers
  // ===========================================================================

  matmul(enc: GPUCommandEncoder, w: GPUBuffer, inp: GPUBuffer, out: GPUBuffer, nRows: number, nCols: number): void {
    // Q4K kernel processes 8 output rows per workgroup (64 threads, 8
    // threads per row — MLC's small-workgroup layout).
    const wgCount = Math.ceil(nRows / 8);
    const maxWG = 65535;
    this.dispatch(enc, this.pipelines.matmulQ4k, [
      this.u16(new Uint32Array([nRows, nCols, 0, 0])),
      { binding: 1, resource: { buffer: w } }, { binding: 2, resource: { buffer: inp } },
      { binding: 3, resource: { buffer: out } },
    ], Math.min(wgCount, maxWG), Math.ceil(wgCount / maxWG));
  }

  /** Q4K matmul with GELU-gated activation fused into the input load:
   *  `input[i] = gelu(gate[i]) * up[i]`. Used for the FFN down projection
   *  when ffn_down is Q4K, replacing the sequential (gelu_gate + matmul)
   *  pair. Saves a full ffnSize-element round-trip through global memory
   *  per layer × 35 layers per decode. */
  matmulQ4KGelu(enc: GPUCommandEncoder, w: GPUBuffer, gate: GPUBuffer, up: GPUBuffer, out: GPUBuffer, nRows: number, nCols: number): void {
    const wgCount = Math.ceil(nRows / 8);
    const maxWG = 65535;
    this.dispatch(enc, this.pipelines.matmulQ4kGelu, [
      this.u16(new Uint32Array([nRows, nCols, 0, 0])),
      { binding: 1, resource: { buffer: w } },
      { binding: 2, resource: { buffer: gate } },
      { binding: 3, resource: { buffer: up } },
      { binding: 4, resource: { buffer: out } },
    ], Math.min(wgCount, maxWG), Math.ceil(wgCount / maxWG));
  }

  matmulQ6k(enc: GPUCommandEncoder, w: GPUBuffer, inp: GPUBuffer, out: GPUBuffer, nRows: number, nCols: number): void {
    // 8 rows per workgroup (matches Q4K, 64-thread small-workgroup layout)
    const wgCount = Math.ceil(nRows / 8);
    const maxWG = 65535;
    this.dispatch(enc, this.pipelines.matmulQ6k, [
      this.u16(new Uint32Array([nRows, nCols, 0, 0])),
      { binding: 1, resource: { buffer: w } }, { binding: 2, resource: { buffer: inp } },
      { binding: 3, resource: { buffer: out } },
    ], Math.min(wgCount, maxWG), Math.ceil(wgCount / maxWG));
  }

  /** Batched Q4K / Q6K matmul: one dispatch, N activation vectors, N output
   *  vectors. Weight matrix is loaded once per workgroup and reused across
   *  every batch entry, which is the whole reason this path exists — it
   *  amortizes the weight-bandwidth cost (dominant on decode-sized matmuls)
   *  across multiple tokens. Called with batchSize=1 is functionally
   *  equivalent to the unbatched path but carries loop overhead that the
   *  compiler can't always strip, so decode callers stick to the unbatched
   *  kernel. The batched kernel is for prefill (eventually) and speculative
   *  decoding (where we need to run a whole token batch through one
   *  weight load to get a real speedup over sequential single-token
   *  dispatches).
   *
   *  Max batch size is 8 — enforced by the shader's compile-time constant
   *  that sizes the shared-memory activation tile budget. Calls with
   *  batchSize > 8 throw so we don't silently corrupt memory. */
  matmulBatched(enc: GPUCommandEncoder, w: GPUBuffer, inp: GPUBuffer, out: GPUBuffer, nRows: number, nCols: number, batchSize: number, isQ6K: boolean): void {
    if (batchSize < 1 || batchSize > 8) {
      throw new Error(`matmulBatched: batchSize ${batchSize} out of range [1, 8]`);
    }
    // For Q4K, use the compile-time-N variants (fully unrolled per-batch
    // scalar accumulators) — they beat the runtime-N version by a wide
    // margin on M1 because the `sums[b]` array in the runtime version
    // spills to memory.
    let pl: GPUComputePipeline;
    if (isQ6K) {
      pl = this.pipelines.matmulQ6kBatched;
    } else if (batchSize === 2) {
      pl = this.pipelines.matmulQ4kBatchedN2;
    } else if (batchSize === 4) {
      pl = this.pipelines.matmulQ4kBatchedN4;
    } else if (batchSize === 8) {
      pl = this.pipelines.matmulQ4kBatchedN8;
    } else {
      pl = this.pipelines.matmulQ4kBatched;
    }
    const wgCount = Math.ceil(nRows / 8);
    const maxWG = 65535;
    this.dispatch(enc, pl, [
      this.u16(new Uint32Array([nRows, nCols, batchSize, 0])),
      { binding: 1, resource: { buffer: w } }, { binding: 2, resource: { buffer: inp } },
      { binding: 3, resource: { buffer: out } },
    ], Math.min(wgCount, maxWG), Math.ceil(wgCount / maxWG));
  }

  /** Batched Q4K matmul with GELU-gated activation fused into the input
   *  load. Used on the prefill path's ffn_down when the weight is Q4K.
   *  Only 2/4/8 compile-time-N variants are supported — caller must pick
   *  one (runLayerBatched only invokes with nBatch in {2,4,8} after
   *  chunk-padding). */
  matmulBatchedGelu(enc: GPUCommandEncoder, w: GPUBuffer, gate: GPUBuffer, up: GPUBuffer, out: GPUBuffer, nRows: number, nCols: number, batchSize: number): void {
    let pl: GPUComputePipeline;
    if (batchSize === 2) pl = this.pipelines.matmulQ4kBatchedGeluN2;
    else if (batchSize === 4) pl = this.pipelines.matmulQ4kBatchedGeluN4;
    else if (batchSize === 8) pl = this.pipelines.matmulQ4kBatchedGeluN8;
    else throw new Error(`matmulBatchedGelu: batchSize ${batchSize} not supported — only 2/4/8`);
    const wgCount = Math.ceil(nRows / 8);
    const maxWG = 65535;
    this.dispatch(enc, pl, [
      this.u16(new Uint32Array([nRows, nCols, batchSize, 0])),
      { binding: 1, resource: { buffer: w } },
      { binding: 2, resource: { buffer: gate } },
      { binding: 3, resource: { buffer: up } },
      { binding: 4, resource: { buffer: out } },
    ], Math.min(wgCount, maxWG), Math.ceil(wgCount / maxWG));
  }

  /** Self-test: verify the batched matmul kernel produces the same output
   *  as N sequential unbatched dispatches. Picks a real weight tensor from
   *  the loaded model, generates deterministic pseudo-random input vectors,
   *  runs both paths, reads back both results, and returns the max element-
   *  wise difference. Used by tests/matmul-batched-conformance.spec.ts.
   *
   *  Correctness constraint: the batched kernel is meant to be bit-identical
   *  to N sequential unbatched calls (same accumulation order per batch,
   *  just scheduled as one dispatch with a shared weight load). If maxDiff
   *  is non-zero the batched kernel has a bug. */
  async conformanceBatchedMatmul(weightName: string, batchSize: number): Promise<{ ok: boolean; maxDiff: number; nRows: number; nCols: number; batchSize: number; isQ6K: boolean; weightName: string }> {
    const t = this.model.tensors.get(weightName);
    if (!t) throw new Error(`conformanceBatchedMatmul: weight "${weightName}" not found`);
    if (t.dims.length !== 2) throw new Error(`conformanceBatchedMatmul: weight "${weightName}" is not 2D`);
    // GGML weight tensors use [nCols, nRows] i.e. dims[0] is the INPUT dim
    // (columns of the matrix when written as out = W @ in) and dims[1] is
    // the output dim. matmul(nRows, nCols) matches that convention.
    const nCols = t.dims[0];
    const nRows = t.dims[1];
    const isQ6K = t.type === GGML_Q6_K;

    const d = this.device;

    // Deterministic input data: small values to keep the result magnitude
    // reasonable and reduce the chance of rounding-order differences (which
    // would otherwise come out of f32 FMA non-associativity).
    const inputData = new Float32Array(batchSize * nCols);
    for (let b = 0; b < batchSize; b++) {
      for (let c = 0; c < nCols; c++) {
        // sin-based deterministic fill, amplitude ~0.01 so sums stay in a
        // numerically friendly range.
        inputData[b * nCols + c] = 0.01 * Math.sin(b * 1.7 + c * 0.13);
      }
    }

    // GPU buffers. Need COPY_SRC on the outputs so we can read them back.
    const inputBuf = d.createBuffer({ size: inputData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const outBatched = d.createBuffer({ size: batchSize * nRows * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const outSequential = d.createBuffer({ size: batchSize * nRows * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    // Per-run scratch for the unbatched dispatch (which always writes to
    // offset 0 of its output buffer).
    const outUnbatchedScratch = d.createBuffer({ size: nRows * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const stagingBatched = d.createBuffer({ size: batchSize * nRows * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const stagingSequential = d.createBuffer({ size: batchSize * nRows * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    d.queue.writeBuffer(inputBuf, 0, inputData);

    // Save and swap uniforms pool — we don't want to stomp on whatever the
    // in-flight decode/prefill state is using.
    const savedUniforms = this.uniforms;
    this.uniforms = new UniformMega(d, 256);

    // --- Sequential path: run the unbatched kernel N times, each reading
    //     from input[b*nCols..(b+1)*nCols) and writing to outSequential[b*nRows..]
    //     via a scratch + copyBufferToBuffer staging step. The unbatched
    //     kernel hardcodes offset 0 on both input and output so we need the
    //     per-batch scratch buffer. ---
    for (let b = 0; b < batchSize; b++) {
      const enc = d.createCommandEncoder();
      this.uniforms.reset();
      // Make a single-batch input slice at offset 0.
      const sliceInput = d.createBuffer({ size: nCols * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      d.queue.writeBuffer(sliceInput, 0, inputData, b * nCols, nCols);
      if (isQ6K) {
        this.matmulQ6k(enc, t.gpuBuffer!, sliceInput, outUnbatchedScratch, nRows, nCols);
      } else {
        this.matmul(enc, t.gpuBuffer!, sliceInput, outUnbatchedScratch, nRows, nCols);
      }
      this.endActivePass();
      enc.copyBufferToBuffer(outUnbatchedScratch, 0, outSequential, b * nRows * 4, nRows * 4);
      this.uniforms.flush();
      d.queue.submit([enc.finish()]);
      sliceInput.destroy();
    }

    // --- Batched path: single dispatch, all N batches through one shader
    //     invocation with a shared weight load. ---
    {
      const enc = d.createCommandEncoder();
      this.uniforms.reset();
      this.matmulBatched(enc, t.gpuBuffer!, inputBuf, outBatched, nRows, nCols, batchSize, isQ6K);
      this.endActivePass();
      this.uniforms.flush();
      d.queue.submit([enc.finish()]);
    }

    // --- Read both back ---
    {
      const enc = d.createCommandEncoder();
      enc.copyBufferToBuffer(outBatched, 0, stagingBatched, 0, batchSize * nRows * 4);
      enc.copyBufferToBuffer(outSequential, 0, stagingSequential, 0, batchSize * nRows * 4);
      d.queue.submit([enc.finish()]);
    }

    await Promise.all([
      stagingBatched.mapAsync(GPUMapMode.READ),
      stagingSequential.mapAsync(GPUMapMode.READ),
    ]);
    const batchedResult = new Float32Array(stagingBatched.getMappedRange().slice(0));
    const sequentialResult = new Float32Array(stagingSequential.getMappedRange().slice(0));
    stagingBatched.unmap();
    stagingSequential.unmap();

    let maxDiff = 0;
    for (let i = 0; i < batchedResult.length; i++) {
      const diff = Math.abs(batchedResult[i] - sequentialResult[i]);
      if (diff > maxDiff) maxDiff = diff;
    }

    // Clean up.
    inputBuf.destroy();
    outBatched.destroy();
    outSequential.destroy();
    outUnbatchedScratch.destroy();
    stagingBatched.destroy();
    stagingSequential.destroy();
    this.uniforms = savedUniforms;

    return { ok: maxDiff === 0, maxDiff, nRows, nCols, batchSize, isQ6K, weightName };
  }

  /** Microbenchmark: batched N=K dispatch vs K sequential N=1 dispatches,
   *  both doing the same total work. Returns per-iteration wall-clock time
   *  for each path so we can decide whether batching actually wins.
   *
   *  Method:
   *    - Build one command encoder containing `iters` matmuls.
   *    - Submit, await queue.onSubmittedWorkDone, measure elapsed ms.
   *    - Divide by iters to get per-dispatch cost.
   *  The batched path uses 1 dispatch per iter (producing K outputs); the
   *  sequential path uses K dispatches per iter (each producing 1 output).
   *  Both paths produce the same N*iters output vectors, so comparing the
   *  two wall-times directly answers "does batching save time per output?" */
  async benchmarkBatchedMatmul(weightName: string, batchSize: number, iters: number): Promise<{ weightName: string; batchSize: number; iters: number; nRows: number; nCols: number; isQ6K: boolean; batchedMs: number; sequentialMs: number; speedup: number }> {
    const t = this.model.tensors.get(weightName);
    if (!t) throw new Error(`benchmarkBatchedMatmul: weight "${weightName}" not found`);
    if (t.dims.length !== 2) throw new Error(`benchmarkBatchedMatmul: weight "${weightName}" is not 2D`);
    const nCols = t.dims[0];
    const nRows = t.dims[1];
    const isQ6K = t.type === GGML_Q6_K;
    const d = this.device;

    // Deterministic input data, sized for the batched dispatch (K activation
    // vectors). Sequential reuses a slice.
    const inputData = new Float32Array(batchSize * nCols);
    for (let i = 0; i < inputData.length; i++) inputData[i] = 0.01 * Math.sin(i * 0.13);

    const inputBuf = d.createBuffer({ size: inputData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const outBatched = d.createBuffer({ size: batchSize * nRows * 4, usage: GPUBufferUsage.STORAGE });
    const outSequential = d.createBuffer({ size: nRows * 4, usage: GPUBufferUsage.STORAGE });
    const sliceInput = d.createBuffer({ size: nCols * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(inputBuf, 0, inputData);
    d.queue.writeBuffer(sliceInput, 0, inputData, 0, nCols);

    const savedUniforms = this.uniforms;
    // Enough uniform space for a full encoder's worth of dispatches.
    this.uniforms = new UniformMega(d, Math.max(iters * (batchSize + 1) * 256, 64 * 1024));

    // Warmup — first dispatch on a fresh pipeline is always slower.
    {
      const enc = d.createCommandEncoder();
      this.uniforms.reset();
      if (isQ6K) this.matmulQ6k(enc, t.gpuBuffer!, sliceInput, outSequential, nRows, nCols);
      else this.matmul(enc, t.gpuBuffer!, sliceInput, outSequential, nRows, nCols);
      this.matmulBatched(enc, t.gpuBuffer!, inputBuf, outBatched, nRows, nCols, batchSize, isQ6K);
      this.endActivePass();
      this.uniforms.flush();
      d.queue.submit([enc.finish()]);
      await d.queue.onSubmittedWorkDone();
    }

    // Sequential: iters iterations, each doing K unbatched dispatches.
    const tSeq0 = performance.now();
    {
      const enc = d.createCommandEncoder();
      this.uniforms.reset();
      for (let i = 0; i < iters; i++) {
        for (let b = 0; b < batchSize; b++) {
          if (isQ6K) this.matmulQ6k(enc, t.gpuBuffer!, sliceInput, outSequential, nRows, nCols);
          else this.matmul(enc, t.gpuBuffer!, sliceInput, outSequential, nRows, nCols);
        }
      }
      this.endActivePass();
      this.uniforms.flush();
      d.queue.submit([enc.finish()]);
      await d.queue.onSubmittedWorkDone();
    }
    const sequentialMs = performance.now() - tSeq0;

    // Batched: iters iterations, each doing 1 batched dispatch.
    const tBat0 = performance.now();
    {
      const enc = d.createCommandEncoder();
      this.uniforms.reset();
      for (let i = 0; i < iters; i++) {
        this.matmulBatched(enc, t.gpuBuffer!, inputBuf, outBatched, nRows, nCols, batchSize, isQ6K);
      }
      this.endActivePass();
      this.uniforms.flush();
      d.queue.submit([enc.finish()]);
      await d.queue.onSubmittedWorkDone();
    }
    const batchedMs = performance.now() - tBat0;

    inputBuf.destroy();
    outBatched.destroy();
    outSequential.destroy();
    sliceInput.destroy();
    this.uniforms = savedUniforms;

    return {
      weightName, batchSize, iters, nRows, nCols, isQ6K,
      batchedMs, sequentialMs, speedup: sequentialMs / batchedMs,
    };
  }

  rmsNorm(enc: GPUCommandEncoder, inp: GPUBuffer, w: GPUBuffer, out: GPUBuffer, n: number, nRows = 1): void {
    const p = new ArrayBuffer(16);
    new Uint32Array(p)[0] = n;
    new Float32Array(p)[1] = RMS_NORM_EPS;
    this.dispatch(enc, this.pipelines.rmsNorm, [
      this.u16(new Uint32Array(p)),
      { binding: 1, resource: { buffer: inp } }, { binding: 2, resource: { buffer: w } },
      { binding: 3, resource: { buffer: out } },
    ], nRows);
  }

  /** Fused RMSNorm + residual add: output = residual + rmsNorm(input, weight).
   *  `nRows` lets batched decode process N rows in one dispatch (one workgroup
   *  per row). The shader reads `row_offset = wid.x * n` so nRows>1 naturally
   *  handles consecutive rows of a stacked batched buffer. */
  rmsNormResAdd(enc: GPUCommandEncoder, inp: GPUBuffer, w: GPUBuffer, residual: GPUBuffer, out: GPUBuffer, n: number, nRows = 1): void {
    const p = new ArrayBuffer(16);
    new Uint32Array(p)[0] = n;
    new Float32Array(p)[1] = RMS_NORM_EPS;
    this.dispatch(enc, this.pipelines.rmsNormResidual, [
      this.u16(new Uint32Array(p)),
      { binding: 1, resource: { buffer: inp } }, { binding: 2, resource: { buffer: w } },
      { binding: 3, resource: { buffer: residual } }, { binding: 4, resource: { buffer: out } },
    ], nRows);
  }

  residualAdd(enc: GPUCommandEncoder, a: GPUBuffer, b: GPUBuffer, out: GPUBuffer, n: number): void {
    const wg = Math.ceil(n / 256);
    this.dispatch(enc, this.pipelines.residualAdd, [
      this.u16(new Uint32Array([n, 0, 0, 0])),
      { binding: 1, resource: { buffer: a } }, { binding: 2, resource: { buffer: b } },
      { binding: 3, resource: { buffer: out } },
    ], Math.min(wg, 65535), Math.ceil(wg / 65535));
  }

  // ===========================================================================
  // KV Cache
  // ===========================================================================

  getCache(layer: number): TQLayerCache {
    let c = this.kvCache.get(layer);
    if (c) return c;
    const dim = this.headDim(layer);
    const kPolarWords = polarWordsPerPos(dim, K_POLAR_CONFIG);
    const vPolarWords = polarWordsPerPos(dim, V_POLAR_CONFIG);
    const kQjlWords = qjlWordsPerPos(dim, K_POLAR_CONFIG);
    const vQjlWords = qjlWordsPerPos(dim, V_POLAR_CONFIG);
    const buf = (n: number) => this.device.createBuffer({
      size: n, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    c = {
      kPolar: buf(MAX_POSITIONS * kPolarWords * 4), kQjl: buf(MAX_POSITIONS * kQjlWords * 4),
      kMaxR: buf(MAX_POSITIONS * 4), kGamma: buf(MAX_POSITIONS * 4),
      vPolar: buf(MAX_POSITIONS * vPolarWords * 4), vQjl: buf(MAX_POSITIONS * vQjlWords * 4),
      vMaxR: buf(MAX_POSITIONS * 4), vGamma: buf(MAX_POSITIONS * 4),
      length: 0,
    };
    this.kvCache.set(layer, c);
    return c;
  }

  resetCache(): void {
    for (const c of this.kvCache.values()) c.length = 0;
    this.position = 0;
  }

  /** Snapshot current KV cache state (GPU→GPU copy). Used to cache system prompt prefill. */
  snapshotCache(): void {
    const d = this.device;
    const enc = d.createCommandEncoder();
    this._snapshot = new Map();
    this._snapshotPosition = this.position;
    for (const [layer, c] of this.kvCache) {
      const bufs: GPUBuffer[] = [c.kPolar, c.kQjl, c.kMaxR, c.kGamma, c.vPolar, c.vQjl, c.vMaxR, c.vGamma];
      const copies: GPUBuffer[] = [];
      for (const src of bufs) {
        const dst = d.createBuffer({ size: src.size, usage: src.usage });
        enc.copyBufferToBuffer(src, 0, dst, 0, src.size);
        copies.push(dst);
      }
      this._snapshot.set(layer, { bufs: copies, length: c.length });
    }
    d.queue.submit([enc.finish()]);
    console.log(`[engine] KV snapshot: ${this._snapshot.size} layers, position=${this._snapshotPosition}`);
  }

  /** Restore KV cache from snapshot (GPU→GPU copy). Fast — no recomputation. */
  restoreCache(): void {
    if (!this._snapshot) { this.resetCache(); return; }
    const d = this.device;
    const enc = d.createCommandEncoder();
    for (const [layer, snap] of this._snapshot) {
      const c = this.getCache(layer);
      const bufs = [c.kPolar, c.kQjl, c.kMaxR, c.kGamma, c.vPolar, c.vQjl, c.vMaxR, c.vGamma];
      for (let i = 0; i < bufs.length; i++) {
        enc.copyBufferToBuffer(snap.bufs[i], 0, bufs[i], 0, bufs[i].size);
      }
      c.length = snap.length;
    }
    d.queue.submit([enc.finish()]);
    this.position = this._snapshotPosition!;
  }

  private _snapshot: Map<number, { bufs: GPUBuffer[]; length: number }> | null = null;
  private _snapshotPosition: number | null = null;

  // ===========================================================================
  // Build-time KV cache: dump / load
  // ===========================================================================

  /**
   * Serialize the current KV cache to a binary blob suitable for bundling.
   *
   * Intended as a build-time step: run the engine on a fixed system prompt,
   * dumpCache() the result, commit the binary, then loadCache() at runtime to
   * skip the ~3 minute prefill. Keyed to the TQ polar config (K_POLAR_CONFIG /
   * V_POLAR_CONFIG) and the token count — a mismatch at load time throws.
   *
   * Binary layout (little-endian, u32 unless noted):
   *   [0..4]  magic = "TQKV" (0x564B5154)
   *   [4..8]  version = 1
   *   [8..12] cached position count
   *   [12..16] number of layer records
   *   [16..20] FNV-1a hash of the system prompt token ids
   *   [20..24] K polar pair bits
   *   [24..28] K qjl bits
   *   [28..32] V polar pair bits
   *   [32..36] V qjl bits
   *   [36..64] reserved
   * Per layer (sorted by layer index):
   *   [0..4]  layer index
   *   [4..8]  head dim
   *   [8..12] length (must match header length)
   *   [12..16] reserved
   *   [16..]  kPolar / kQjl / kMaxR / kGamma / vPolar / vQjl / vMaxR / vGamma
   *           compacted to [word 0 pos 0..L-1][word 1 pos 0..L-1]...
   *           (the GPU layout is [word][pos] with stride MAX_POSITIONS,
   *            we drop the unused tail so the file size scales with `length`)
   */
  async dumpCache(systemTokenIds: number[]): Promise<Uint8Array> {
    const d = this.device;
    const layers = Array.from(this.kvCache.entries()).sort((a, b) => a[0] - b[0]);
    if (layers.length === 0) throw new Error("dumpCache: KV cache is empty — run prefill first");
    const length = layers[0][1].length;
    for (const [, c] of layers) {
      if (c.length !== length) {
        throw new Error(`dumpCache: layer lengths differ (${c.length} vs ${length}) — cache is inconsistent`);
      }
    }

    // Header + layer records, sized up-front
    const HEADER_BYTES = 64;
    const LAYER_HEADER_BYTES = 16;
    let total = HEADER_BYTES;
    for (const [layer] of layers) {
      const dim = this.headDim(layer);
      const kp = polarWordsPerPos(dim, K_POLAR_CONFIG);
      const kq = qjlWordsPerPos(dim, K_POLAR_CONFIG);
      const vp = polarWordsPerPos(dim, V_POLAR_CONFIG);
      const vq = qjlWordsPerPos(dim, V_POLAR_CONFIG);
      total += LAYER_HEADER_BYTES + length * (kp + kq + vp + vq + 4) * 4;
    }
    const out = new Uint8Array(total);
    const outU32 = new Uint32Array(out.buffer);

    const kPairBits = K_POLAR_CONFIG.radiusBits + K_POLAR_CONFIG.angleBits;
    const vPairBits = V_POLAR_CONFIG.radiusBits + V_POLAR_CONFIG.angleBits;
    outU32[0] = 0x564B5154;                   // "TQKV"
    outU32[1] = 1;                            // version
    outU32[2] = length;
    outU32[3] = layers.length;
    outU32[4] = fnv1aU32(systemTokenIds);
    outU32[5] = kPairBits;
    outU32[6] = K_POLAR_CONFIG.qjlBits;
    outU32[7] = vPairBits;
    outU32[8] = V_POLAR_CONFIG.qjlBits;
    // Rest of header is zeroed (reserved).

    let cursor = HEADER_BYTES;
    for (const [layerIdx, cache] of layers) {
      const dim = this.headDim(layerIdx);
      const hdr = new Uint32Array(out.buffer, cursor, 4);
      hdr[0] = layerIdx;
      hdr[1] = dim;
      hdr[2] = length;
      hdr[3] = 0;
      cursor += LAYER_HEADER_BYTES;

      const bufs: Array<{ src: GPUBuffer; wordsPerPos: number }> = [
        { src: cache.kPolar, wordsPerPos: polarWordsPerPos(dim, K_POLAR_CONFIG) },
        { src: cache.kQjl,   wordsPerPos: qjlWordsPerPos(dim, K_POLAR_CONFIG) },
        { src: cache.kMaxR,  wordsPerPos: 1 },
        { src: cache.kGamma, wordsPerPos: 1 },
        { src: cache.vPolar, wordsPerPos: polarWordsPerPos(dim, V_POLAR_CONFIG) },
        { src: cache.vQjl,   wordsPerPos: qjlWordsPerPos(dim, V_POLAR_CONFIG) },
        { src: cache.vMaxR,  wordsPerPos: 1 },
        { src: cache.vGamma, wordsPerPos: 1 },
      ];

      for (const { src, wordsPerPos } of bufs) {
        // Copy the full buffer (MAX_POSITIONS positions) to a staging buffer,
        // then extract only the first `length` positions per word into the
        // output blob. MAX_POSITIONS is large but this runs once at build time.
        const byteLen = wordsPerPos * MAX_POSITIONS * 4;
        const staging = d.createBuffer({
          size: byteLen,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        const enc = d.createCommandEncoder();
        enc.copyBufferToBuffer(src, 0, staging, 0, byteLen);
        d.queue.submit([enc.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const full = new Uint32Array(staging.getMappedRange().slice(0));
        staging.unmap();
        staging.destroy();

        const outView = new Uint32Array(out.buffer, cursor, wordsPerPos * length);
        for (let w = 0; w < wordsPerPos; w++) {
          const rowStart = w * MAX_POSITIONS;
          const packStart = w * length;
          for (let p = 0; p < length; p++) outView[packStart + p] = full[rowStart + p];
        }
        cursor += wordsPerPos * length * 4;
      }
    }

    if (cursor !== total) throw new Error(`dumpCache: size mismatch ${cursor} vs ${total}`);
    return out;
  }

  /**
   * Load a KV cache blob produced by dumpCache(). Populates kvCache buffers
   * directly via queue.writeBuffer and sets this.position to the cached
   * length. Throws if the magic/version/polar bits don't match the current
   * build, or if the system prompt hash differs from the one baked in.
   */
  loadCache(blob: ArrayBuffer, systemTokenIds: number[]): void {
    const d = this.device;
    const u32 = new Uint32Array(blob);
    if (u32[0] !== 0x564B5154) throw new Error("loadCache: bad magic (not a TQKV blob)");
    if (u32[1] !== 1) throw new Error(`loadCache: version ${u32[1]} unsupported`);
    const length = u32[2];
    const numLayers = u32[3];
    const expectedHash = u32[4];
    const kPairBits = u32[5];
    const kQjlBits = u32[6];
    const vPairBits = u32[7];
    const vQjlBits = u32[8];

    const gotKPair = K_POLAR_CONFIG.radiusBits + K_POLAR_CONFIG.angleBits;
    const gotVPair = V_POLAR_CONFIG.radiusBits + V_POLAR_CONFIG.angleBits;
    if (kPairBits !== gotKPair || kQjlBits !== K_POLAR_CONFIG.qjlBits ||
        vPairBits !== gotVPair || vQjlBits !== V_POLAR_CONFIG.qjlBits) {
      throw new Error(
        `loadCache: polar config mismatch — cache has K=${kPairBits}+${kQjlBits} V=${vPairBits}+${vQjlBits}, ` +
        `engine wants K=${gotKPair}+${K_POLAR_CONFIG.qjlBits} V=${gotVPair}+${V_POLAR_CONFIG.qjlBits}. ` +
        `Rebuild the cache.`,
      );
    }
    const gotHash = fnv1aU32(systemTokenIds);
    if (gotHash !== expectedHash) {
      throw new Error(
        `loadCache: system prompt hash mismatch (cache=0x${expectedHash.toString(16)}, ` +
        `current=0x${gotHash.toString(16)}). Rebuild the cache.`,
      );
    }

    const HEADER_BYTES = 64;
    const LAYER_HEADER_BYTES = 16;
    let cursor = HEADER_BYTES;
    for (let i = 0; i < numLayers; i++) {
      const hdr = new Uint32Array(blob, cursor, 4);
      const layerIdx = hdr[0];
      const dim = hdr[1];
      const layerLength = hdr[2];
      cursor += LAYER_HEADER_BYTES;
      if (layerLength !== length) {
        throw new Error(`loadCache: layer ${layerIdx} length ${layerLength} != header length ${length}`);
      }
      if (dim !== this.headDim(layerIdx)) {
        throw new Error(`loadCache: layer ${layerIdx} dim ${dim} != engine ${this.headDim(layerIdx)}`);
      }

      const cache = this.getCache(layerIdx);
      const bufs: Array<{ dst: GPUBuffer; wordsPerPos: number }> = [
        { dst: cache.kPolar, wordsPerPos: polarWordsPerPos(dim, K_POLAR_CONFIG) },
        { dst: cache.kQjl,   wordsPerPos: qjlWordsPerPos(dim, K_POLAR_CONFIG) },
        { dst: cache.kMaxR,  wordsPerPos: 1 },
        { dst: cache.kGamma, wordsPerPos: 1 },
        { dst: cache.vPolar, wordsPerPos: polarWordsPerPos(dim, V_POLAR_CONFIG) },
        { dst: cache.vQjl,   wordsPerPos: qjlWordsPerPos(dim, V_POLAR_CONFIG) },
        { dst: cache.vMaxR,  wordsPerPos: 1 },
        { dst: cache.vGamma, wordsPerPos: 1 },
      ];
      for (const { dst, wordsPerPos } of bufs) {
        // Packed layout is [word 0 pos 0..L-1][word 1 pos 0..L-1]... so each word
        // gets written at its [word][pos] stride-MAX_POSITIONS slot in the GPU
        // buffer one row at a time.
        const packedWords = wordsPerPos * length;
        const packed = new Uint32Array(blob, cursor, packedWords);
        for (let w = 0; w < wordsPerPos; w++) {
          const dstOffsetBytes = w * MAX_POSITIONS * 4;
          const rowSrc = packed.subarray(w * length, (w + 1) * length);
          d.queue.writeBuffer(dst, dstOffsetBytes, rowSrc);
        }
        cursor += packedWords * 4;
      }
      cache.length = length;
    }

    this.position = length;
    console.log(`[engine] loadCache: ${numLayers} layers, position=${length}`);
  }

  // ===========================================================================
  // Multi-branch KV: mount a pre-baked branch as the active system cache
  // ===========================================================================
  //
  // v1 flat mount: a branch IS the system cache. mountKV(name) wipes any
  // tokens past the previous system cache and replaces the system KV with
  // the named branch's pre-baked KV. After mount, engine.position equals
  // the branch's token count — caller is responsible for re-prefilling the
  // user's turn + any seed tokens they want the model to see before
  // resuming decode.
  //
  // Why no "splice onto the end of router" in v1: a branch's K was RoPE-
  // encoded for positions [0, branch_len). Mounting it at position P > 0
  // would make those K vectors point at the wrong positional phase, so
  // attention would be wrong. v2 adds a re-RoPE pass (#102) that enables
  // arbitrary-position mount and nested sub-shells.

  private branches = new Map<string, { blob: ArrayBuffer; tokenIds: number[] }>();
  private _activeBranch: string | null = null;

  /** Get the name of the branch currently mounted as the system cache,
   *  or null if no branch has been mounted (e.g. plain loadCache path). */
  get activeBranch(): string | null { return this._activeBranch; }

  /** Make a branch mountable later. Call once per branch at worker init,
   *  after parsing the multi-branch system-cache container. Stores the
   *  blob + tokens in CPU memory; does NOT touch the GPU until mountKV()
   *  is called. */
  registerBranch(name: string, blob: ArrayBuffer, tokenIds: number[]): void {
    this.branches.set(name, { blob, tokenIds });
  }

  /** Mount the named branch as the active system cache. Equivalent to
   *  loadCache(branch.blob, branch.tokenIds) plus bookkeeping. Throws if
   *  the branch is not registered.
   *
   *  Side-effects:
   *    - All GPU KV cache buffers are overwritten with the branch's bytes
   *    - engine.position is set to the branch's token count
   *    - engine.activeBranch is set to `name`
   *    - snapshot state (if any) is invalidated — caller should re-snapshot
   *      if they want restoreCache() to return to the new branch */
  mountKV(name: string): void {
    const entry = this.branches.get(name);
    if (!entry) {
      const available = [...this.branches.keys()].join(", ") || "(none registered)";
      throw new Error(`mountKV: unknown branch "${name}". Registered: ${available}`);
    }
    this.loadCache(entry.blob, entry.tokenIds);
    this._activeBranch = name;
  }

  /** List registered branch names (in insertion order). Useful for
   *  diagnostics and for the worker's "which branches did we load?" log. */
  get registeredBranches(): string[] { return [...this.branches.keys()]; }

  // ===========================================================================
  // Run one decoder layer
  // ===========================================================================

  runLayer(enc: GPUCommandEncoder, layer: number, inputBuf: GPUBuffer, outputBuf: GPUBuffer, position: number): void {
    const s = this.scratch;
    const dim = this.headDim(layer);
    const isGlobal = isGlobalLayer(layer);
    const rot = this.rotBuf(layer);
    // Gemma 4 shared-KV: layers 15..34 reuse cache from layer 13 (sliding) or 14 (global)
    // and must NOT compute or store fresh K/V. Q is still computed locally.
    const reuseLayer = kvReuseLayer(layer);
    const kvCacheLayer = reuseLayer >= 0 ? reuseLayer : layer;
    const cache = this.getCache(kvCacheLayer);
    const kPolarWords = polarWordsPerPos(dim, K_POLAR_CONFIG);
    const vPolarWords = polarWordsPerPos(dim, V_POLAR_CONFIG);
    const kQjlWords = qjlWordsPerPos(dim, K_POLAR_CONFIG);
    const vQjlWords = qjlWordsPerPos(dim, V_POLAR_CONFIG);
    // Gemma 4: rope_dimension_count = full head dim for both sliding (256) and
    // global (512). Global layers additionally apply freq_factors (proportional
    // RoPE) from rope_freqs.weight.
    const ropeDims = dim;
    const ropeTheta = isGlobal ? ROPE_THETA_GLOBAL : ROPE_THETA_SLIDING;
    const qSize = NUM_HEADS * dim;
    const kvSize = NUM_KV_HEADS * dim;
    const L = "blk." + layer + ".";

    const mkRopeParams = (nHeads: number): ArrayBuffer => {
      const buf = new ArrayBuffer(32);
      const u = new Uint32Array(buf);
      new Float32Array(buf)[2] = ropeTheta;
      u[0] = dim; u[1] = position; u[3] = ropeDims; u[4] = nHeads; u[5] = 1;
      u[6] = isGlobal ? 1 : 0; // use_freq_factors — global layers use rope_freqs
      return buf;
    };
    const ropeFreqsBuf = isGlobal
      ? this.weight("rope_freqs.weight")
      : this.dummyFreqFactors;

    const dumpL0 = layer === 0;

    // 1. Pre-attention norm
    this.rmsNorm(enc, inputBuf, this.weight(L + "attn_norm.weight"), s.normed, HIDDEN_SIZE);
    if (dumpL0) this.dumpForDebug(enc, s.normed, HIDDEN_SIZE, "attn_norm-0");

    // 2. Q/K/V projections. Shared-KV layers (15..34 in Gemma 4 E2B)
    //    discard their K/V outputs — the cache was populated by the source
    //    layer (13/14) and step 5 below guards tq_encode with `reuseLayer
    //    < 0`. Skip the K and V matmuls here too. Each is ~1/4 of Q's
    //    compute (kvSize=512 vs qSize=2048); saving 40 of them per step
    //    (20 shared layers × 2 projections) takes a measurable slice out
    //    of the matmul budget.
    const vTensor = this.model.tensors.get(L + "attn_v.weight")!;
    if (reuseLayer < 0) {
      this.multiDispatch(enc, "matmul", [
        this.matmulOp(this.weight(L + "attn_q.weight"), s.normed, s.q, qSize, HIDDEN_SIZE),
        this.matmulOp(this.weight(L + "attn_k.weight"), s.normed, s.k, kvSize, HIDDEN_SIZE),
        this.matmulOp(vTensor.gpuBuffer!, s.normed, s.v, kvSize, HIDDEN_SIZE, vTensor.type === GGML_Q6_K),
      ]);
    } else {
      this.matmul(enc, this.weight(L + "attn_q.weight"), s.normed, s.q, qSize, HIDDEN_SIZE);
    }
    if (dumpL0) {
      this.dumpForDebug(enc, s.q, qSize, "Qcur-0");
      if (reuseLayer < 0) {
        this.dumpForDebug(enc, s.k, kvSize, "Kcur-0");
        this.dumpForDebug(enc, s.v, kvSize, "Vcur-0");
      }
    }

    // 3. QK-norm + V-norm. Shared-KV layers skip K-norm and V-norm.
    //    Gemma 4 normalizes V with unweighted RMS (ggml_rms_norm without
    //    weight), so we pass a buffer of ones as the "weight".
    {
      const normP = new ArrayBuffer(16);
      new Uint32Array(normP)[0] = dim;
      new Float32Array(normP)[1] = RMS_NORM_EPS;
      if (reuseLayer < 0) {
        this.multiDispatch(enc, "norm", [
          { pl: this.pipelines.rmsNorm, wgX: NUM_HEADS, entries: [
            this.u16(new Uint32Array(normP)),
            { binding: 1, resource: { buffer: s.q } }, { binding: 2, resource: { buffer: this.weight(L + "attn_q_norm.weight") } },
            { binding: 3, resource: { buffer: s.qNormed } },
          ]},
          { pl: this.pipelines.rmsNorm, wgX: NUM_KV_HEADS, entries: [
            this.u16(new Uint32Array(normP)),
            { binding: 1, resource: { buffer: s.k } }, { binding: 2, resource: { buffer: this.weight(L + "attn_k_norm.weight") } },
            { binding: 3, resource: { buffer: s.kNormed } },
          ]},
          { pl: this.pipelines.rmsNorm, wgX: NUM_KV_HEADS, entries: [
            this.u16(new Uint32Array(normP)),
            { binding: 1, resource: { buffer: s.v } }, { binding: 2, resource: { buffer: s.onesWeight } },
            { binding: 3, resource: { buffer: s.vNormed } },
          ]},
        ]);
      } else {
        this.dispatch(enc, this.pipelines.rmsNorm, [
          this.u16(new Uint32Array(normP)),
          { binding: 1, resource: { buffer: s.q } }, { binding: 2, resource: { buffer: this.weight(L + "attn_q_norm.weight") } },
          { binding: 3, resource: { buffer: s.qNormed } },
        ], NUM_HEADS);
      }
    }
    if (dumpL0) {
      this.dumpForDebug(enc, s.qNormed, qSize, "Qcur_normed-0");
      if (reuseLayer < 0) {
        this.dumpForDebug(enc, s.kNormed, kvSize, "Kcur_normed-0");
        this.dumpForDebug(enc, s.vNormed, kvSize, "Vcur_normed-0");
      }
    }

    // 4. RoPE Q (and K if own-KV). Shared-KV layers skip K RoPE.
    if (reuseLayer < 0) {
      this.multiDispatch(enc, "rope", [
        { pl: this.pipelines.rope, wgX: Math.ceil(NUM_HEADS * dim / 2 / 256), entries: [
          this.u32(mkRopeParams(NUM_HEADS)),
          { binding: 1, resource: { buffer: s.qNormed } },
          { binding: 2, resource: { buffer: ropeFreqsBuf } },
        ]},
        { pl: this.pipelines.rope, wgX: Math.ceil(NUM_KV_HEADS * dim / 2 / 256), entries: [
          this.u32(mkRopeParams(NUM_KV_HEADS)),
          { binding: 1, resource: { buffer: s.kNormed } },
          { binding: 2, resource: { buffer: ropeFreqsBuf } },
        ]},
      ]);
    } else {
      this.dispatch(enc, this.pipelines.rope, [
        this.u32(mkRopeParams(NUM_HEADS)),
        { binding: 1, resource: { buffer: s.qNormed } },
        { binding: 2, resource: { buffer: ropeFreqsBuf } },
      ], Math.ceil(NUM_HEADS * dim / 2 / 256));
    }
    if (dumpL0) {
      this.dumpForDebug(enc, s.qNormed, qSize, "Qcur_pos-0");
      this.dumpForDebug(enc, s.kNormed, kvSize, "Kcur_pos-0");
    }

    // 5. TQ-encode current K/V into the cache. For shared-KV reuse layers
    //    (15..34), skip — the source layer (13/14) has already populated the
    //    shared cache for this token.
    const pos = cache.length;
    let numPos = pos;
    if (reuseLayer < 0) {
      numPos = pos + 1;
      const kEncParams = this.u32(new Uint32Array([dim, 1, kPolarWords, kQjlWords, pos, MAX_POSITIONS, 0, 0]).buffer);
      const vEncParams = this.u32(new Uint32Array([dim, 1, vPolarWords, vQjlWords, pos, MAX_POSITIONS, 0, 0]).buffer);
      const signsBuf = this.signsBuf(layer);
      this.multiDispatch(enc, "tq_encode", [
        { pl: this.pipelines.tqEncodeK, wgX: 1, entries: [
          kEncParams, { binding: 1, resource: { buffer: s.kNormed } },
          { binding: 2, resource: { buffer: signsBuf } }, { binding: 3, resource: { buffer: cache.kPolar } },
          { binding: 4, resource: { buffer: cache.kQjl } }, { binding: 5, resource: { buffer: cache.kMaxR } },
          { binding: 6, resource: { buffer: cache.kGamma } },
        ]},
        { pl: this.pipelines.tqEncodeV, wgX: 1, entries: [
          vEncParams, { binding: 1, resource: { buffer: s.vNormed } },
          { binding: 2, resource: { buffer: signsBuf } }, { binding: 3, resource: { buffer: cache.vPolar } },
          { binding: 4, resource: { buffer: cache.vQjl } }, { binding: 5, resource: { buffer: cache.vMaxR } },
          { binding: 6, resource: { buffer: cache.vGamma } },
        ]},
      ]);
      cache.length = numPos;

      // Debug: decode K and V back out of the cache and dump for the layer-0
      // conformance probe so we can measure per-element K/V round-trip error.
      if (dumpL0 && this.debugDumps) {
        const decParamsK = this.u32(new Uint32Array([dim, 1, kPolarWords, kQjlWords, pos, MAX_POSITIONS, 0, 0]).buffer);
        const decParamsV = this.u32(new Uint32Array([dim, 1, vPolarWords, vQjlWords, pos, MAX_POSITIONS, 0, 0]).buffer);
        this.dispatch(enc, this.pipelines.tqDecodeK, [
          decParamsK, { binding: 1, resource: { buffer: cache.kPolar } },
          { binding: 2, resource: { buffer: cache.kQjl } }, { binding: 3, resource: { buffer: cache.kMaxR } },
          { binding: 4, resource: { buffer: cache.kGamma } }, { binding: 5, resource: { buffer: rot } },
          { binding: 6, resource: { buffer: s.kDecoded } },
        ], 1);
        this.dispatch(enc, this.pipelines.tqDecodeV, [
          decParamsV, { binding: 1, resource: { buffer: cache.vPolar } },
          { binding: 2, resource: { buffer: cache.vQjl } }, { binding: 3, resource: { buffer: cache.vMaxR } },
          { binding: 4, resource: { buffer: cache.vGamma } }, { binding: 5, resource: { buffer: rot } },
          { binding: 6, resource: { buffer: s.vDecoded } },
        ], 1);
        this.dumpForDebug(enc, s.kDecoded, kvSize, "K_decoded-0");
        this.dumpForDebug(enc, s.vDecoded, kvSize, "V_decoded-0");
        // Also dump max_r and gamma for both K and V so we can see the rotated magnitude.
        this.dumpForDebug(enc, cache.kMaxR, 1, "K_maxR-0");
        this.dumpForDebug(enc, cache.vMaxR, 1, "V_maxR-0");
        this.dumpForDebug(enc, cache.kGamma, 1, "K_gamma-0");
        this.dumpForDebug(enc, cache.vGamma, 1, "V_gamma-0");
      }
    }

    // 6. TQ attention — everything reads directly from the TQ-compressed cache.
    //    Q is rotated into TQ space once, scored against all positions in a single
    //    dispatch, softmaxed per head, then combined with V (also in TQ form) and
    //    inverse-rotated back out. No raw K/V at any step.
    //
    //    IMPORTANT: pass the query's absolute position (= `position` arg), not
    //    cache.length. For shared-KV layers the source layer (13/14) has already
    //    bumped cache.length, so cache.length = position + 1 and using it as the
    //    query position would make the causal mask include position+1.
    const winStart = (!isGlobal && SLIDING_WINDOW > 0) ? Math.max(0, numPos - SLIDING_WINDOW) : 0;

    // 6a. Rotate Q via FWHT: one workgroup per head.
    {
      const signs = this.signsBuf(layer);
      this.dispatch(enc, this.pipelines.tqRotate, [
        this.u16(new Uint32Array([dim, NUM_HEADS, 0, 0])),
        { binding: 1, resource: { buffer: s.qNormed } },
        { binding: 2, resource: { buffer: signs } },
        { binding: 3, resource: { buffer: s.rotQ } },
      ], NUM_HEADS);
    }

    // 6b. TQ scores = Q_rot . K_tq for each (head, position). Stored packed as
    //     s.scores[head * numPos + pos].
    this.dispatch(enc, this.pipelines.tqAttention, [
      this.u32(new Uint32Array([dim, numPos, kPolarWords, kQjlWords, winStart, NUM_HEADS, MAX_POSITIONS, 0]).buffer),
      { binding: 1, resource: { buffer: s.rotQ } },
      { binding: 2, resource: { buffer: cache.kPolar } },
      { binding: 3, resource: { buffer: cache.kQjl } },
      { binding: 4, resource: { buffer: cache.kMaxR } },
      { binding: 5, resource: { buffer: cache.kGamma } },
      { binding: 6, resource: { buffer: s.scores } },
    ], Math.ceil((numPos - winStart) / 256), NUM_HEADS);

    if (dumpL0) this.dumpForDebug(enc, s.scores, NUM_HEADS * numPos, "scores-0");

    // 6c. Multi-head softmax (NUM_HEADS workgroups, shared query position).
    this.dispatch(enc, this.pipelines.mhSoftmax, [
      this.u16(new Uint32Array([numPos, winStart, position, 0])),
      { binding: 1, resource: { buffer: s.scores } },
    ], NUM_HEADS);

    // 6d. TQ weighted sum: scores * V_tq -> s.rotOut (in rotated space).
    // Split into two passes for parallelism — the old single-shader version
    // only dispatched NUM_HEADS (=8) workgroups per layer, so at long
    // contexts the position loop became the wall. Phase 1 splits dims
    // across workgroups (DIM_TILE=64) and splits positions across 4 lanes
    // per dim within each workgroup, landing at ~4× more workgroups with
    // ~4× less per-thread work. Phase 2 stays per-head and just does the
    // R^T @ qjl_sum mat-vec, which is small.
    const rotT = this.rotBufT(layer);
    const DIM_TILE = 64;
    const numDimTiles = Math.ceil(dim / DIM_TILE);
    this.dispatch(enc, this.pipelines.tqWeightedSumP1, [
      this.u32(new Uint32Array([dim, numPos, vPolarWords, vQjlWords, winStart, MAX_POSITIONS, 0, 0]).buffer),
      { binding: 1, resource: { buffer: s.scores } },
      { binding: 2, resource: { buffer: cache.vPolar } },
      { binding: 3, resource: { buffer: cache.vQjl } },
      { binding: 4, resource: { buffer: cache.vMaxR } },
      { binding: 5, resource: { buffer: cache.vGamma } },
      { binding: 6, resource: { buffer: s.wsumPolar } },
      { binding: 7, resource: { buffer: s.wsumQjl } },
    ], NUM_HEADS, numDimTiles);
    // P2 uses a DIM_TILE=64 layout (one workgroup per (head, dim_tile)
    // pair) to unblock M1's parallelism — the old single-workgroup-per-
    // head layout ran 2-8 workgroups total and left most compute slots
    // idle. At dim=256 that's 8×4=32 workgroups; at dim=512, 8×8=64.
    // P2 now uses FWHT: one WG per head (was DIM_TILE=64 multi-WG).
    this.dispatch(enc, this.pipelines.tqWeightedSumP2, [
      this.u16(new Uint32Array([dim, 0, 0, 0])),
      { binding: 1, resource: { buffer: s.wsumPolar } },
      { binding: 2, resource: { buffer: s.wsumQjl } },
      { binding: 3, resource: { buffer: this.signsBuf(layer) } },
      { binding: 4, resource: { buffer: s.rotOut } },
    ], NUM_HEADS);

    // 6e. Inverse rotate via FWHT: one workgroup per head.
    {
      const signs = this.signsBuf(layer);
      this.dispatch(enc, this.pipelines.tqInverseRotate, [
        this.u16(new Uint32Array([dim, NUM_HEADS, 0, 0])),
        { binding: 1, resource: { buffer: s.rotOut } },
        { binding: 2, resource: { buffer: signs } },
        { binding: 3, resource: { buffer: s.attnOut } },
      ], NUM_HEADS);
    }

    if (dumpL0) this.dumpForDebug(enc, s.attnOut, qSize, "kqv_out-0");

    // 7. Output projection
    this.matmul(enc, this.weight(L + "attn_output.weight"), s.attnOut, s.attnProj, HIDDEN_SIZE, qSize);
    if (dumpL0) this.dumpForDebug(enc, s.attnProj, HIDDEN_SIZE, "node_33");

    // 8+9. Fused post-attention norm + residual: hidden1 = input + rmsNorm(attnProj)
    this.rmsNormResAdd(enc, s.attnProj, this.weight(L + "post_attention_norm.weight"), inputBuf, s.hidden1, HIDDEN_SIZE);
    if (dumpL0) this.dumpForDebug(enc, s.hidden1, HIDDEN_SIZE, "attn_out-0");

    // 10. Pre-FFN norm
    this.rmsNorm(enc, s.hidden1, this.weight(L + "ffn_norm.weight"), s.mlpNormed, HIDDEN_SIZE);

    // 11. Gated MLP — gate/up merged (independent: both read mlpNormed)
    const ffnSize = layer >= 15 ? 12288 : 6144;
    this.multiDispatch(enc, "matmul", [
      this.matmulOp(this.weight(L + "ffn_gate.weight"), s.mlpNormed, s.gate, ffnSize, HIDDEN_SIZE),
      this.matmulOp(this.weight(L + "ffn_up.weight"), s.mlpNormed, s.up, ffnSize, HIDDEN_SIZE),
    ]);
    // Fused path (#78): when ffn_down is Q4K, skip the standalone gelu_gate
    // dispatch and run a variant matmul that reads gate[] + up[], applies
    // GELU-gated mixing inline, and multiplies by Q4K weights in one pass.
    // Eliminates the ffnSize-element round-trip of s.geluOut through global
    // memory. Q6K path keeps the unfused pair — no matmul_q6k_gelu variant
    // yet, and ffn_down is almost always Q4K in Gemma 4 E2B so the Q6K
    // fallback is rare enough not to justify a second shader.
    const ffnDownTensor = this.model.tensors.get(L + "ffn_down.weight")!;
    if (ffnDownTensor.type === GGML_Q6_K) {
      this.dispatch(enc, this.pipelines.gelu, [
        this.u16(new Uint32Array([ffnSize, 0, 0, 0])),
        { binding: 1, resource: { buffer: s.gate } }, { binding: 2, resource: { buffer: s.up } },
        { binding: 3, resource: { buffer: s.geluOut } },
      ], Math.ceil(ffnSize / 256));
      this.matmulQ6k(enc, ffnDownTensor.gpuBuffer!, s.geluOut, s.mlpOut, HIDDEN_SIZE, ffnSize);
    } else {
      this.matmulQ4KGelu(enc, ffnDownTensor.gpuBuffer!, s.gate, s.up, s.mlpOut, HIDDEN_SIZE, ffnSize);
    }

    // 12+13. Fused post-FFN norm + residual: output = hidden1 + rmsNorm(mlpOut)
    this.rmsNormResAdd(enc, s.mlpOut, this.weight(L + "post_ffw_norm.weight"), s.hidden1, outputBuf, HIDDEN_SIZE);
    if (dumpL0) {
      this.dumpForDebug(enc, outputBuf, HIDDEN_SIZE, "pe_in-0");
      this.dumpForDebug(enc, s.perLayerProj, N_EMBD_PER_LAYER, "inp_per_layer_slice0");
    }

    // 14. Per-Layer Embedding (Gemma 4):
    //     gated       = inp_gate @ outputBuf     [256]
    //     geluGated   = gelu(gated) * inp_per_layer[layer]  [256]  (element-wise)
    //     projected   = proj @ geluGated         [1536]
    //     outputBuf  += rms_norm(projected) * post_norm      [1536]
    const inpGate = this.weight(L + "inp_gate.weight");
    const projW = this.weight(L + "proj.weight");
    const wg256 = Math.ceil(N_EMBD_PER_LAYER / 8);
    this.dispatch(enc, this.pipelines.matmulF32, [
      this.u16(new Uint32Array([N_EMBD_PER_LAYER, HIDDEN_SIZE, 0, 0])),
      { binding: 1, resource: { buffer: inpGate } },
      { binding: 2, resource: { buffer: outputBuf } },
      { binding: 3, resource: { buffer: s.pleGated } },
    ], Math.min(wg256, 65535), Math.ceil(wg256 / 65535));
    if (dumpL0) this.dumpForDebug(enc, s.pleGated, N_EMBD_PER_LAYER, "ple_gate-0");
    if (dumpL0) this.dumpForDebug(enc, s.perLayerProj, N_EMBD_PER_LAYER, "ple_layer_slice-0");
    // Fused GELU-gated F32 matmul: skip the standalone gelu dispatch AND the
    // pleLayerSlice copy. The kernel reads its `slice` binding at base
    // offset = layer * N_EMBD_PER_LAYER * 4 directly out of perLayerProj,
    // so the layer-specific slice never needs to be materialized in a
    // separate buffer. Saves one copyBufferToBuffer + its pass-boundary
    // per layer × 35 layers per decode step.
    const wgHidden = Math.ceil(HIDDEN_SIZE / 8);
    this.dispatch(enc, this.pipelines.matmulF32Gelu, [
      this.u16(new Uint32Array([HIDDEN_SIZE, N_EMBD_PER_LAYER, 0, 0])),
      { binding: 1, resource: { buffer: projW } },
      { binding: 2, resource: { buffer: s.pleGated } },
      { binding: 3, resource: { buffer: s.perLayerProj, offset: layer * N_EMBD_PER_LAYER * 4, size: N_EMBD_PER_LAYER * 4 } },
      { binding: 4, resource: { buffer: s.pleProjected } },
    ], Math.min(wgHidden, 65535), Math.ceil(wgHidden / 65535));
    if (dumpL0) this.dumpForDebug(enc, s.pleProjected, HIDDEN_SIZE, "ple_projected-0");
    // per_layer_embd_out = rms_norm(pleProjected) * post_norm
    this.rmsNorm(enc, s.pleProjected, this.weight(L + "post_norm.weight"), s.attnPostNormed, HIDDEN_SIZE);
    if (dumpL0) this.dumpForDebug(enc, s.attnPostNormed, HIDDEN_SIZE, "per_layer_embd_out-0");
    // Fused PLE residual + layer-output scale: outputBuf = (outputBuf +
    // attnPostNormed) * layer_output_scale[0]. Previously two separate
    // dispatches (residualAddInplace + scaleByBuf) — now one kernel per
    // layer × 35 layers saved per decode.
    {
      const outScaleBuf = this.weight(L + "layer_output_scale.weight");
      const resWG = Math.ceil(HIDDEN_SIZE / 256);
      this.dispatch(enc, this.pipelines.residualAddScaleInplace, [
        this.u16(new Uint32Array([HIDDEN_SIZE, 0, 0, 0])),
        { binding: 1, resource: { buffer: outputBuf } },
        { binding: 2, resource: { buffer: s.attnPostNormed } },
        { binding: 3, resource: { buffer: outScaleBuf } },
      ], Math.min(resWG, 65535), Math.ceil(resWG / 65535));
    }
    if (dumpL0) this.dumpForDebug(enc, outputBuf, HIDDEN_SIZE, "node_62-0");
  }

  // ===========================================================================
  // Run one decoder layer — batched variant
  // ===========================================================================
  //
  // Processes `nBatch` tokens in parallel through a single layer. Input/output
  // buffers hold nBatch stacked HIDDEN_SIZE vectors; most scratch buffers are
  // already sized × MAX_SPEC_BATCH in create().
  //
  // Batching strategy:
  //   - rms_norm: nRows = nBatch × original_nRows (existing shader handles it)
  //   - rope: n_batch = nBatch (existing shader handles it)
  //   - matmul Q/K/V/attn_output/gate/up/down: matmulBatched
  //   - tq_encode: num_vectors = nBatch (existing shader handles it)
  //   - residual_add: n = nBatch × HIDDEN_SIZE (flat buffer)
  //   - gelu: count = nBatch × ffnSize (flat buffer)
  //   - Attention (rotate, scores, softmax, wsum, invrot): per-slot sequential
  //     loop because each slot has a different attention range (causal mask at
  //     positionStart+slot). Uses binding offsets to write each slot's attnOut.
  //
  // KV cache: all slots' K/V are encoded in one tq_encode dispatch at positions
  // positionStart..positionStart+nBatch-1. The attention loop uses numPos =
  // positionStart + slot + 1 to enforce per-slot causal mask.
  runLayerBatched(
    enc: GPUCommandEncoder,
    layer: number,
    inputBuf: GPUBuffer,
    outputBuf: GPUBuffer,
    positionStart: number,
    nBatch: number,
  ): void {
    const s = this.scratch;
    const dim = this.headDim(layer);
    const isGlobal = isGlobalLayer(layer);
    const rot = this.rotBuf(layer);
    const reuseLayer = kvReuseLayer(layer);
    const kvCacheLayer = reuseLayer >= 0 ? reuseLayer : layer;
    const cache = this.getCache(kvCacheLayer);
    const kPolarWords = polarWordsPerPos(dim, K_POLAR_CONFIG);
    const vPolarWords = polarWordsPerPos(dim, V_POLAR_CONFIG);
    const kQjlWords = qjlWordsPerPos(dim, K_POLAR_CONFIG);
    const vQjlWords = qjlWordsPerPos(dim, V_POLAR_CONFIG);
    const ropeDims = dim;
    const ropeTheta = isGlobal ? ROPE_THETA_GLOBAL : ROPE_THETA_SLIDING;
    const qSize = NUM_HEADS * dim;
    const kvSize = NUM_KV_HEADS * dim;
    const L = "blk." + layer + ".";
    const ropeFreqsBuf = isGlobal ? this.weight("rope_freqs.weight") : this.dummyFreqFactors;

    // Helper: WebGPU binding with byte offset into a buffer. Offset must be a
    // multiple of minStorageBufferOffsetAlignment (256 bytes on Metal). All our
    // per-slot scratches' slot sizes (HIDDEN_SIZE*4 = 6144, qSize*4 ≥ 4096) are
    // multiples of 256, so slot * size is always aligned.
    const bindOff = (buf: GPUBuffer, slot: number, sizeF32: number): GPUBindGroupEntry => ({
      binding: 0,
      resource: { buffer: buf, offset: slot * sizeF32 * 4, size: sizeF32 * 4 },
    });

    // 1. Pre-attention norm — nBatch rows of HIDDEN_SIZE.
    this.rmsNorm(enc, inputBuf, this.weight(L + "attn_norm.weight"), s.normed, HIDDEN_SIZE, nBatch);

    // 2. Q/K/V projections — batched matmul. Shared-KV layers skip K and V;
    //    tq_encode guard at step 5 already does this, so matching the skip
    //    earlier avoids ~40 % of Q/K/V matmul work during prefill too.
    const vTensor = this.model.tensors.get(L + "attn_v.weight")!;
    this.matmulBatched(enc, this.weight(L + "attn_q.weight"), s.normed, s.q, qSize, HIDDEN_SIZE, nBatch, false);
    if (reuseLayer < 0) {
      this.matmulBatched(enc, this.weight(L + "attn_k.weight"), s.normed, s.k, kvSize, HIDDEN_SIZE, nBatch, false);
      this.matmulBatched(enc, vTensor.gpuBuffer!, s.normed, s.v, kvSize, HIDDEN_SIZE, nBatch, vTensor.type === GGML_Q6_K);
    }

    // 3. QK-norm + V-norm. Shared-KV layers skip K-norm and V-norm.
    {
      const normP = new ArrayBuffer(16);
      new Uint32Array(normP)[0] = dim;
      new Float32Array(normP)[1] = RMS_NORM_EPS;
      this.dispatch(enc, this.pipelines.rmsNorm, [
        this.u16(new Uint32Array(normP)),
        { binding: 1, resource: { buffer: s.q } },
        { binding: 2, resource: { buffer: this.weight(L + "attn_q_norm.weight") } },
        { binding: 3, resource: { buffer: s.qNormed } },
      ], nBatch * NUM_HEADS);
      if (reuseLayer < 0) {
        this.dispatch(enc, this.pipelines.rmsNorm, [
          this.u16(new Uint32Array(normP)),
          { binding: 1, resource: { buffer: s.k } },
          { binding: 2, resource: { buffer: this.weight(L + "attn_k_norm.weight") } },
          { binding: 3, resource: { buffer: s.kNormed } },
        ], nBatch * NUM_KV_HEADS);
        this.dispatch(enc, this.pipelines.rmsNorm, [
          this.u16(new Uint32Array(normP)),
          { binding: 1, resource: { buffer: s.v } },
          { binding: 2, resource: { buffer: s.onesWeight } },
          { binding: 3, resource: { buffer: s.vNormed } },
        ], nBatch * NUM_KV_HEADS);
      }
    }

    // 4. RoPE Q/K — n_batch = nBatch. Shared-KV layers skip K RoPE.
    {
      const mkRopeParamsBatched = (nHeads: number): ArrayBuffer => {
        const buf = new ArrayBuffer(32);
        const u = new Uint32Array(buf);
        new Float32Array(buf)[2] = ropeTheta;
        u[0] = dim;
        u[1] = positionStart;
        u[3] = ropeDims;
        u[4] = nHeads;
        u[5] = nBatch;
        u[6] = isGlobal ? 1 : 0;
        return buf;
      };
      this.dispatch(enc, this.pipelines.rope, [
        this.u32(mkRopeParamsBatched(NUM_HEADS)),
        { binding: 1, resource: { buffer: s.qNormed } },
        { binding: 2, resource: { buffer: ropeFreqsBuf } },
      ], Math.ceil(nBatch * NUM_HEADS * dim / 2 / 256));
      if (reuseLayer < 0) {
        this.dispatch(enc, this.pipelines.rope, [
          this.u32(mkRopeParamsBatched(NUM_KV_HEADS)),
          { binding: 1, resource: { buffer: s.kNormed } },
          { binding: 2, resource: { buffer: ropeFreqsBuf } },
        ], Math.ceil(nBatch * NUM_KV_HEADS * dim / 2 / 256));
      }
    }

    // 5. TQ-encode K/V — num_vectors = nBatch, write_pos = positionStart.
    // Skip for shared-KV reuse layers (15..34).
    let numPosAfterEncode = cache.length;
    if (reuseLayer < 0) {
      numPosAfterEncode = cache.length + nBatch;
      const kEncParams = this.u32(new Uint32Array([dim, nBatch, kPolarWords, kQjlWords, cache.length, MAX_POSITIONS, 0, 0]).buffer);
      const vEncParams = this.u32(new Uint32Array([dim, nBatch, vPolarWords, vQjlWords, cache.length, MAX_POSITIONS, 0, 0]).buffer);
      this.dispatch(enc, this.pipelines.tqEncodeK, [
        kEncParams,
        { binding: 1, resource: { buffer: s.kNormed } },
        { binding: 2, resource: { buffer: this.signsBuf(layer) } },
        { binding: 3, resource: { buffer: cache.kPolar } },
        { binding: 4, resource: { buffer: cache.kQjl } },
        { binding: 5, resource: { buffer: cache.kMaxR } },
        { binding: 6, resource: { buffer: cache.kGamma } },
      ], nBatch);
      this.dispatch(enc, this.pipelines.tqEncodeV, [
        vEncParams,
        { binding: 1, resource: { buffer: s.vNormed } },
        { binding: 2, resource: { buffer: this.signsBuf(layer) } },
        { binding: 3, resource: { buffer: cache.vPolar } },
        { binding: 4, resource: { buffer: cache.vQjl } },
        { binding: 5, resource: { buffer: cache.vMaxR } },
        { binding: 6, resource: { buffer: cache.vGamma } },
      ], nBatch);
      cache.length = numPosAfterEncode;
    }

    // 6. TQ attention — sequential per slot. Each slot's query uses its own
    //    causal attention range numPos = positionStartCache + slot + 1 (causal
    //    includes own position) and writes its result to s.attnOut[slot * qSize].
    const rotT = this.rotBufT(layer);
    const positionStartCache = reuseLayer >= 0 ? cache.length - nBatch : cache.length - nBatch;
    // ^ cache.length has been bumped to include all nBatch; slot 0 sees
    //   positions 0..positionStartCache+1, slot j sees 0..positionStartCache+j+1.

    for (let slot = 0; slot < nBatch; slot++) {
      const slotPos = positionStart + slot;
      const numPos = Math.min(positionStartCache + slot + 1, MAX_POSITIONS);
      const winStart = (!isGlobal && SLIDING_WINDOW > 0) ? Math.max(0, numPos - SLIDING_WINDOW) : 0;

      // 6a. Rotate Q via FWHT.
      this.dispatch(enc, this.pipelines.tqRotate, [
        this.u16(new Uint32Array([dim, NUM_HEADS, 0, 0])),
        { ...bindOff(s.qNormed, slot, qSize), binding: 1 },
        { binding: 2, resource: { buffer: this.signsBuf(layer) } },
        { binding: 3, resource: { buffer: s.rotQ } },
      ], NUM_HEADS);

      // 6b. TQ scores.
      this.dispatch(enc, this.pipelines.tqAttention, [
        this.u32(new Uint32Array([dim, numPos, kPolarWords, kQjlWords, winStart, NUM_HEADS, MAX_POSITIONS, 0]).buffer),
        { binding: 1, resource: { buffer: s.rotQ } },
        { binding: 2, resource: { buffer: cache.kPolar } },
        { binding: 3, resource: { buffer: cache.kQjl } },
        { binding: 4, resource: { buffer: cache.kMaxR } },
        { binding: 5, resource: { buffer: cache.kGamma } },
        { binding: 6, resource: { buffer: s.scores } },
      ], Math.ceil((numPos - winStart) / 256), NUM_HEADS);

      // 6c. Softmax.
      this.dispatch(enc, this.pipelines.mhSoftmax, [
        this.u16(new Uint32Array([numPos, winStart, slotPos, 0])),
        { binding: 1, resource: { buffer: s.scores } },
      ], NUM_HEADS);

      // 6d. Weighted sum phase 1 + phase 2.
      const DIM_TILE = 64;
      const numDimTiles = Math.ceil(dim / DIM_TILE);
      this.dispatch(enc, this.pipelines.tqWeightedSumP1, [
        this.u32(new Uint32Array([dim, numPos, vPolarWords, vQjlWords, winStart, MAX_POSITIONS, 0, 0]).buffer),
        { binding: 1, resource: { buffer: s.scores } },
        { binding: 2, resource: { buffer: cache.vPolar } },
        { binding: 3, resource: { buffer: cache.vQjl } },
        { binding: 4, resource: { buffer: cache.vMaxR } },
        { binding: 5, resource: { buffer: cache.vGamma } },
        { binding: 6, resource: { buffer: s.wsumPolar } },
        { binding: 7, resource: { buffer: s.wsumQjl } },
      ], NUM_HEADS, numDimTiles);
      this.dispatch(enc, this.pipelines.tqWeightedSumP2, [
        this.u16(new Uint32Array([dim, 0, 0, 0])),
        { binding: 1, resource: { buffer: s.wsumPolar } },
        { binding: 2, resource: { buffer: s.wsumQjl } },
        { binding: 3, resource: { buffer: this.signsBuf(layer) } },
        { binding: 4, resource: { buffer: s.rotOut } },
      ], NUM_HEADS);

      // 6e. Inverse rotate via FWHT.
      this.dispatch(enc, this.pipelines.tqInverseRotate, [
        this.u16(new Uint32Array([dim, NUM_HEADS, 0, 0])),
        { binding: 1, resource: { buffer: s.rotOut } },
        { binding: 2, resource: { buffer: this.signsBuf(layer) } },
        { ...bindOff(s.attnOut, slot, qSize), binding: 3 },
      ], NUM_HEADS);
    }

    // 7. Output projection — batched matmul on the stacked attnOut buffer.
    this.matmulBatched(enc, this.weight(L + "attn_output.weight"), s.attnOut, s.attnProj, HIDDEN_SIZE, qSize, nBatch, false);

    // 8+9. Fused post-attention norm + residual — nBatch rows.
    this.rmsNormResAdd(enc, s.attnProj, this.weight(L + "post_attention_norm.weight"), inputBuf, s.hidden1, HIDDEN_SIZE, nBatch);

    // 10. Pre-FFN norm — nBatch rows.
    this.rmsNorm(enc, s.hidden1, this.weight(L + "ffn_norm.weight"), s.mlpNormed, HIDDEN_SIZE, nBatch);

    // 11. Gated MLP — gate/up/down all batched.
    const ffnSize = layer >= 15 ? 12288 : 6144;
    const ffnDownTensor = this.model.tensors.get(L + "ffn_down.weight")!;
    this.matmulBatched(enc, this.weight(L + "ffn_gate.weight"), s.mlpNormed, s.gate, ffnSize, HIDDEN_SIZE, nBatch, false);
    this.matmulBatched(enc, this.weight(L + "ffn_up.weight"), s.mlpNormed, s.up, ffnSize, HIDDEN_SIZE, nBatch, false);
    // Fused GELU-gated ffn_down when nBatch is one of the compile-time-N
    // variants (2/4/8) AND ffn_down is Q4K. Saves one standalone dispatch
    // + a (nBatch × ffnSize)-element round-trip through global memory,
    // which is the biggest single contributor to prefill bandwidth after
    // matmul itself.
    const canFuse = ffnDownTensor.type !== GGML_Q6_K && (nBatch === 2 || nBatch === 4 || nBatch === 8);
    if (canFuse) {
      this.matmulBatchedGelu(enc, ffnDownTensor.gpuBuffer!, s.gate, s.up, s.mlpOut, HIDDEN_SIZE, ffnSize, nBatch);
    } else {
      this.dispatch(enc, this.pipelines.gelu, [
        this.u16(new Uint32Array([nBatch * ffnSize, 0, 0, 0])),
        { binding: 1, resource: { buffer: s.gate } },
        { binding: 2, resource: { buffer: s.up } },
        { binding: 3, resource: { buffer: s.geluOut } },
      ], Math.ceil(nBatch * ffnSize / 256));
      this.matmulBatched(enc, ffnDownTensor.gpuBuffer!, s.geluOut, s.mlpOut, HIDDEN_SIZE, ffnSize, nBatch, ffnDownTensor.type === GGML_Q6_K);
    }

    // 12+13. Fused post-FFN norm + residual — nBatch rows.
    this.rmsNormResAdd(enc, s.mlpOut, this.weight(L + "post_ffw_norm.weight"), s.hidden1, outputBuf, HIDDEN_SIZE, nBatch);

    // 14. Per-Layer Embedding — runs N times sequentially (each slot has its
    // own PLE output). This is small (~2 matmuls of size 256×1536 in bf16/f32)
    // and keeping it sequential avoids refactoring the matmulBf16/F32 kernels.
    const inpGate = this.weight(L + "inp_gate.weight");
    const projW = this.weight(L + "proj.weight");
    for (let slot = 0; slot < nBatch; slot++) {
      const wg256 = Math.ceil(N_EMBD_PER_LAYER / 8);
      this.dispatch(enc, this.pipelines.matmulF32, [
        this.u16(new Uint32Array([N_EMBD_PER_LAYER, HIDDEN_SIZE, 0, 0])),
        { binding: 1, resource: { buffer: inpGate } },
        { ...bindOff(outputBuf, slot, HIDDEN_SIZE), binding: 2 },
        { ...bindOff(s.pleGated, slot, N_EMBD_PER_LAYER), binding: 3 },
      ], Math.min(wg256, 65535), Math.ceil(wg256 / 65535));
      // Fused GELU-gated F32 matmul (#78) — same pattern as the decode-path
      // PLE fusion, with the same zero-copy slice bind that eliminates the
      // intermediate pleLayerSlice materialization.
      const wgHidden = Math.ceil(HIDDEN_SIZE / 8);
      const perLayerSliceOffset = slot * NUM_LAYERS * N_EMBD_PER_LAYER * 4 + layer * N_EMBD_PER_LAYER * 4;
      this.dispatch(enc, this.pipelines.matmulF32Gelu, [
        this.u16(new Uint32Array([HIDDEN_SIZE, N_EMBD_PER_LAYER, 0, 0])),
        { binding: 1, resource: { buffer: projW } },
        { ...bindOff(s.pleGated, slot, N_EMBD_PER_LAYER), binding: 2 },
        { binding: 3, resource: { buffer: s.perLayerProj, offset: perLayerSliceOffset, size: N_EMBD_PER_LAYER * 4 } },
        { ...bindOff(s.pleProjected, slot, HIDDEN_SIZE), binding: 4 },
      ], Math.min(wgHidden, 65535), Math.ceil(wgHidden / 65535));
    }
    // Per-layer-embd norm + fused residual-add + scale, all slots at once.
    this.rmsNorm(enc, s.pleProjected, this.weight(L + "post_norm.weight"), s.attnPostNormed, HIDDEN_SIZE, nBatch);
    {
      const outScaleBuf = this.weight(L + "layer_output_scale.weight");
      const totalN = nBatch * HIDDEN_SIZE;
      const resWG = Math.ceil(totalN / 256);
      this.dispatch(enc, this.pipelines.residualAddScaleInplace, [
        this.u16(new Uint32Array([totalN, 0, 0, 0])),
        { binding: 1, resource: { buffer: outputBuf } },
        { binding: 2, resource: { buffer: s.attnPostNormed } },
        { binding: 3, resource: { buffer: outScaleBuf } },
      ], Math.min(resWG, 65535), Math.ceil(resWG / 65535));
    }
  }

  // ===========================================================================
  // Embedding + Logit Projection
  // ===========================================================================

  /** Dequantize one row from Q4_K embedding table, then multiply by sqrt(hidden_size).
   *  convert_hf_to_gguf.py's Gemma4Model does NOT pre-scale the embedding, so the scale
   *  must be applied at runtime — matching llama.cpp's ggml_scale(inpL, sqrtf(n_embd)).
   *
   *  Reads `row` from `this._tokenIdBuf[slot]` — a persistent GPU-resident id buffer.
   *  generateToken stages its token via queue.writeBuffer at slot 0. Batch prefill
   *  writes CHUNK ids at once and passes slot=i for each dispatch. */
  embedToken(enc: GPUCommandEncoder, slot: number, outBuf: GPUBuffer): void {
    this.dispatch(enc, this.pipelines.getRowsQ4k, [
      this.u16(new Uint32Array([HIDDEN_SIZE, slot, 0, 0])),
      { binding: 1, resource: { buffer: this.weight("token_embd.weight") } },
      { binding: 2, resource: { buffer: outBuf } },
      { binding: 3, resource: { buffer: this._tokenIdBuf! } },
    ], Math.ceil(HIDDEN_SIZE / 256));
    this.dumpForDebug(enc, outBuf, HIDDEN_SIZE, "embd_raw");
    this.scaleInplace(enc, outBuf, HIDDEN_SIZE, Math.sqrt(HIDDEN_SIZE));
  }

  /** Compute the per-layer embedding inputs for the current token.
   *  Mirrors llama.cpp's build_inp_per_layer + project_per_layer_inputs.
   *
   *  Output: `scratch.perLayerRaw` holds the combined per-layer input of shape
   *  [N_EMBD_PER_LAYER × NUM_LAYERS], sliced per-layer inside runLayer.
   *
   *  Steps:
   *   1. per_layer_raw = per_layer_token_embd[tokenId] * sqrt(N_EMBD_PER_LAYER)
   *   2. per_layer_proj = per_layer_model_proj @ inp_scaled
   *   3. per_layer_proj *= 1 / sqrt(HIDDEN_SIZE)
   *   4. per_layer_proj = rms_norm(per_layer_proj, per_layer_proj_norm)  -- per 256-chunk
   *   5. per_layer_raw += per_layer_proj
   *   6. per_layer_raw *= 1 / sqrt(2)
   */
  computePerLayerInputs(enc: GPUCommandEncoder, slot: number, inpScaled: GPUBuffer, slotOffset = 0, inpScaledOffset = 0): void {
    const s = this.scratch;
    // slotOffset indexes per-slot blocks in the batched scratch layout used
    // by runLayerBatched. When slotOffset === 0 the bindings use the whole
    // buffer (same as unbatched); when slotOffset > 0 each output binding is
    // offset by slotOffset × PER_LAYER_TOTAL bytes so multiple slots can be
    // populated in the same command encoder without clobbering each other.
    const plOff = slotOffset * PER_LAYER_TOTAL * 4;
    const plSize = PER_LAYER_TOTAL * 4;
    const perLayerRawBinding = slotOffset === 0
      ? { binding: 2, resource: { buffer: s.perLayerRaw } }
      : { binding: 2, resource: { buffer: s.perLayerRaw, offset: plOff, size: plSize } };
    const perLayerProjBinding = (b: number) => slotOffset === 0
      ? { binding: b, resource: { buffer: s.perLayerProj } }
      : { binding: b, resource: { buffer: s.perLayerProj, offset: plOff, size: plSize } };
    const perLayerFinalBinding = (b: number) => slotOffset === 0
      ? { binding: b, resource: { buffer: s.perLayerFinal } }
      : { binding: b, resource: { buffer: s.perLayerFinal, offset: plOff, size: plSize } };
    const perLayerRawViewForScale = slotOffset === 0 ? s.perLayerRaw : undefined;
    const perLayerProjViewForScale = slotOffset === 0 ? s.perLayerProj : undefined;

    // Step 1: get row from Q5_K table, scale by sqrt(N_EMBD_PER_LAYER)
    this.dispatch(enc, this.pipelines.getRowsQ5k, [
      this.u16(new Uint32Array([PER_LAYER_TOTAL, slot, 0, 0])),
      { binding: 1, resource: { buffer: this.weight("per_layer_token_embd.weight") } },
      perLayerRawBinding,
      { binding: 3, resource: { buffer: this._tokenIdBuf! } },
    ], Math.ceil(PER_LAYER_TOTAL / 256));
    this.dumpForDebug(enc, s.perLayerRaw, PER_LAYER_TOTAL, "per_layer_q5k_raw");
    if (perLayerRawViewForScale !== undefined) {
      this.scaleInplace(enc, perLayerRawViewForScale, PER_LAYER_TOTAL, Math.sqrt(N_EMBD_PER_LAYER));
    } else {
      this.scaleInplaceOffset(enc, s.perLayerRaw, plOff, PER_LAYER_TOTAL, Math.sqrt(N_EMBD_PER_LAYER));
    }
    this.dumpForDebug(enc, s.perLayerRaw, PER_LAYER_TOTAL, "inp_per_layer_selected");

    // Step 2: per_layer_proj = per_layer_model_proj @ inp_scaled
    //   per_layer_model_proj.dims = [1536, 8960] → treated as [8960 rows × 1536 cols]
    const projRows = PER_LAYER_TOTAL;
    const wgProj = Math.ceil(projRows / 8);
    const inpScaledBinding = inpScaledOffset === 0
      ? { binding: 2, resource: { buffer: inpScaled } }
      : { binding: 2, resource: { buffer: inpScaled, offset: inpScaledOffset, size: HIDDEN_SIZE * 4 } };
    this.dispatch(enc, this.pipelines.matmulBf16, [
      this.u16(new Uint32Array([projRows, HIDDEN_SIZE, 0, 0])),
      { binding: 1, resource: { buffer: this.weight("per_layer_model_proj.weight") } },
      inpScaledBinding,
      perLayerProjBinding(3),
    ], Math.min(wgProj, 65535), Math.ceil(wgProj / 65535));

    // Step 3: scale by 1 / sqrt(HIDDEN_SIZE)
    if (perLayerProjViewForScale !== undefined) {
      this.scaleInplace(enc, perLayerProjViewForScale, PER_LAYER_TOTAL, 1 / Math.sqrt(HIDDEN_SIZE));
    } else {
      this.scaleInplaceOffset(enc, s.perLayerProj, plOff, PER_LAYER_TOTAL, 1 / Math.sqrt(HIDDEN_SIZE));
    }

    // Step 4: RMS norm each N_EMBD_PER_LAYER-chunk (one per layer) with per_layer_proj_norm
    //   weight. Writes into perLayerFinal so we don't alias perLayerProj.
    const normP = new ArrayBuffer(16);
    new Uint32Array(normP)[0] = N_EMBD_PER_LAYER;
    new Float32Array(normP)[1] = RMS_NORM_EPS;
    this.dispatch(enc, this.pipelines.rmsNorm, [
      this.u16(new Uint32Array(normP)),
      perLayerProjBinding(1),
      { binding: 2, resource: { buffer: this.weight("per_layer_proj_norm.weight") } },
      perLayerFinalBinding(3),
    ], NUM_LAYERS);

    // Step 5: perLayerProj = perLayerRaw + perLayerFinal  (three distinct buffers)
    const addWG = Math.ceil(PER_LAYER_TOTAL / 256);
    this.dispatch(enc, this.pipelines.residualAdd, [
      this.u16(new Uint32Array([PER_LAYER_TOTAL, 0, 0, 0])),
      slotOffset === 0
        ? { binding: 1, resource: { buffer: s.perLayerRaw } }
        : { binding: 1, resource: { buffer: s.perLayerRaw, offset: plOff, size: plSize } },
      perLayerFinalBinding(2),
      perLayerProjBinding(3),
    ], Math.min(addWG, 65535), Math.ceil(addWG / 65535));
    // Step 6: scale perLayerProj by 1/sqrt(2) — this is the final per-layer input buffer.
    if (perLayerProjViewForScale !== undefined) {
      this.scaleInplace(enc, perLayerProjViewForScale, PER_LAYER_TOTAL, 1 / Math.sqrt(2));
    } else {
      this.scaleInplaceOffset(enc, s.perLayerProj, plOff, PER_LAYER_TOTAL, 1 / Math.sqrt(2));
    }
  }

  /** scaleInplace variant that binds the buffer with a byte offset so only a
   *  slice gets scaled. Used by the batched compute path. */
  scaleInplaceOffset(enc: GPUCommandEncoder, buf: GPUBuffer, byteOffset: number, count: number, scale: number): void {
    const paramBuf = new ArrayBuffer(16);
    new Uint32Array(paramBuf)[0] = count;
    new Float32Array(paramBuf)[1] = scale;
    const wg = Math.ceil(count / 256);
    this.dispatch(enc, this.pipelines.scaleInplace, [
      this.u16(new Uint32Array(paramBuf)),
      { binding: 1, resource: { buffer: buf, offset: byteOffset, size: count * 4 } },
    ], Math.min(wg, 65535), Math.ceil(wg / 65535));
  }

  /** Multiply every element of `buf[0..count)` by `scale` in place. */
  scaleInplace(enc: GPUCommandEncoder, buf: GPUBuffer, count: number, scale: number): void {
    const paramBuf = new ArrayBuffer(16);
    new Uint32Array(paramBuf)[0] = count;
    new Float32Array(paramBuf)[1] = scale;
    const wg = Math.ceil(count / 256);
    this.dispatch(enc, this.pipelines.scaleInplace, [
      this.u16(new Uint32Array(paramBuf)),
      { binding: 1, resource: { buffer: buf } },
    ], Math.min(wg, 65535), Math.ceil(wg / 65535));
  }

  /** Apply final RMSNorm + logit projection (tied embeddings) + Gemma 4 final_logit_softcapping. */
  projectLogits(enc: GPUCommandEncoder, hiddenBuf: GPUBuffer, logitsBuf: GPUBuffer): void {
    this.catOverride = "logits";
    const s = this.scratch;
    this.rmsNorm(enc, hiddenBuf, this.weight("output_norm.weight"), s.normed, HIDDEN_SIZE);
    this.dumpForDebug(enc, s.normed, HIDDEN_SIZE, "result_norm");
    this.matmul(enc, this.weight("token_embd.weight"), s.normed, logitsBuf, VOCAB_SIZE, HIDDEN_SIZE);
    this.dumpForDebug(enc, logitsBuf, VOCAB_SIZE, "logits_pre_softcap");
    // Final logit softcap: logits = cap * tanh(logits / cap). Gemma 4 uses cap = 30.
    const softcapP = new ArrayBuffer(16);
    new Uint32Array(softcapP)[0] = VOCAB_SIZE;
    new Float32Array(softcapP)[1] = LOGIT_SOFTCAP;
    const softcapWG = Math.ceil(VOCAB_SIZE / 256);
    this.dispatch(enc, this.pipelines.softcap, [
      this.u16(new Uint32Array(softcapP)),
      { binding: 1, resource: { buffer: logitsBuf } },
    ], Math.min(softcapWG, 65535), Math.ceil(softcapWG / 65535));
    this.catOverride = null;
  }

  /** GPU argmax: find max index without reading full logits to CPU. Reads back 4 bytes instead of 1MB. */
  gpuArgmax(enc: GPUCommandEncoder, dataBuf: GPUBuffer, count: number, resultBuf: GPUBuffer): void {
    const nGroups = Math.min(Math.ceil(count / 256), 256);
    const params = this.u16(new Uint32Array([count, nGroups, 0, 0]));
    this.dispatch(enc, this.pipelines.argmaxPass1, [
      params,
      { binding: 1, resource: { buffer: dataBuf } },
      { binding: 2, resource: { buffer: resultBuf } },
    ], nGroups);
    // Pass2 doesn't read data (binding 1) — WGSL optimizes it out of the layout
    this.dispatch(enc, this.pipelines.argmaxPass2, [
      params,
      { binding: 2, resource: { buffer: resultBuf } },
    ], 1);
  }

  // ===========================================================================
  // Debug readbacks — used by the llama.cpp conformance test
  // ===========================================================================

  /** Conformance-test dumps. Key = tensor label, value = { first3, sum, full } after the next
   *  generateToken(tokenId) or prefill() run. Set to null to disable capture. Enable by
   *  assigning an empty Map before the forward pass. */
  public debugDumps: Map<string, { first3: number[]; sum: number; full: Float32Array }> | null = null;

  /** Copy a subrange of a GPU buffer into a newly created readback buffer and, when the
   *  parent command buffer is submitted, map it and store {first3, sum, full} under `label`
   *  in `debugDumps`. */
  private dumpForDebug(enc: GPUCommandEncoder, src: GPUBuffer, count: number, label: string): void {
    if (!this.debugDumps) return;
    const bytes = count * 4;
    const staging = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.endActivePass();
    enc.copyBufferToBuffer(src, 0, staging, 0, bytes);
    this._pendingDumps.push({ label, staging, count });
  }

  private _pendingDumps: { label: string; staging: GPUBuffer; count: number }[] = [];

  private async flushDebugDumps(): Promise<void> {
    if (!this.debugDumps || this._pendingDumps.length === 0) return;
    for (const { label, staging, count } of this._pendingDumps) {
      await staging.mapAsync(GPUMapMode.READ);
      const full = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      let sum = 0;
      for (let i = 0; i < count; i++) sum += full[i];
      const first3 = Array.from(full.slice(0, 3));
      this.debugDumps.set(label, { first3, sum, full });
      staging.destroy();
    }
    this._pendingDumps.length = 0;
  }

  // ===========================================================================
  // Generate (full token cycle)
  // ===========================================================================

  // ===========================================================================
  // Batch Prefill
  // ===========================================================================

  /**
   * Prefill N tokens through all 35 layers, returning the argmax of the final
   * token's logits.
   *
   * Strategy: run the same single-token forward pass (runLayer) for every token,
   * but pack CHUNK tokens per command encoder + submit. That avoids the per-token
   * mapAsync sync that dominates true one-token-at-a-time generation. The very
   * last token goes through the standard generateToken path so we get argmax.
   *
   * Correctness: runLayer is known-good against llama.cpp (conformance tests pass
   * within 1%). queueForwardToken just reuses runLayer without logits/argmax, so
   * prefill picks up the same correctness guarantees for free — no dedicated
   * runBatchLayer to keep in sync.
   */
  async prefill(tokenIds: number[], onProgress?: (done: number, total: number) => void, _onStatus?: (msg: string) => void): Promise<number> {
    const N = tokenIds.length;
    if (N === 0) return 0;
    if (N === 1) return this.generateToken(tokenIds[0]);

    // Chunked batched prefill: each CHUNK tokens run through ONE batched
    // forward pass (runLayerBatched) so the matmul weight bandwidth is
    // amortized across all CHUNK activation vectors. At CHUNK=8 the
    // decodeBatchGenuine microbenchmark shows 1.76× per-token speedup
    // vs sequential generateToken, and for prefill all N tokens are
    // known + 100% accepted so the speedup translates directly into
    // a faster startup.
    //
    // CHUNK is capped at the batched matmul's max N (8). Per forward
    // pass we issue far fewer dispatches than the sequential CHUNK=16
    // path (140 matmul passes → ~35 batched matmul dispatches per
    // layer), so the uniform megabuffer has plenty of headroom.
    const CHUNK = 8;
    const d = this.device;
    this.ensureTokenBufs();

    // Forward-pass-only path for tokens [0..N-1). Batched forward passes
    // skip the per-slot logits/argmax work since we only need the final
    // token's argmax — each chunk just runs the 35-layer forward pass.
    for (let start = 0; start < N - 1; start += CHUNK) {
      const end = Math.min(start + CHUNK, N - 1);
      const B = end - start;
      const enc = d.createCommandEncoder();
      this.uniforms.reset();
      const chunkIds = new Uint32Array(B);
      for (let i = 0; i < B; i++) chunkIds[i] = tokenIds[start + i];
      d.queue.writeBuffer(this._tokenIdBuf!, 0, chunkIds);

      this.queueBatchedForwardPass(enc, B);
      this.position += B;

      this.endActivePass();
      this.uniforms.flush();
      d.queue.submit([enc.finish()]);
      onProgress?.(end, N);
      if (end < N - 1) await new Promise(r => setTimeout(r, 0));
    }

    // Final token runs through generateToken so we get its argmax back.
    const last = await this.generateToken(tokenIds[N - 1]);
    onProgress?.(N, N);
    console.log(`[engine] Prefill: ${N} tokens via batched forward-pass (CHUNK=${CHUNK})`);
    return last;
  }

  /** Append a single batched forward pass for B slots to `enc`. Reads token
   *  ids from _tokenIdBuf[0..B), runs embed → per-layer inputs → 35-layer
   *  batched loop, and leaves the final hidden state at s.hiddenB[slot *
   *  HIDDEN_SIZE] (odd layer count). Does NOT compute logits/argmax and
   *  does NOT update this.position — caller bumps it. */
  private queueBatchedForwardPass(enc: GPUCommandEncoder, B: number): void {
    const s = this.scratch;
    const positionStart = this.position;

    // Embed B tokens into slot-offset regions of s.hiddenA.
    for (let slot = 0; slot < B; slot++) {
      const off = slot * HIDDEN_SIZE * 4;
      this.dispatch(enc, this.pipelines.getRowsQ4k, [
        this.u16(new Uint32Array([HIDDEN_SIZE, slot, 0, 0])),
        { binding: 1, resource: { buffer: this.weight("token_embd.weight") } },
        { binding: 2, resource: { buffer: s.hiddenA, offset: off, size: HIDDEN_SIZE * 4 } },
        { binding: 3, resource: { buffer: this._tokenIdBuf! } },
      ], Math.ceil(HIDDEN_SIZE / 256));
      this.scaleInplaceOffset(enc, s.hiddenA, off, HIDDEN_SIZE, Math.sqrt(HIDDEN_SIZE));
    }

    // Per-layer inputs (one call per slot with slotOffset).
    for (let slot = 0; slot < B; slot++) {
      this.computePerLayerInputs(enc, slot, s.hiddenA, slot, slot * HIDDEN_SIZE * 4);
    }

    // 35-layer batched loop.
    for (let layer = 0; layer < NUM_LAYERS; layer++) {
      const inputBuf = (layer % 2 === 0) ? s.hiddenA : s.hiddenB;
      const outputBuf = (layer % 2 === 0) ? s.hiddenB : s.hiddenA;
      this.runLayerBatched(enc, layer, inputBuf, outputBuf, positionStart, B);
    }
  }

  /**
   * Append forward-pass dispatches for a single token to `enc`. Reuses runLayer so
   * the computation matches the single-token decode path bit-for-bit, but omits
   * logits/argmax and does NOT reset uniforms, flush, or submit. Caller owns the
   * encoder lifecycle (used by prefill to batch multiple tokens per submit).
   */
  private queueForwardToken(enc: GPUCommandEncoder, slot: number): void {
    const s = this.scratch;
    this.embedToken(enc, slot, s.hiddenA);
    this.computePerLayerInputs(enc, slot, s.hiddenA);
    for (let layer = 0; layer < NUM_LAYERS; layer++) {
      const input = (layer % 2 === 0) ? s.hiddenA : s.hiddenB;
      const output = (layer % 2 === 0) ? s.hiddenB : s.hiddenA;
      this.runLayer(enc, layer, input, output, this.position);
    }
    this.position++;
  }

  /** Wait for all submitted GPU work to complete. */
  async waitForGPU(): Promise<void> {
    await this.device.queue.onSubmittedWorkDone();
  }

  async generateToken(tokenId: number): Promise<number> {
    const s = this.scratch;
    const d = this.device;
    this.profiler?.reset();
    const enc = d.createCommandEncoder();
    this.uniforms.reset();

    this.ensureTokenBufs();
    // Seed the GPU-resident token id with the caller-supplied value so that
    // getRowsQ4k/Q5k pick up this token on the embed step.
    d.queue.writeBuffer(this._tokenIdBuf!, 0, new Uint32Array([tokenId]));

    // 1. Embed → 35 layers → logits → GPU argmax (all in one encoder)
    this.embedToken(enc, 0, s.hiddenA);
    this.dumpForDebug(enc, s.hiddenA, HIDDEN_SIZE, "inp_scaled");
    this.computePerLayerInputs(enc, 0, s.hiddenA);
    for (let layer = 0; layer < NUM_LAYERS; layer++) {
      const input = (layer % 2 === 0) ? s.hiddenA : s.hiddenB;
      const output = (layer % 2 === 0) ? s.hiddenB : s.hiddenA;
      this.runLayer(enc, layer, input, output, this.position);
      if (layer === 0) {
        this.dumpForDebug(enc, output, HIDDEN_SIZE, "layer0_out");
      }
    }

    this.projectLogits(enc, s.hiddenB, this._logitsBuf!);
    this.dumpForDebug(enc, this._logitsBuf!, VOCAB_SIZE, "result_output");
    this.maskLogits(enc, this._logitsBuf!);
    this.gpuArgmax(enc, this._logitsBuf!, VOCAB_SIZE, this._argmaxBuf!);

    // Read back 4 bytes (argmax index) instead of 1MB of logits
    this.endActivePass();
    enc.copyBufferToBuffer(this._argmaxBuf!, 0, this._stagingBuf!, 0, 4);
    this.profiler?.resolve(enc);
    this.uniforms.flush();
    d.queue.submit([enc.finish()]);
    this.position++;

    // Map staging buffers in parallel (argmax + profiler)
    let tokenIdx: number;
    if (this.profiler) {
      await Promise.all([this._stagingBuf!.mapAsync(GPUMapMode.READ), this.profiler.mapAsync()]);
      tokenIdx = new Uint32Array(this._stagingBuf!.getMappedRange().slice(0))[0];
      this._stagingBuf!.unmap();
      this.lastProfile = this.profiler.read();
    } else {
      await this._stagingBuf!.mapAsync(GPUMapMode.READ);
      tokenIdx = new Uint32Array(this._stagingBuf!.getMappedRange().slice(0))[0];
      this._stagingBuf!.unmap();
    }
    await this.flushDebugDumps();
    return tokenIdx;
  }

  /**
   * Speculative-decode primitive: run `tokenIds` as a sequential chain of
   * forward passes in one encoder, compute logits + argmax at every
   * position, and return the B argmax values in order.
   *
   * Semantics: on entry, `this.position` is P and the KV cache has entries
   * 0..P-1. On return, `this.position` is P + B and the cache has new
   * entries at P, P+1, ..., P+B-1 (one per input token). The returned
   * `result[i]` is the model's argmax-next-token prediction conditioned
   * on the KV state after tokens[0..i] have been processed.
   *
   * Single-encoder execution guarantees: the per-slot `projectLogits +
   * gpuArgmax + copyBufferToBuffer` chain runs in-order inside one
   * compute pass, so the copy into `_batchArgmaxBuf[slot * 4]` lands
   * before the next slot's projectLogits overwrites `s.normed`,
   * `_logitsBuf`, and `_argmaxBuf`.
   *
   * Caller is responsible for calling `rollbackKV(P + accepted)` if only
   * a prefix of the batch is accepted by speculative verification.
   */
  async decodeBatch(tokenIds: number[]): Promise<number[]> {
    if (tokenIds.length === 0) return [];
    if (tokenIds.length === 1) return [await this.generateToken(tokenIds[0])];
    if (tokenIds.length > InferenceEngine.MAX_SPEC_BATCH) {
      throw new Error(`decodeBatch: batch size ${tokenIds.length} exceeds MAX_SPEC_BATCH=${InferenceEngine.MAX_SPEC_BATCH}`);
    }

    const s = this.scratch;
    const d = this.device;
    const B = tokenIds.length;

    this.ensureTokenBufs();
    // Seed all B token ids at slots 0..B-1 of _tokenIdBuf.
    d.queue.writeBuffer(this._tokenIdBuf!, 0, new Uint32Array(tokenIds));

    const enc = d.createCommandEncoder();
    this.uniforms.reset();

    for (let slot = 0; slot < B; slot++) {
      // Forward pass for this slot. embedToken reads token id from
      // _tokenIdBuf[slot], runs all 35 layers, lands the final hidden
      // state in hiddenB (last layer index 34 is even → output hiddenB).
      this.embedToken(enc, slot, s.hiddenA);
      this.computePerLayerInputs(enc, slot, s.hiddenA);
      for (let layer = 0; layer < NUM_LAYERS; layer++) {
        const input = (layer % 2 === 0) ? s.hiddenA : s.hiddenB;
        const output = (layer % 2 === 0) ? s.hiddenB : s.hiddenA;
        this.runLayer(enc, layer, input, output, this.position);
      }
      this.position++;

      // Logits + argmax for this slot. s.normed / _logitsBuf / _argmaxBuf
      // are shared scratch; WebGPU's in-order execution inside one compute
      // pass guarantees the copy below sees this slot's argmax before the
      // next iteration starts overwriting the scratch.
      this.projectLogits(enc, s.hiddenB, this._logitsBuf!);
      this.gpuArgmax(enc, this._logitsBuf!, VOCAB_SIZE, this._argmaxBuf!);
      this.endActivePass();
      enc.copyBufferToBuffer(this._argmaxBuf!, 0, this._batchArgmaxBuf!, slot * 4, 4);
    }

    // One-shot readback of all B argmax values.
    this.endActivePass();
    enc.copyBufferToBuffer(this._batchArgmaxBuf!, 0, this._batchStagingBuf!, 0, B * 4);
    this.uniforms.flush();
    d.queue.submit([enc.finish()]);

    await this._batchStagingBuf!.mapAsync(GPUMapMode.READ);
    const result = Array.from(new Uint32Array(this._batchStagingBuf!.getMappedRange(0, B * 4).slice(0)));
    this._batchStagingBuf!.unmap();
    return result;
  }

  /**
   * Genuinely-batched decode: runs all N tokens through one batched forward
   * pass that uses matmulBatched for Q/K/V/attn_out/gate/up/down. Distinct
   * from `decodeBatch` above which just enqueues N sequential runLayer calls
   * into one encoder — same kernel work but pipelined CPU→GPU, no actual
   * amortization of weight loads across slots.
   *
   * For speculative decoding this is the path that matters: matmul weight
   * bandwidth is shared across N activations, so the matmul category (50%
   * of per-token time) amortizes by ~1.85× at N=4.
   *
   * Attention is still per-slot sequential inside runLayerBatched because
   * each slot has a different causal range and the tq_scores / tq_wsum
   * kernels aren't yet batched over queries. That's the next kernel to
   * rewrite if this path shows any speedup.
   */
  async decodeBatchGenuine(tokenIds: number[]): Promise<number[]> {
    if (tokenIds.length === 0) return [];
    if (tokenIds.length > InferenceEngine.MAX_SPEC_BATCH) {
      throw new Error(`decodeBatchGenuine: batch size ${tokenIds.length} exceeds MAX_SPEC_BATCH=${InferenceEngine.MAX_SPEC_BATCH}`);
    }

    const s = this.scratch;
    const d = this.device;
    const B = tokenIds.length;

    this.ensureTokenBufs();
    d.queue.writeBuffer(this._tokenIdBuf!, 0, new Uint32Array(tokenIds));

    const enc = d.createCommandEncoder();
    this.uniforms.reset();
    const positionStart = this.position;

    // 1. Embed each slot into hiddenA[slot * HIDDEN_SIZE] via an offset
    //    binding on the output. Each dispatch reads _tokenIdBuf[slot].
    for (let slot = 0; slot < B; slot++) {
      const off = slot * HIDDEN_SIZE * 4;
      this.dispatch(enc, this.pipelines.getRowsQ4k, [
        this.u16(new Uint32Array([HIDDEN_SIZE, slot, 0, 0])),
        { binding: 1, resource: { buffer: this.weight("token_embd.weight") } },
        { binding: 2, resource: { buffer: s.hiddenA, offset: off, size: HIDDEN_SIZE * 4 } },
        { binding: 3, resource: { buffer: this._tokenIdBuf! } },
      ], Math.ceil(HIDDEN_SIZE / 256));
      this.scaleInplaceOffset(enc, s.hiddenA, off, HIDDEN_SIZE, Math.sqrt(HIDDEN_SIZE));
    }

    // 2. Per-layer inputs for each slot — each call reads the slot's
    //    inpScaled slice via inpScaledOffset and writes its PER_LAYER_TOTAL
    //    block via slotOffset.
    for (let slot = 0; slot < B; slot++) {
      this.computePerLayerInputs(enc, slot, s.hiddenA, slot, slot * HIDDEN_SIZE * 4);
    }

    // 3. Layer loop: runLayerBatched processes all N slots in one sweep.
    for (let layer = 0; layer < NUM_LAYERS; layer++) {
      const inputBuf = (layer % 2 === 0) ? s.hiddenA : s.hiddenB;
      const outputBuf = (layer % 2 === 0) ? s.hiddenB : s.hiddenA;
      this.runLayerBatched(enc, layer, inputBuf, outputBuf, positionStart, B);
    }
    this.position += B;

    // 4. Logits + argmax for each slot. NUM_LAYERS = 35 is odd → layer 34
    //    is even → reads hiddenA, writes hiddenB → final output is in hiddenB.
    //    projectLogits reads hiddenBuf and uses s.normed as internal scratch,
    //    so we must copy each slot's hidden into a staging buffer that is
    //    NOT s.normed. s.attnPostNormed is HIDDEN_SIZE-sized and unused at
    //    this point in the pass.
    const finalBuf = s.hiddenB;
    for (let slot = 0; slot < B; slot++) {
      this.endActivePass();
      enc.copyBufferToBuffer(finalBuf, slot * HIDDEN_SIZE * 4, s.attnPostNormed, 0, HIDDEN_SIZE * 4);
      this.projectLogits(enc, s.attnPostNormed, this._logitsBuf!);
      this.gpuArgmax(enc, this._logitsBuf!, VOCAB_SIZE, this._argmaxBuf!);
      this.endActivePass();
      enc.copyBufferToBuffer(this._argmaxBuf!, 0, this._batchArgmaxBuf!, slot * 4, 4);
    }

    this.endActivePass();
    enc.copyBufferToBuffer(this._batchArgmaxBuf!, 0, this._batchStagingBuf!, 0, B * 4);
    this.uniforms.flush();
    d.queue.submit([enc.finish()]);

    await this._batchStagingBuf!.mapAsync(GPUMapMode.READ);
    const result = Array.from(new Uint32Array(this._batchStagingBuf!.getMappedRange(0, B * 4).slice(0)));
    this._batchStagingBuf!.unmap();
    return result;
  }

  /**
   * Roll back the KV cache and position counter to `targetPosition`.
   * Used by speculative decode after a partial acceptance: the forward
   * passes we ran for the rejected draft tokens left KV entries in the
   * cache that must be discarded before the next decode step.
   *
   * Since the TQ cache is append-only f16 compressed data with a
   * `length` sentinel per layer, rollback is just "lower the length"
   * — no zeroing, no memset. Future writes at the same positions
   * overwrite the stale entries.
   */
  rollbackKV(targetPosition: number): void {
    if (targetPosition > this.position) {
      throw new Error(`rollbackKV: target ${targetPosition} exceeds current position ${this.position}`);
    }
    if (targetPosition < 0) {
      throw new Error(`rollbackKV: target ${targetPosition} is negative`);
    }
    for (const [, cache] of this.kvCache) {
      if (cache.length > targetPosition) cache.length = targetPosition;
    }
    this.position = targetPosition;
  }

  private _logitsBuf: GPUBuffer | null = null;
  private _argmaxBuf: GPUBuffer | null = null;
  private _stagingBuf: GPUBuffer | null = null;
  // GPU-resident token id. Holds the row index read by get-rows-q4k/q5k in the
  // embed step. queue.writeBuffer seeds it for the first token; subsequent
  // tokens can chain it from argmaxBuf via copyBufferToBuffer to avoid CPU sync.
  private _tokenIdBuf: GPUBuffer | null = null;
  // Speculative decode batch slots: each slot holds one u32 argmax that we
  // copy from `_argmaxBuf[0]` after each per-slot projectLogits + gpuArgmax
  // inside a batched decode encoder. Sized for up to MAX_SPEC_BATCH tokens.
  private static readonly MAX_SPEC_BATCH = 16;
  private _batchArgmaxBuf: GPUBuffer | null = null;
  private _batchStagingBuf: GPUBuffer | null = null;
  // Streaming pipeline state: K staging buffers (readback of each token's argmax)
  // and K UniformMega instances so back-to-back encoders don't clobber each
  // other's uniform writes. K=4 is enough to keep the GPU queue full: by the
  // time we drain slot N, the GPU has long finished N's encoder, so mapAsync
  // resolves immediately and we avoid the ~5 ms sync stall K=2 still pays.
  private static readonly STREAM_DEPTH = 4;
  private _streamStagings: GPUBuffer[] | null = null;
  private _uniformsPool: UniformMega[] | null = null;

  // Constrained-decoding logit mask state. `_maskBuf` holds `_maskCount`
  // concatenated bitmaps, each `_maskWords` u32s long. `_maskActiveIndex`
  // selects which bitmap the next mask dispatch uses; `_maskEnabled` gates
  // the dispatch entirely. Callers drive state transitions via
  // `setLogitMaskIndex` (worker-side grammar state machine in our case).
  private _maskBuf: GPUBuffer | null = null;
  private _maskWords: number = 0;
  private _maskCount: number = 0;
  private _maskEnabled: boolean = false;
  private _maskActiveIndex: number = 0;

  private ensureTokenBufs(): void {
    const d = this.device;
    if (!this._logitsBuf) {
      this._logitsBuf = d.createBuffer({ size: VOCAB_SIZE * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      // Argmax intermediate: 256 groups × 2 u32 = 2KB (index + value pairs)
      this._argmaxBuf = d.createBuffer({ size: 256 * 2 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      this._stagingBuf = d.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      // 64 slots × 4 bytes: sized for batch prefill's CHUNK=16 with 4× headroom.
      // Also holds the single-slot stream chain used by generateToken.
      this._tokenIdBuf = d.createBuffer({ size: 64 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
      // Speculative decode batch argmax buffers: MAX_SPEC_BATCH × 4 bytes.
      const batchBytes = InferenceEngine.MAX_SPEC_BATCH * 4;
      this._batchArgmaxBuf = d.createBuffer({ size: batchBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      this._batchStagingBuf = d.createBuffer({ size: batchBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    }
  }

  private ensureStreamBufs(): void {
    const d = this.device;
    if (this._streamStagings) return;
    const K = InferenceEngine.STREAM_DEPTH;
    this._streamStagings = [];
    this._uniformsPool = [];
    for (let i = 0; i < K; i++) {
      this._streamStagings.push(d.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }));
      // Per-encoder UniformMega. 2048 slots × 256 B = 512 KB each, enough for
      // ~8 full forward passes — plenty of headroom for one encoder.
      this._uniformsPool.push(new UniformMega(d, 2048));
    }
  }

  /**
   * Upload `bitmapCount` concatenated grammar bitmaps. Each bitmap is
   * `ceil(VOCAB_SIZE / 32)` u32s with bit `i` = 1 meaning token `i` is
   * allowed. Call once per grammar definition; reuse across many tokens.
   */
  uploadMaskBitmaps(combined: Uint32Array, bitmapCount: number): void {
    if (combined.length % bitmapCount !== 0) {
      throw new Error(`mask length ${combined.length} not divisible by bitmapCount ${bitmapCount}`);
    }
    this._maskWords = combined.length / bitmapCount;
    this._maskCount = bitmapCount;
    const bytes = combined.byteLength;
    if (!this._maskBuf || this._maskBuf.size < bytes) {
      this._maskBuf?.destroy();
      this._maskBuf = this.device.createBuffer({
        size: bytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    }
    this.device.queue.writeBuffer(this._maskBuf, 0, combined.buffer as ArrayBuffer, combined.byteOffset, combined.byteLength);
  }

  /** Enable masking with bitmap `idx` for subsequent forward passes. */
  setLogitMaskIndex(idx: number): void {
    if (!this._maskBuf) throw new Error("mask bitmaps not uploaded");
    if (idx < 0 || idx >= this._maskCount) {
      throw new Error(`mask index ${idx} out of range [0, ${this._maskCount})`);
    }
    this._maskActiveIndex = idx;
    this._maskEnabled = true;
  }

  disableLogitMask(): void { this._maskEnabled = false; }

  /** Zero out disallowed logits in-place before argmax. No-op when disabled. */
  private maskLogits(enc: GPUCommandEncoder, logitsBuf: GPUBuffer): void {
    if (!this._maskEnabled || !this._maskBuf) return;
    const offset = this._maskActiveIndex * this._maskWords;
    const params = this.u16(new Uint32Array([VOCAB_SIZE, offset, 0, 0]));
    const wg = Math.ceil(VOCAB_SIZE / 256);
    this.dispatch(enc, this.pipelines.logitMask, [
      params,
      { binding: 1, resource: { buffer: logitsBuf } },
      { binding: 2, resource: { buffer: this._maskBuf } },
    ], wg);
  }

  /**
   * Pipelined streaming decode.
   *
   * Submits encoder N+1 before waiting for encoder N's argmax readback, so the
   * GPU queue is never starved between tokens. Uses a ring of two staging
   * buffers + two uniform megabuffers (primary + alt), alternating per step.
   *
   * Chain mechanism: at the end of each encoder, argmaxBuf[0] (the token id the
   * GPU just picked) is copied via copyBufferToBuffer into tokenIdBuf[0]. The
   * next encoder's embedToken reads row from tokenIdBuf[0] — no CPU roundtrip.
   *
   * `onToken` fires as each token's readback resolves. Its `await` is honored
   * (the loop waits before submitting more work), so callers can backpressure.
   * Stopping on EOS: the K in-flight tokens past the stop are discarded silently.
   */
  async streamTokens(
    initialTokenId: number,
    maxTokens: number,
    eosSet: Set<number>,
    onToken: (tokenId: number, stats: ReturnType<InferenceEngine["getStats"]>) => Promise<boolean | void> | boolean | void,
  ): Promise<void> {
    if (maxTokens <= 0) return;
    const d = this.device;
    const s = this.scratch;
    this.ensureTokenBufs();
    this.ensureStreamBufs();

    const stagings = this._streamStagings!;
    const uniformsPool = this._uniformsPool!;
    const K = InferenceEngine.STREAM_DEPTH;
    const savedUniforms = this.uniforms;

    // Seed the first token id into slot 0.
    d.queue.writeBuffer(this._tokenIdBuf!, 0, new Uint32Array([initialTokenId]));

    type InFlight = { staging: GPUBuffer };
    const inFlight: InFlight[] = [];
    let stopped = false;

    const drainOldest = async (): Promise<void> => {
      const f = inFlight.shift()!;
      await f.staging.mapAsync(GPUMapMode.READ);
      const tok = new Uint32Array(f.staging.getMappedRange().slice(0))[0];
      f.staging.unmap();
      if (stopped) return; // already past EOS — drop trailing speculations
      const cancel = await onToken(tok, this.getStats());
      if (cancel === true || eosSet.has(tok)) stopped = true;
    };

    try {
      for (let i = 0; i < maxTokens; i++) {
        if (stopped) break;
        if (inFlight.length >= K) await drainOldest();
        if (stopped) break;

        const slotIdx = i % K;
        const uniforms = uniformsPool[slotIdx];
        this.uniforms = uniforms;
        uniforms.reset();

        const enc = d.createCommandEncoder();
        this.embedToken(enc, 0, s.hiddenA);
        this.computePerLayerInputs(enc, 0, s.hiddenA);
        for (let layer = 0; layer < NUM_LAYERS; layer++) {
          const input = (layer % 2 === 0) ? s.hiddenA : s.hiddenB;
          const output = (layer % 2 === 0) ? s.hiddenB : s.hiddenA;
          this.runLayer(enc, layer, input, output, this.position);
        }
        this.projectLogits(enc, s.hiddenB, this._logitsBuf!);
        this.maskLogits(enc, this._logitsBuf!);
        this.gpuArgmax(enc, this._logitsBuf!, VOCAB_SIZE, this._argmaxBuf!);
        // Chain: next encoder's embed reads tokenIdBuf[0], which we update from
        // this encoder's argmax result. Both copies are sequenced in the queue.
        this.endActivePass();
        enc.copyBufferToBuffer(this._argmaxBuf!, 0, this._tokenIdBuf!, 0, 4);
        enc.copyBufferToBuffer(this._argmaxBuf!, 0, stagings[slotIdx], 0, 4);

        uniforms.flush();
        d.queue.submit([enc.finish()]);
        this.position++;
        inFlight.push({ staging: stagings[slotIdx] });
      }
      while (inFlight.length > 0) await drainOldest();
    } finally {
      this.uniforms = savedUniforms;
    }
  }

  // ===========================================================================
  // Stats
  // ===========================================================================

  getStats(): { positions: number; compressedMB: number; uncompressedMB: number; ratio: number } {
    let maxLen = 0;
    let compBytes = 0;
    let rawBytes = 0;
    for (const [layer, c] of this.kvCache) {
      if (c.length > maxLen) maxLen = c.length;
      const dim = this.headDim(layer);
      const kPolarWords = polarWordsPerPos(dim, K_POLAR_CONFIG);
      const vPolarWords = polarWordsPerPos(dim, V_POLAR_CONFIG);
      const kQjlWords = qjlWordsPerPos(dim, K_POLAR_CONFIG);
      const vQjlWords = qjlWordsPerPos(dim, V_POLAR_CONFIG);
      const kPerPos = (kPolarWords + kQjlWords) * 4 + 8;
      const vPerPos = (vPolarWords + vQjlWords) * 4 + 8;
      compBytes += c.length * (kPerPos + vPerPos);
      rawBytes += c.length * dim * 2 * 2;
    }
    return {
      positions: maxLen,
      compressedMB: compBytes / (1024 * 1024),
      uncompressedMB: rawBytes / (1024 * 1024),
      ratio: rawBytes > 0 ? rawBytes / compBytes : 0,
    };
  }
}

// =============================================================================
// TQ Rotation Matrix — randomized Hadamard transform (RHT)
// =============================================================================
//
// H @ D where H is the Walsh-Hadamard matrix (dim × dim, entries ±1/sqrt(dim))
// and D is a diagonal sign flip derived from `seed`. Hadamard is the optimal
// outlier-dispersion rotation for LLM activations: every value of the input
// contributes ±1/sqrt(dim) to every output, so a single 50× outlier (e.g. the
// Gemma V "attention sink" dim) gets spread uniformly instead of dominating
// max_r after rotation. The random sign flip D breaks symmetries so repeated
// inputs don't collide on the Hadamard structure.
//
// Requires dim to be a power of two (matches Gemma 4 E2B head dims 256/512).

function generateHadamardSigns(dim: number, seed: number): Int32Array {
  let rng = BigInt(seed) || 1n;
  const nextU32 = (): number => {
    rng ^= (rng << 13n) & 0xFFFFFFFFFFFFFFFFn;
    rng ^= (rng >> 7n) & 0xFFFFFFFFFFFFFFFFn;
    rng ^= (rng << 17n) & 0xFFFFFFFFFFFFFFFFn;
    rng &= 0xFFFFFFFFFFFFFFFFn;
    return Number(rng & 0xFFFFFFFFn);
  };
  const signs = new Int32Array(dim);
  for (let i = 0; i < dim; i++) signs[i] = (nextU32() & 1) === 0 ? 1 : -1;
  return signs;
}

function generateRotationMatrix(dim: number, seed: number): Float32Array {
  if ((dim & (dim - 1)) !== 0) throw new Error(`Hadamard rotation requires power-of-two dim, got ${dim}`);

  // xorshift64 PRNG for reproducible random signs.
  let rng = BigInt(seed) || 1n;
  const nextU32 = (): number => {
    rng ^= (rng << 13n) & 0xFFFFFFFFFFFFFFFFn;
    rng ^= (rng >> 7n) & 0xFFFFFFFFFFFFFFFFn;
    rng ^= (rng << 17n) & 0xFFFFFFFFFFFFFFFFn;
    rng &= 0xFFFFFFFFFFFFFFFFn;
    return Number(rng & 0xFFFFFFFFn);
  };
  const signs = new Int8Array(dim);
  for (let i = 0; i < dim; i++) signs[i] = (nextU32() & 1) === 0 ? 1 : -1;

  // Build H[k, i] = (-1)^{popcount(k & i)} / sqrt(dim), then premultiply column i by signs[i].
  const mat = new Float32Array(dim * dim);
  const scale = 1 / Math.sqrt(dim);
  for (let k = 0; k < dim; k++) {
    for (let i = 0; i < dim; i++) {
      // popcount of (k & i)
      let bits = k & i;
      bits = bits - ((bits >>> 1) & 0x55555555);
      bits = (bits & 0x33333333) + ((bits >>> 2) & 0x33333333);
      bits = (bits + (bits >>> 4)) & 0x0F0F0F0F;
      const parity = (bits * 0x01010101) >>> 24;
      const s = (parity & 1) === 0 ? 1 : -1;
      mat[k * dim + i] = s * signs[i] * scale;
    }
  }
  return mat;
}

// FNV-1a 32-bit hash over a u32 sequence. Used to fingerprint a system prompt
// token stream so loadCache() can detect stale caches.
function fnv1aU32(values: number[]): number {
  let h = 0x811C9DC5;
  for (let i = 0; i < values.length; i++) {
    const v = values[i] >>> 0;
    for (let s = 0; s < 32; s += 8) {
      h ^= (v >>> s) & 0xFF;
      h = Math.imul(h, 0x01000193) >>> 0;
    }
  }
  return h >>> 0;
}

