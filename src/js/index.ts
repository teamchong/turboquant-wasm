/**
 * TurboQuant — WASM SIMD vector compression (3 bits/dim)
 *
 * Architecture:
 *   JS Float32Array -> [Zig WASM + relaxed SIMD] -> compressed bytes
 *
 * The Zig engine (turboquant.wasm) compresses vectors using polar + QJL
 * quantization with Gaussian QR rotation, all SIMD-accelerated.
 *
 * Browser requirements:
 *   - WASM SIMD128:     Chrome 91+, Firefox 89+, Safari 16.4+
 *   - WASM Relaxed SIMD: Chrome 114+, Firefox 128+, Safari 18+
 *
 * Usage:
 *   import { TurboQuant } from "turboquant";
 *   const tq = await TurboQuant.init({ dim: 1024, seed: 42 });
 *   const compressed = tq.encode(myFloat32Array);
 *   const score = tq.dot(queryVector, compressed);
 *   tq.destroy();
 */

import type { TQGpuIndex } from "./gpu-index.js";
import { wasmBase64 } from "./turboquant-wasm.generated.js";

function decodeBase64(b64: string): Uint8Array {
  if (typeof Buffer !== "undefined") {
    const buf = Buffer.from(b64, "base64");
    return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  }
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

// --- WASM export types ---

interface TurboQuantExports {
  memory: WebAssembly.Memory;
  tq_engine_create(dim: number, seed: number): number;
  tq_engine_destroy(handle: number): void;
  tq_encode(handle: number, inputPtr: number, dim: number, outLenPtr: number): number;
  tq_decode(handle: number, compPtr: number, compLen: number, outLenPtr: number): number;
  tq_dot(handle: number, queryPtr: number, dim: number, compPtr: number, compLen: number): number;
  tq_dot_batch(handle: number, queryPtr: number, dim: number, compPtr: number, bytesPerVector: number, numVectors: number, outScoresPtr: number): void;
  tq_rotate_query(handle: number, queryPtr: number, dim: number, outPtr: number): void;
  tq_alloc(len: number): number;
  tq_free(ptr: number, len: number): void;
  tq_alloc_f32(count: number): number;
  tq_free_f32(ptr: number, count: number): void;
  tq_stream_create(engineHandle: number, maxPositions: number): number;
  tq_stream_destroy(handle: number): void;
  tq_stream_append(handle: number, vectorPtr: number, dim: number): number;
  tq_stream_append_batch(handle: number, vectorsPtr: number, dim: number, count: number): number;
  tq_stream_get_compressed(handle: number, outLenPtr: number): number;
  tq_stream_decode_position(handle: number, position: number, outPtr: number, dim: number): number;
  tq_stream_rewind(handle: number, position: number): void;
  tq_stream_length(handle: number): number;
  tq_stream_bytes_per_vector(handle: number): number;
}

// --- Singleton WASM instance ---

let wasmInstance: TurboQuantExports | null = null;
let wasmReady: Promise<TurboQuantExports> | null = null;

async function getWasm(): Promise<TurboQuantExports> {
  if (wasmInstance) return wasmInstance;
  if (wasmReady) return wasmReady;

  wasmReady = (async () => {
    const bytes = decodeBase64(wasmBase64);
    const module = await WebAssembly.compile(bytes as BufferSource);
    const instance = await WebAssembly.instantiate(module, {});
    wasmInstance = instance.exports as unknown as TurboQuantExports;
    return wasmInstance;
  })();

  return wasmReady;
}

/**
 * Load from an external WASM source instead of the embedded binary.
 * Useful for streaming instantiation in browsers:
 *   TurboQuant.init({ dim: 1024, seed: 42, wasm: fetch("/turboquant.wasm") })
 */
async function getWasmFrom(
  source: Response | Promise<Response> | BufferSource,
): Promise<TurboQuantExports> {
  let instance: WebAssembly.Instance;
  if (source instanceof Response || source instanceof Promise) {
    const r = await WebAssembly.instantiateStreaming(source, {});
    instance = r.instance;
  } else {
    const r = await WebAssembly.instantiate(source, {});
    instance = (r as unknown as WebAssembly.WebAssemblyInstantiatedSource).instance;
  }
  wasmInstance = instance.exports as unknown as TurboQuantExports;
  return wasmInstance;
}

// --- Helpers ---

function wasmWriteF32(ex: TurboQuantExports, data: Float32Array): number {
  const ptr = ex.tq_alloc(data.byteLength);
  if (!ptr) throw new Error("TurboQuant: WASM alloc failed");
  new Float32Array(ex.memory.buffer, ptr, data.length).set(data);
  return ptr;
}

function wasmWriteU8(ex: TurboQuantExports, data: Uint8Array): number {
  const ptr = ex.tq_alloc(data.length);
  if (!ptr) throw new Error("TurboQuant: WASM alloc failed");
  // Re-read ex.memory.buffer AFTER alloc — it may have grown and detached the old buffer.
  new Uint8Array(ex.memory.buffer, ptr, data.length).set(data);
  return ptr;
}

function wasmAllocU32(ex: TurboQuantExports): number {
  const ptr = ex.tq_alloc(4);
  if (!ptr) throw new Error("TurboQuant: WASM alloc failed");
  return ptr;
}

function wasmReadU32(ex: TurboQuantExports, ptr: number): number {
  return new Uint32Array(ex.memory.buffer, ptr, 1)[0];
}

// --- Public API ---

export interface TurboQuantConfig {
  /** Vector dimension (must be a power of 2). */
  dim: number;
  /** Deterministic seed for rotation matrix. */
  seed: number;
  /** Optional external WASM source (Response, Promise<Response>, or ArrayBuffer). */
  wasm?: Response | Promise<Response> | BufferSource;
}

export class TurboQuant {
  readonly dim: number;
  #handle: number;
  #ex: TurboQuantExports;
  #cachedRef: Uint8Array | null = null;
  #cachedPtr: number = 0;
  #cachedLen: number = 0;
  #cachedBpv: number = 0;
  #scoresPtr: number = 0;
  #scoresCount: number = 0;
  #scoresOut: Float32Array | null = null;
  #queryPtr: number = 0;
  #queryLen: number = 0;
  #rotOutPtr: number = 0;
  #gpuIndex: TQGpuIndex | null = null;
  #gpuChecked = false;
  #gpuDataRef: Uint8Array | null = null;

  private constructor(ex: TurboQuantExports, handle: number, dim: number) {
    this.#ex = ex;
    this.#handle = handle;
    this.dim = dim;
  }

  /**
   * Create a TurboQuant engine.
   *
   * ```ts
   * const tq = await TurboQuant.init({ dim: 1024, seed: 42 });
   * ```
   */
  static async init(config: TurboQuantConfig): Promise<TurboQuant> {
    const ex = config.wasm ? await getWasmFrom(config.wasm) : await getWasm();
    const handle = ex.tq_engine_create(config.dim, config.seed);
    if (handle < 0) {
      throw new Error(
        `TurboQuant: failed to create engine (dim=${config.dim}, seed=${config.seed})`,
      );
    }
    return new TurboQuant(ex, handle, config.dim);
  }

  /**
   * Compress a float32 vector (~3 bits/dim).
   * @param vector - Float32Array of length `dim`
   * @returns Compressed bytes
   */
  encode(vector: Float32Array): Uint8Array {
    if (vector.length !== this.dim) {
      throw new Error(`TurboQuant: expected ${this.dim} dims, got ${vector.length}`);
    }
    const ex = this.#ex;
    const inputPtr = wasmWriteF32(ex, vector);
    const outLenPtr = wasmAllocU32(ex);

    const resultPtr = ex.tq_encode(this.#handle, inputPtr, this.dim, outLenPtr);
    const outLen = wasmReadU32(ex, outLenPtr);

    ex.tq_free(inputPtr, vector.byteLength);
    ex.tq_free(outLenPtr, 4);

    if (!resultPtr) throw new Error("TurboQuant: encode failed");

    const result = new Uint8Array(ex.memory.buffer, resultPtr, outLen).slice();
    ex.tq_free(resultPtr, outLen);
    return result;
  }

  /**
   * Decompress back to a float32 vector.
   * @param compressed - Bytes from `encode()`
   * @returns Reconstructed Float32Array
   */
  decode(compressed: Uint8Array): Float32Array {
    const ex = this.#ex;
    const compPtr = wasmWriteU8(ex, compressed);
    const outLenPtr = wasmAllocU32(ex);

    const resultPtr = ex.tq_decode(this.#handle, compPtr, compressed.length, outLenPtr);
    const outLen = wasmReadU32(ex, outLenPtr);

    ex.tq_free(compPtr, compressed.length);
    ex.tq_free(outLenPtr, 4);

    if (!resultPtr) throw new Error("TurboQuant: decode failed");

    const result = new Float32Array(ex.memory.buffer, resultPtr, outLen).slice();
    ex.tq_free_f32(resultPtr, outLen);
    return result;
  }

  /**
   * Estimate dot product between a query and compressed vector.
   * Faster than decode + manual dot — operates directly on compressed bytes.
   *
   * @param query - Float32Array of length `dim`
   * @param compressed - Bytes from `encode()`
   * @returns Estimated inner product
   */
  dot(query: Float32Array, compressed: Uint8Array): number {
    if (query.length !== this.dim) {
      throw new Error(`TurboQuant: expected ${this.dim} dims, got ${query.length}`);
    }
    const ex = this.#ex;
    const qPtr = wasmWriteF32(ex, query);
    const compPtr = wasmWriteU8(ex, compressed);

    const score = ex.tq_dot(
      this.#handle,
      qPtr,
      this.dim,
      compPtr,
      compressed.length,
    );

    ex.tq_free(qPtr, query.byteLength);
    ex.tq_free(compPtr, compressed.length);

    return score;
  }

  /**
   * Batch dot product: compute dot(query, vectors[i]) for all vectors.
   * Much faster than calling dot() in a loop — one WASM call, query rotated once.
   *
   * @param query - Float32Array of length `dim`
   * @param compressedConcat - All compressed vectors concatenated into one Uint8Array
   * @param bytesPerVector - Size of each compressed vector (from encode().length)
   * @returns Float32Array of scores, one per vector.
   *   The returned array is reused across calls — copy with .slice() if you need to retain it.
   */
  async dotBatch(query: Float32Array, compressedConcat: Uint8Array, bytesPerVector: number): Promise<Float32Array> {
    // Auto-detect WebGPU on first call or when data changes
    if (!this.#gpuChecked || compressedConcat !== this.#gpuDataRef) {
      this.#gpuChecked = true;
      if (this.#gpuIndex) { this.#gpuIndex.destroy(); this.#gpuIndex = null; }
      if (typeof navigator !== "undefined" && navigator.gpu) {
        return this.#initGpuAndSearch(query, compressedConcat, bytesPerVector);
      }
    }
    if (this.#gpuIndex) {
      return this.#gpuIndex.dotBatchGpu(query);
    }
    return this.#cpuDotBatch(query, compressedConcat, bytesPerVector);
  }

  #cpuDotBatch(query: Float32Array, compressedConcat: Uint8Array, bytesPerVector: number): Float32Array {
    if (query.length !== this.dim) {
      throw new Error(`TurboQuant: expected ${this.dim} dims, got ${query.length}`);
    }
    const numVectors = Math.floor(compressedConcat.length / bytesPerVector);
    const ex = this.#ex;

    if (compressedConcat !== this.#cachedRef || bytesPerVector !== this.#cachedBpv) {
      if (this.#cachedPtr) ex.tq_free(this.#cachedPtr, this.#cachedLen);
      this.#cachedPtr = wasmWriteU8(ex, compressedConcat);
      this.#cachedLen = compressedConcat.length;
      this.#cachedRef = compressedConcat;
      this.#cachedBpv = bytesPerVector;
    }

    if (this.#scoresCount !== numVectors) {
      if (this.#scoresPtr) ex.tq_free_f32(this.#scoresPtr, this.#scoresCount);
      this.#scoresPtr = ex.tq_alloc_f32(numVectors);
      if (!this.#scoresPtr) throw new Error("TurboQuant: WASM alloc failed for scores");
      this.#scoresCount = numVectors;
      this.#scoresOut = new Float32Array(numVectors);
    }

    if (this.#queryLen !== query.byteLength) {
      if (this.#queryPtr) ex.tq_free(this.#queryPtr, this.#queryLen);
      this.#queryPtr = ex.tq_alloc(query.byteLength);
      if (!this.#queryPtr) throw new Error("TurboQuant: WASM alloc failed for query");
      this.#queryLen = query.byteLength;
    }
    new Float32Array(ex.memory.buffer, this.#queryPtr, query.length).set(query);

    ex.tq_dot_batch(
      this.#handle, this.#queryPtr, this.dim,
      this.#cachedPtr, bytesPerVector, numVectors, this.#scoresPtr,
    );

    this.#scoresOut!.set(new Float32Array(ex.memory.buffer, this.#scoresPtr, numVectors));
    return this.#scoresOut!;
  }

  async #initGpuAndSearch(query: Float32Array, compressedConcat: Uint8Array, bytesPerVector: number): Promise<Float32Array> {
    try {
      const { TQGpuIndex } = await import("./gpu-index.js");
      this.#gpuIndex = await TQGpuIndex.create(this, compressedConcat as Uint8Array, bytesPerVector);
      this.#gpuDataRef = compressedConcat;
    } catch (e) {
      console.warn("TurboQuant: WebGPU init failed, using CPU SIMD", e);
    }
    if (this.#gpuIndex) {
      return this.#gpuIndex.dotBatchGpu(query);
    }
    return this.#cpuDotBatch(query, compressedConcat, bytesPerVector);
  }

  /**
   * Rotate a query vector into TQ's internal rotation space.
   * Used by WebGPU path: the rotated query is uploaded as a GPU uniform,
   * then the compute shader computes dot products directly on compressed data.
   */
  rotateQuery(query: Float32Array): Float32Array {
    if (query.length !== this.dim) {
      throw new Error(`TurboQuant: expected ${this.dim} dims, got ${query.length}`);
    }
    const ex = this.#ex;

    // Reuse cached query buffer
    if (this.#queryLen !== query.byteLength) {
      if (this.#queryPtr) ex.tq_free(this.#queryPtr, this.#queryLen);
      this.#queryPtr = ex.tq_alloc(query.byteLength);
      if (!this.#queryPtr) throw new Error("TurboQuant: WASM alloc failed for query");
      this.#queryLen = query.byteLength;
    }
    new Float32Array(ex.memory.buffer, this.#queryPtr, query.length).set(query);

    if (!this.#rotOutPtr) {
      this.#rotOutPtr = ex.tq_alloc_f32(this.dim);
      if (!this.#rotOutPtr) throw new Error("TurboQuant: WASM alloc failed for rotated query");
    }

    ex.tq_rotate_query(this.#handle, this.#queryPtr, this.dim, this.#rotOutPtr);

    const rotated = new Float32Array(this.dim);
    rotated.set(new Float32Array(ex.memory.buffer, this.#rotOutPtr, this.dim));
    return rotated;
  }

  /**
   * Create a streaming compressed vector buffer.
   * Vectors are TQ-compressed on append, decompressed buffer maintained for readback.
   *
   * @param maxPositions - Initial capacity (grows automatically)
   */
  createStream(maxPositions: number): TQStream {
    const handle = this.#ex.tq_stream_create(this.#handle, maxPositions);
    if (handle < 0) throw new Error("TurboQuant: failed to create TQStream");
    const bpv = this.#ex.tq_stream_bytes_per_vector(handle);
    return new TQStream(this.#ex, handle, this.dim, bpv);
  }

  /** Release engine resources. Call when done. Safe to call multiple times. */
  destroy(): void {
    if (this.#handle < 0) return;
    if (this.#gpuIndex) {
      this.#gpuIndex.destroy();
      this.#gpuIndex = null;
    }
    if (this.#cachedPtr) {
      this.#ex.tq_free(this.#cachedPtr, this.#cachedLen);
      this.#cachedPtr = 0;
      this.#cachedRef = null;
    }
    if (this.#scoresPtr) {
      this.#ex.tq_free_f32(this.#scoresPtr, this.#scoresCount);
      this.#scoresPtr = 0;
      this.#scoresOut = null;
    }
    if (this.#queryPtr) {
      this.#ex.tq_free(this.#queryPtr, this.#queryLen);
      this.#queryPtr = 0;
    }
    if (this.#rotOutPtr) {
      this.#ex.tq_free_f32(this.#rotOutPtr, this.dim);
      this.#rotOutPtr = 0;
    }
    this.#ex.tq_engine_destroy(this.#handle);
    this.#handle = -1;
  }
}

/**
 * Streaming compressed vector buffer. Compress-only storage.
 * Use dotBatch on getCompressed() for scoring — never decompress for search.
 * Use decodePosition() only when you need individual float values.
 */
export class TQStream {
  readonly dim: number;
  readonly bytesPerVector: number;
  #handle: number;
  #ex: TurboQuantExports;
  #inputPtr: number = 0;
  #inputLen: number = 0;

  /** @internal — use TurboQuant.createStream() */
  constructor(ex: TurboQuantExports, handle: number, dim: number, bpv: number) {
    this.#ex = ex;
    this.#handle = handle;
    this.dim = dim;
    this.bytesPerVector = bpv;
  }

  /** Append a single vector. Compresses and stores. No decompression. */
  append(vector: Float32Array): void {
    if (vector.length !== this.dim) {
      throw new Error(`TQStream: expected ${this.dim} dims, got ${vector.length}`);
    }
    const ex = this.#ex;
    if (this.#inputLen !== vector.byteLength) {
      if (this.#inputPtr) ex.tq_free(this.#inputPtr, this.#inputLen);
      this.#inputPtr = ex.tq_alloc(vector.byteLength);
      if (!this.#inputPtr) throw new Error("TQStream: WASM alloc failed");
      this.#inputLen = vector.byteLength;
    }
    new Float32Array(ex.memory.buffer, this.#inputPtr, vector.length).set(vector);
    const rc = ex.tq_stream_append(this.#handle, this.#inputPtr, this.dim);
    if (rc < 0) throw new Error("TQStream: append failed");
  }

  /** Append multiple vectors at once. */
  appendBatch(vectors: Float32Array, count?: number): void {
    const n = count ?? Math.floor(vectors.length / this.dim);
    const ex = this.#ex;
    const byteLen = n * this.dim * 4;
    if (this.#inputLen !== byteLen) {
      if (this.#inputPtr) ex.tq_free(this.#inputPtr, this.#inputLen);
      this.#inputPtr = ex.tq_alloc(byteLen);
      if (!this.#inputPtr) throw new Error("TQStream: WASM alloc failed");
      this.#inputLen = byteLen;
    }
    new Float32Array(ex.memory.buffer, this.#inputPtr, n * this.dim).set(
      vectors.subarray(0, n * this.dim),
    );
    const rc = ex.tq_stream_append_batch(this.#handle, this.#inputPtr, this.dim, n);
    if (rc < 0) throw new Error("TQStream: appendBatch failed");
  }

  /** Get full compressed store as a copy. Use with dotBatch for scoring. */
  getCompressed(): Uint8Array {
    const ex = this.#ex;
    const outLenPtr = wasmAllocU32(ex);
    const ptr = ex.tq_stream_get_compressed(this.#handle, outLenPtr);
    const len = wasmReadU32(ex, outLenPtr);
    ex.tq_free(outLenPtr, 4);
    if (!ptr) return new Uint8Array(0);
    return new Uint8Array(ex.memory.buffer, ptr, len).slice();
  }

  /** Decode a single position. Only use when you need individual float values. */
  decodePosition(position: number): Float32Array {
    const ex = this.#ex;
    const outPtr = ex.tq_alloc_f32(this.dim);
    if (!outPtr) throw new Error("TQStream: WASM alloc failed");
    const rc = ex.tq_stream_decode_position(this.#handle, position, outPtr, this.dim);
    if (rc < 0) {
      ex.tq_free_f32(outPtr, this.dim);
      throw new Error(`TQStream: decodePosition(${position}) failed`);
    }
    const result = new Float32Array(ex.memory.buffer, outPtr, this.dim).slice();
    ex.tq_free_f32(outPtr, this.dim);
    return result;
  }

  /** Number of vectors currently stored. */
  get length(): number {
    return this.#ex.tq_stream_length(this.#handle);
  }

  /** Truncate stream to given position. */
  rewind(position: number): void {
    this.#ex.tq_stream_rewind(this.#handle, position);
  }

  /** Release resources. */
  destroy(): void {
    if (this.#handle < 0) return;
    if (this.#inputPtr) {
      this.#ex.tq_free(this.#inputPtr, this.#inputLen);
      this.#inputPtr = 0;
    }
    this.#ex.tq_stream_destroy(this.#handle);
    this.#handle = -1;
  }
}

export default TurboQuant;
