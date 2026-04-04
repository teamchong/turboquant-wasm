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
  tq_alloc(len: number): number;
  tq_free(ptr: number, len: number): void;
  tq_alloc_f32(count: number): number;
  tq_free_f32(ptr: number, count: number): void;
}

// --- Singleton WASM instance ---

let wasmInstance: TurboQuantExports | null = null;
let wasmReady: Promise<TurboQuantExports> | null = null;

async function getWasm(): Promise<TurboQuantExports> {
  if (wasmInstance) return wasmInstance;
  if (wasmReady) return wasmReady;

  wasmReady = (async () => {
    const bytes = decodeBase64(wasmBase64);
    const module = await WebAssembly.compile(bytes);
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

  /** Release engine resources. Call when done. */
  destroy(): void {
    this.#ex.tq_engine_destroy(this.#handle);
  }
}

export default TurboQuant;
