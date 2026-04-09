/**
 * LiteRT + TurboQuant WASM glue.
 * Matches @litertjs/core architecture: LiteRT C API + WebGPU.
 * 6 env imports + 18 WASI imports.
 */

const WASM_URL = "/turboquant-litert.wasm";

// ============================================================================
// LiteRT C API exports
// ============================================================================
export interface LiteRtExports {
  memory: WebAssembly.Memory;
  wasm_malloc(size: number): number;
  wasm_free(ptr: number): void;

  // Environment
  LiteRtCreateEnvironment(numOpts: number, opts: number, env: number): number;
  LiteRtDestroyEnvironment(env: number): void;

  // Model
  LiteRtCreateModelFromBuffer(buf: number, size: number, model: number): number;
  LiteRtDestroyModel(model: number): void;

  // Compiled model
  LiteRtCreateCompiledModel(env: number, model: number, opts: number, compiled: number): number;
  LiteRtDestroyCompiledModel(compiled: number): void;
  LiteRtRunCompiledModel(compiled: number, numInputs: number, inputs: number, numOutputs: number, outputs: number): number;

  // Tensor buffer
  LiteRtCreateTensorBufferFromHostMemory(type: number, dims: number, ndims: number, data: number, size: number, buf: number): number;
  LiteRtDestroyTensorBuffer(buf: number): void;
  LiteRtLockTensorBuffer(buf: number, mode: number, ptr: number): number;
  LiteRtUnlockTensorBuffer(buf: number): number;
  LiteRtGetTensorBufferSize(buf: number, size: number): number;

  // TQ KV cache
  tq_kv_create(head_dim: number, max_positions: number): number;
  tq_kv_destroy(stream_id: number): void;
  tq_kv_append(stream_id: number, data_ptr: number, dim: number): number;
  tq_kv_dot_batch(stream_id: number, query_ptr: number, dim: number, out: number, max: number): number;
  tq_kv_decode_position(stream_id: number, pos: number, out: number, dim: number): number;
  tq_kv_length(stream_id: number): number;
  tq_kv_compressed_size(stream_id: number): number;
}

// ============================================================================
// Memory helpers
// ============================================================================
export function writeBytes(e: LiteRtExports, data: Uint8Array): number {
  const ptr = e.wasm_malloc(data.byteLength);
  new Uint8Array(e.memory.buffer, ptr, data.byteLength).set(data);
  return ptr;
}

export function readString(e: LiteRtExports, ptr: number): string {
  const mem = new Uint8Array(e.memory.buffer);
  let end = ptr;
  while (mem[end] !== 0) end++;
  return new TextDecoder().decode(mem.subarray(ptr, end));
}

export function writeU32(e: LiteRtExports, value: number): number {
  const ptr = e.wasm_malloc(4);
  new DataView(e.memory.buffer).setUint32(ptr, value, true);
  return ptr;
}

export function readU32(e: LiteRtExports, ptr: number): number {
  return new DataView(e.memory.buffer).getUint32(ptr, true);
}

// ============================================================================
// WASM instantiation
// ============================================================================
let wasm: LiteRtExports | null = null;

export async function initLiteRt(): Promise<LiteRtExports> {
  // 6 env imports — all safe no-ops for WASM
  const envProxy = new Proxy({} as Record<string, Function>, {
    get(_target, prop: string) {
      return (..._args: unknown[]) => 0;
    },
  });

  const wasiImports: Record<string, Function> = {
    clock_time_get(_id: number, _prec: bigint, time_ptr: number): number {
      new DataView(wasm!.memory.buffer).setBigUint64(
        time_ptr, BigInt(Math.floor(performance.now() * 1e6)), true);
      return 0;
    },
    environ_get() { return 0; },
    environ_sizes_get(cp: number, sp: number): number {
      const v = new DataView(wasm!.memory.buffer);
      v.setUint32(cp, 0, true);
      v.setUint32(sp, 0, true);
      return 0;
    },
    fd_close() { return 0; },
    fd_fdstat_get() { return 0; },
    fd_fdstat_set_flags() { return 0; },
    fd_filestat_get() { return 8; },
    fd_filestat_set_size() { return 0; },
    fd_pread() { return 8; },
    fd_prestat_get() { return 8; },
    fd_prestat_dir_name() { return 8; },
    fd_read() { return 0; },
    fd_seek() { return 0; },
    fd_write(_fd: number, iovsPtr: number, iovsLen: number, nwrittenPtr: number): number {
      const v = new DataView(wasm!.memory.buffer);
      const mem = new Uint8Array(wasm!.memory.buffer);
      let written = 0;
      const parts: string[] = [];
      for (let i = 0; i < iovsLen; i++) {
        const ptr = v.getUint32(iovsPtr + i * 8, true);
        const len = v.getUint32(iovsPtr + i * 8 + 4, true);
        parts.push(new TextDecoder().decode(mem.subarray(ptr, ptr + len)));
        written += len;
      }
      console.log(parts.join(""));
      v.setUint32(nwrittenPtr, written, true);
      return 0;
    },
    path_open() { return 44; },
    path_filestat_get() { return 8; },
    poll_oneoff() { return 0; },
    proc_exit(code: number): void { throw new Error(`WASM exit: ${code}`); },
    sched_yield() { return 0; },
  };

  const response = await fetch(WASM_URL);
  const module = await WebAssembly.compileStreaming(response);
  const instance = await WebAssembly.instantiate(module, {
    env: envProxy,
    wasi_snapshot_preview1: wasiImports,
  });
  wasm = instance.exports as unknown as LiteRtExports;
  return wasm;
}

export function getWasm(): LiteRtExports {
  if (!wasm) throw new Error("LiteRT not initialized");
  return wasm;
}

// ============================================================================
// High-level API matching @litertjs/core pattern
// ============================================================================

/**
 * Load a .tflite model from a Uint8Array into the WASM runtime.
 * Returns the model handle (LiteRtModel pointer).
 * JS fetches the model, copies into WASM heap, calls LiteRtCreateModelFromBuffer.
 */
export function loadModel(e: LiteRtExports, modelData: Uint8Array): number {
  const ptr = writeBytes(e, modelData);
  const modelOut = e.wasm_malloc(4); // pointer to LiteRtModel
  const status = e.LiteRtCreateModelFromBuffer(ptr, modelData.byteLength, modelOut);
  const model = readU32(e, modelOut);
  e.wasm_free(modelOut);
  e.wasm_free(ptr); // LiteRT copies internally
  if (status !== 0 || !model) {
    throw new Error(`LiteRtCreateModelFromBuffer failed: status=${status}`);
  }
  return model;
}

/**
 * Create a LiteRT environment (with optional WebGPU device).
 */
export function createEnvironment(e: LiteRtExports): number {
  const envOut = e.wasm_malloc(4);
  const status = e.LiteRtCreateEnvironment(0, 0, envOut);
  const env = readU32(e, envOut);
  e.wasm_free(envOut);
  if (status !== 0 || !env) {
    throw new Error(`LiteRtCreateEnvironment failed: status=${status}`);
  }
  return env;
}

/**
 * Compile a model for execution (applies XNNPack/WebGPU optimization).
 */
export function compileModel(e: LiteRtExports, env: number, model: number): number {
  const compiledOut = e.wasm_malloc(4);
  const status = e.LiteRtCreateCompiledModel(env, model, 0, compiledOut);
  const compiled = readU32(e, compiledOut);
  e.wasm_free(compiledOut);
  if (status !== 0 || !compiled) {
    throw new Error(`LiteRtCreateCompiledModel failed: status=${status}`);
  }
  return compiled;
}
