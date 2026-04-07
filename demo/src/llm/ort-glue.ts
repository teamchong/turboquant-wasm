/**
 * ORT + TurboQuant unified WASM glue.
 * Provides JSEP extern fn implementations and model loading.
 */

const WASM_URL = "/dist/turboquant-llm.wasm";

interface OrtExports {
  memory: WebAssembly.Memory;
  _start(): void;

  // ORT WASM API (from onnxruntime/wasm/api.cc)
  OrtInit(num_threads: number, logging_level: number): number;
  OrtCreateSessionFromBuffer(
    buffer_ptr: number,
    buffer_size: number,
    session_options_ptr: number,
    session_ptr_ptr: number,
  ): number;
  OrtRun(
    session_ptr: number,
    input_names_ptr: number,
    input_tensors_ptr: number,
    input_count: number,
    output_names_ptr: number,
    output_count: number,
    output_tensors_ptr: number,
  ): number;
  OrtCreateTensor(
    data_type: number,
    data_ptr: number,
    data_size: number,
    dims_ptr: number,
    dims_count: number,
  ): number;
  OrtGetTensorData(
    tensor_ptr: number,
    data_ptr_ptr: number,
    dims_ptr_ptr: number,
    dims_count_ptr: number,
  ): number;
  OrtReleaseTensor(tensor_ptr: number): void;
  OrtReleaseSession(session_ptr: number): void;

  // Memory management
  malloc(size: number): number;
  free(ptr: number): void;

  // TQ bridge (from tq_bridge.zig)
  tq_kv_create(head_dim: number, max_positions: number): number;
  tq_kv_destroy(stream_id: number): void;
  tq_kv_append(stream_id: number, data_ptr: number, dim: number): number;
  tq_kv_dot_batch(
    stream_id: number,
    query_ptr: number,
    dim: number,
    out_scores: number,
    max_scores: number,
  ): number;
  tq_kv_decode_position(
    stream_id: number,
    position: number,
    out_ptr: number,
    dim: number,
  ): number;
  tq_kv_length(stream_id: number): number;
  tq_kv_compressed_size(stream_id: number): number;
}

let wasm: OrtExports | null = null;

// JSEP (JavaScript Execution Provider) implementations.
// These are called by ORT's C++ code via extern fn declarations.
// CPU execution — all ops run in WASM. WebGPU matmul dispatch
// will replace jsep_run when the WebGPU path is wired.
const jsepImports = {
  env: {
    jsep_alloc(size: number): number {
      return wasm!.malloc(size);
    },
    jsep_free(ptr: number): number {
      wasm!.free(ptr);
      return 0;
    },
    jsep_download(src: number, dst: number, bytes: number): void {
      const mem = new Uint8Array(wasm!.memory.buffer);
      mem.copyWithin(dst, src, src + bytes);
    },
    jsep_copy(src: number, dst: number, bytes: number, _gpu_to_cpu: number): void {
      const mem = new Uint8Array(wasm!.memory.buffer);
      mem.copyWithin(dst, src, src + bytes);
    },
    jsep_create_kernel(_optype: number, _ptr: number, _attr: number): void {},
    jsep_release_kernel(_ptr: number): void {},
    jsep_run(
      _kernel: number,
      _num_inputs: number,
      _inputs: number,
      _num_outputs: number,
      _outputs: number,
      _attrs: number,
    ): number {
      return 0; // CPU provider handles all ops
    },
    jsep_capture_begin(): void {},
    jsep_capture_end(): void {},
    jsep_replay(): void {},
    emscripten_get_now(): number {
      return performance.now();
    },
  },
  wasi_snapshot_preview1: {
    args_get(): number { return 0; },
    args_sizes_get(argc_ptr: number, argv_buf_size_ptr: number): number {
      const view = new DataView(wasm!.memory.buffer);
      view.setUint32(argc_ptr, 0, true);
      view.setUint32(argv_buf_size_ptr, 0, true);
      return 0;
    },
    environ_get(): number { return 0; },
    environ_sizes_get(count_ptr: number, size_ptr: number): number {
      const view = new DataView(wasm!.memory.buffer);
      view.setUint32(count_ptr, 0, true);
      view.setUint32(size_ptr, 0, true);
      return 0;
    },
    clock_time_get(_id: number, _precision: bigint, time_ptr: number): number {
      const view = new DataView(wasm!.memory.buffer);
      view.setBigUint64(time_ptr, BigInt(Math.floor(performance.now() * 1e6)), true);
      return 0;
    },
    fd_close(): number { return 0; },
    fd_fdstat_get(): number { return 0; },
    fd_prestat_get(): number { return 8; }, // EBADF — no preopened dirs
    fd_prestat_dir_name(): number { return 8; },
    fd_read(): number { return 0; },
    fd_seek(): number { return 0; },
    fd_write(_fd: number, iovs_ptr: number, iovs_len: number, nwritten_ptr: number): number {
      const view = new DataView(wasm!.memory.buffer);
      const mem = new Uint8Array(wasm!.memory.buffer);
      let written = 0;
      for (let i = 0; i < iovs_len; i++) {
        const ptr = view.getUint32(iovs_ptr + i * 8, true);
        const len = view.getUint32(iovs_ptr + i * 8 + 4, true);
        const text = new TextDecoder().decode(mem.subarray(ptr, ptr + len));
        console.log(text);
        written += len;
      }
      view.setUint32(nwritten_ptr, written, true);
      return 0;
    },
    proc_exit(code: number): void {
      throw new Error(`WASM exit: ${code}`);
    },
  },
};

export async function initOrt(): Promise<OrtExports> {
  const response = await fetch(WASM_URL);
  const { instance } = await WebAssembly.instantiateStreaming(response, jsepImports);
  wasm = instance.exports as unknown as OrtExports;

  // Initialize WASI runtime
  try { wasm._start(); } catch (_) { /* proc_exit throws by design */ }

  // Initialize ORT runtime
  const rc = wasm.OrtInit(1, 3); // 1 thread, warning level logging
  if (rc !== 0) throw new Error(`OrtInit failed with code ${rc}`);

  return wasm;
}

export function getWasm(): OrtExports {
  if (!wasm) throw new Error("ORT not initialized — call initOrt() first");
  return wasm;
}

export type { OrtExports };
