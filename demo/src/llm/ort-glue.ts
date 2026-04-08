/**
 * ORT + TurboQuant unified WASM glue.
 * Provides WASI, JSEP, and C++ runtime imports for the Zig-compiled ORT binary.
 * 30 imports total — all with correct signatures.
 */

const WASM_URL = "/dist/turboquant-llm.wasm";

interface OrtExports {
  memory: WebAssembly.Memory;
  OrtInit(num_threads: number, logging_level: number): number;
  OrtCreateSession(
    buffer_ptr: number, buffer_size: number,
    session_options_ptr: number, session_ptr_ptr: number,
  ): number;
  OrtRun(
    session_ptr: number, input_names_ptr: number, input_tensors_ptr: number,
    input_count: number, output_names_ptr: number, output_count: number,
    output_tensors_ptr: number,
  ): number;
  OrtCreateTensor(
    data_type: number, data_ptr: number, data_size: number,
    dims_ptr: number, dims_count: number,
  ): number;
  OrtGetTensorData(
    tensor_ptr: number, data_ptr_ptr: number,
    dims_ptr_ptr: number, dims_count_ptr: number,
  ): number;
  OrtReleaseTensor(tensor_ptr: number): void;
  OrtReleaseSession(session_ptr: number): void;
  OrtCreateSessionOptions(
    graph_optimization_level: number,
    enable_cpu_mem_arena: number,
    enable_mem_pattern: number,
    execution_mode: number,
    enable_profiling: number,
    log_id: number,
    log_severity_level: number,
    log_verbosity_level: number,
    optimized_model_filepath: number,
    options_ptr_ptr: number,
  ): number;
  OrtReleaseSessionOptions(options_ptr: number): void;
  wasm_malloc(size: number): number;
  wasm_free(ptr: number): void;
  tq_kv_create(head_dim: number, max_positions: number): number;
  tq_kv_destroy(stream_id: number): void;
  tq_kv_append(stream_id: number, data_ptr: number, dim: number): number;
  tq_kv_dot_batch(
    stream_id: number, query_ptr: number, dim: number,
    out_scores: number, max_scores: number,
  ): number;
  tq_kv_decode_position(
    stream_id: number, position: number, out_ptr: number, dim: number,
  ): number;
  tq_kv_length(stream_id: number): number;
  tq_kv_compressed_size(stream_id: number): number;
}

let wasm: OrtExports | null = null;

function noop() {}
function zero() { return 0; }

export async function initOrt(): Promise<OrtExports> {
  const imports: WebAssembly.Imports = {
    env: {
      // JSEP — CPU execution, all ops run in WASM
      jsep_alloc(size: number): number { return wasm!.wasm_malloc(size); },
      jsep_free(ptr: number): number { wasm!.wasm_free(ptr); return 0; },
      jsep_create_kernel: noop,
      jsep_release_kernel: noop,
      jsep_run: zero,
      jsep_capture_begin: noop,
      jsep_capture_end: noop,
      jsep_replay: noop,

      // C++ runtime — single-threaded WASM
      __cxa_thread_atexit: noop,

      // Abseil threading — single-threaded, no contention
      AbslInternalPerThreadSemPost_lts_20250814: noop,
      AbslInternalPerThreadSemWait_lts_20250814(_timeout: bigint): number { return 0; },
      // CreateThreadIdentity — returns a pointer; 0 = no identity (single-threaded)
      _ZN4absl12lts_2025081424synchronization_internal20CreateThreadIdentityEv: zero,

    },

    wasi_snapshot_preview1: {
      clock_time_get(_id: number, _prec: bigint, time_ptr: number): number {
        new DataView(wasm!.memory.buffer).setBigUint64(
          time_ptr, BigInt(Math.floor(performance.now() * 1e6)), true,
        );
        return 0;
      },
      environ_get: zero,
      environ_sizes_get(count_ptr: number, size_ptr: number): number {
        const v = new DataView(wasm!.memory.buffer);
        v.setUint32(count_ptr, 0, true);
        v.setUint32(size_ptr, 0, true);
        return 0;
      },
      fd_close: zero,
      fd_fdstat_get: zero,
      fd_fdstat_set_flags: zero,
      fd_prestat_get(): number { return 8; },
      fd_prestat_dir_name(): number { return 8; },
      fd_read: zero,
      fd_seek: zero,
      fd_write(_fd: number, iovs_ptr: number, iovs_len: number, nwritten_ptr: number): number {
        const v = new DataView(wasm!.memory.buffer);
        const mem = new Uint8Array(wasm!.memory.buffer);
        let written = 0;
        for (let i = 0; i < iovs_len; i++) {
          const ptr = v.getUint32(iovs_ptr + i * 8, true);
          const len = v.getUint32(iovs_ptr + i * 8 + 4, true);
          console.log(new TextDecoder().decode(mem.subarray(ptr, ptr + len)));
          written += len;
        }
        v.setUint32(nwritten_ptr, written, true);
        return 0;
      },
      path_open: zero,
      path_filestat_get(): number { return 8; },
      path_readlink(): number { return 8; },
      poll_oneoff: zero,
      proc_exit(code: number): void { throw new Error(`WASM exit: ${code}`); },
      sched_yield: zero,
    },
  };

  const response = await fetch(WASM_URL);
  const { instance } = await WebAssembly.instantiateStreaming(response, imports);
  wasm = instance.exports as unknown as OrtExports;

  const rc = wasm.OrtInit(1, 3);
  if (rc !== 0) throw new Error(`OrtInit failed: ${rc}`);

  return wasm;
}

export function getWasm(): OrtExports {
  if (!wasm) throw new Error("ORT not initialized");
  return wasm;
}

export type { OrtExports };
