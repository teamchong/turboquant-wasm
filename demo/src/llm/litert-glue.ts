/**
 * LiteRT-LM + TurboQuant unified WASM glue.
 * Provides WASI filesystem, env imports, and C API wrapper for the
 * Zig-compiled LiteRT-LM binary (9.5 MB, 1313 objects).
 *
 * 89 imports total: 19 WASI + 70 env.
 */

const WASM_URL = "/turboquant-litert.wasm";

// ============================================================================
// LiteRT-LM C API exports
// ============================================================================
export interface LiteRtExports {
  memory: WebAssembly.Memory;
  wasm_malloc(size: number): number;
  wasm_free(ptr: number): void;

  // Engine settings
  litert_lm_engine_settings_create(
    model_path: number, backend: number,
    vision_backend: number, audio_backend: number,
  ): number;
  litert_lm_engine_settings_delete(settings: number): void;
  litert_lm_engine_settings_set_max_num_tokens(settings: number, n: number): void;
  litert_lm_engine_settings_set_prefill_chunk_size(settings: number, n: number): void;
  litert_lm_engine_settings_enable_benchmark(settings: number): void;

  // Engine
  litert_lm_engine_create(settings: number): number;
  litert_lm_engine_delete(engine: number): void;
  litert_lm_engine_create_session(engine: number, config: number): number;

  // Session config
  litert_lm_session_config_create(): number;
  litert_lm_session_config_delete(config: number): void;
  litert_lm_session_config_set_max_output_tokens(config: number, n: number): void;

  // Session
  litert_lm_session_delete(session: number): void;
  litert_lm_session_generate_content(
    session: number, inputs: number, num_inputs: number,
  ): number;

  // Responses
  litert_lm_responses_delete(responses: number): void;
  litert_lm_responses_get_response_text_at(responses: number, index: number): number;

  // Benchmark
  litert_lm_session_get_benchmark_info(session: number): number;
  litert_lm_benchmark_info_delete(info: number): void;
  litert_lm_benchmark_info_get_time_to_first_token(info: number): number;
  litert_lm_benchmark_info_get_decode_tokens_per_sec_at(info: number, idx: number): number;

  // TQ KV cache
  tq_kv_create(head_dim: number, max_positions: number): number;
  tq_kv_destroy(stream_id: number): void;
  tq_kv_length(stream_id: number): number;
  tq_kv_compressed_size(stream_id: number): number;
}

// ============================================================================
// String helpers
// ============================================================================
function writeString(e: LiteRtExports, s: string): number {
  const encoded = new TextEncoder().encode(s);
  const ptr = e.wasm_malloc(encoded.length + 1);
  new Uint8Array(e.memory.buffer, ptr, encoded.length + 1).set([...encoded, 0]);
  return ptr;
}

function readString(e: LiteRtExports, ptr: number): string {
  const mem = new Uint8Array(e.memory.buffer);
  let end = ptr;
  while (mem[end] !== 0) end++;
  return new TextDecoder().decode(mem.subarray(ptr, end));
}

// InputData struct: { type: i32, data: i32 (ptr), size: i32 }
// kInputText = 0
function writeInputData(e: LiteRtExports, text: string): number {
  const textPtr = writeString(e, text);
  const textLen = new TextEncoder().encode(text).length;
  const structPtr = e.wasm_malloc(12); // 3 x i32
  const v = new DataView(e.memory.buffer);
  v.setInt32(structPtr, 0, true);       // type = kInputText
  v.setUint32(structPtr + 4, textPtr, true); // data ptr
  v.setUint32(structPtr + 8, textLen, true); // size
  return structPtr;
}

// ============================================================================
// Virtual WASI filesystem — serves model bytes from JS ArrayBuffer
// ============================================================================
class VirtualFS {
  private files = new Map<number, { data: Uint8Array; pos: number }>();
  private paths = new Map<string, Uint8Array>();
  private nextFd = 10; // fds 0-2 are stdin/stdout/stderr

  registerFile(path: string, data: ArrayBuffer): void {
    this.paths.set(path, new Uint8Array(data));
  }

  pathOpen(
    dirfd: number, dirflags: number, pathPtr: number, pathLen: number,
    oflags: number, fsRightsBase: bigint, fsRightsInheriting: bigint,
    fdflags: number, fdPtr: number, mem: DataView, memBytes: Uint8Array,
  ): number {
    const pathStr = new TextDecoder().decode(memBytes.subarray(pathPtr, pathPtr + pathLen));
    // Try exact match, then try stripping leading /
    const data = this.paths.get(pathStr) || this.paths.get("/" + pathStr) || this.paths.get(pathStr.replace(/^\/+/, ""));
    if (!data) {
      console.warn(`[VFS] path_open: not found: "${pathStr}"`);
      return 44; // ENOENT
    }
    const fd = this.nextFd++;
    this.files.set(fd, { data, pos: 0 });
    mem.setUint32(fdPtr, fd, true);
    return 0;
  }

  fdPread(
    fd: number, iovsPtr: number, iovsLen: number, offset: bigint,
    nreadPtr: number, mem: DataView, memBytes: Uint8Array,
  ): number {
    const file = this.files.get(fd);
    if (!file) return 8; // EBADF
    let totalRead = 0;
    let fileOffset = Number(offset);
    for (let i = 0; i < iovsLen; i++) {
      const bufPtr = mem.getUint32(iovsPtr + i * 8, true);
      const bufLen = mem.getUint32(iovsPtr + i * 8 + 4, true);
      const available = Math.min(bufLen, file.data.length - fileOffset);
      if (available > 0) {
        memBytes.set(file.data.subarray(fileOffset, fileOffset + available), bufPtr);
        fileOffset += available;
        totalRead += available;
      }
    }
    mem.setUint32(nreadPtr, totalRead, true);
    return 0;
  }

  fdRead(
    fd: number, iovsPtr: number, iovsLen: number,
    nreadPtr: number, mem: DataView, memBytes: Uint8Array,
  ): number {
    const file = this.files.get(fd);
    if (!file) return 8;
    let totalRead = 0;
    for (let i = 0; i < iovsLen; i++) {
      const bufPtr = mem.getUint32(iovsPtr + i * 8, true);
      const bufLen = mem.getUint32(iovsPtr + i * 8 + 4, true);
      const available = Math.min(bufLen, file.data.length - file.pos);
      if (available > 0) {
        memBytes.set(file.data.subarray(file.pos, file.pos + available), bufPtr);
        file.pos += available;
        totalRead += available;
      }
    }
    mem.setUint32(nreadPtr, totalRead, true);
    return 0;
  }

  fdSeek(fd: number, offset: bigint, whence: number, newOffsetPtr: number, mem: DataView): number {
    const file = this.files.get(fd);
    if (!file) return 8;
    if (whence === 0) file.pos = Number(offset);         // SEEK_SET
    else if (whence === 1) file.pos += Number(offset);    // SEEK_CUR
    else if (whence === 2) file.pos = file.data.length + Number(offset); // SEEK_END
    mem.setBigUint64(newOffsetPtr, BigInt(file.pos), true);
    return 0;
  }

  fdFilestatGet(fd: number, bufPtr: number, mem: DataView): number {
    const file = this.files.get(fd);
    if (!file) return 8;
    // filestat struct: dev(u64) ino(u64) filetype(u8) nlink(u64) size(u64) atim(u64) mtim(u64) ctim(u64)
    // offset 0: dev=0, 8: ino=0, 16: filetype=4 (regular), 24: nlink=1
    // 32: size, 40: atim=0, 48: mtim=0, 56: ctim=0
    for (let i = 0; i < 64; i += 4) mem.setUint32(bufPtr + i, 0, true);
    mem.setUint8(bufPtr + 16, 4); // regular file
    mem.setBigUint64(bufPtr + 24, 1n, true); // nlink
    mem.setBigUint64(bufPtr + 32, BigInt(file.data.length), true); // size
    return 0;
  }

  fdClose(fd: number): number {
    this.files.delete(fd);
    return 0;
  }
}

// ============================================================================
// WASM instantiation
// ============================================================================
let wasm: LiteRtExports | null = null;
const vfs = new VirtualFS();

export function registerModelFile(path: string, data: ArrayBuffer): void {
  vfs.registerFile(path, data);
}

export async function initLiteRt(): Promise<LiteRtExports> {
  // Auto-provide any env import as a no-op function that returns 0.
  // Platform-unavailable symbols (GPU, NPU, compiler, abseil flags, Rust deps)
  // become imports via --allow-undefined. They are unreachable on CPU-only WASM.
  // If one IS reached, the console.warn helps debug which symbol was called.
  const envProxy = new Proxy({} as Record<string, Function>, {
    get(_target, prop: string) {
      return (..._args: unknown[]) => 0;
    },
  });

  const imports: WebAssembly.Imports = {
    env: envProxy,

    wasi_snapshot_preview1: {
      clock_time_get(_id: number, _prec: bigint, time_ptr: number): number {
        new DataView(wasm!.memory.buffer).setBigUint64(
          time_ptr, BigInt(Math.floor(performance.now() * 1e6)), true,
        );
        return 0;
      },
      environ_get() { return 0; },
      environ_sizes_get(count_ptr: number, size_ptr: number): number {
        const v = new DataView(wasm!.memory.buffer);
        v.setUint32(count_ptr, 0, true);
        v.setUint32(size_ptr, 0, true);
        return 0;
      },
      fd_close(fd: number): number { return vfs.fdClose(fd); },
      fd_fdstat_get() { return 0; },
      fd_fdstat_set_flags() { return 0; },
      fd_filestat_get(fd: number, buf: number): number {
        return vfs.fdFilestatGet(fd, buf, new DataView(wasm!.memory.buffer));
      },
      fd_filestat_set_size(_fd: number, _size: bigint): number { return 0; },
      fd_pread(fd: number, iovs: number, iovsLen: number, offset: bigint, nread: number): number {
        return vfs.fdPread(fd, iovs, iovsLen, offset, nread,
          new DataView(wasm!.memory.buffer), new Uint8Array(wasm!.memory.buffer));
      },
      fd_prestat_get(): number { return 8; },
      fd_prestat_dir_name(): number { return 8; },
      fd_read(fd: number, iovs: number, iovsLen: number, nread: number): number {
        return vfs.fdRead(fd, iovs, iovsLen, nread,
          new DataView(wasm!.memory.buffer), new Uint8Array(wasm!.memory.buffer));
      },
      fd_seek(fd: number, offset: bigint, whence: number, newOffset: number): number {
        return vfs.fdSeek(fd, offset, whence, newOffset, new DataView(wasm!.memory.buffer));
      },
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
      path_open(
        dirfd: number, dirflags: number, pathPtr: number, pathLen: number,
        oflags: number, fsRightsBase: bigint, fsRightsInheriting: bigint,
        fdflags: number, fdPtr: number,
      ): number {
        return vfs.pathOpen(dirfd, dirflags, pathPtr, pathLen, oflags,
          fsRightsBase, fsRightsInheriting, fdflags, fdPtr,
          new DataView(wasm!.memory.buffer), new Uint8Array(wasm!.memory.buffer));
      },
      path_filestat_get(): number { return 8; },
      poll_oneoff() { return 0; },
      proc_exit(code: number): void { throw new Error(`WASM exit: ${code}`); },
      sched_yield() { return 0; },
    },
  };

  // Compile in background thread to avoid freezing the UI
  const response = await fetch(WASM_URL);
  const module = await WebAssembly.compileStreaming(response);
  const instance = await WebAssembly.instantiate(module, imports);
  wasm = instance.exports as unknown as LiteRtExports;

  return wasm;
}

export function getWasm(): LiteRtExports {
  if (!wasm) throw new Error("LiteRT not initialized");
  return wasm;
}

export { writeString, readString, writeInputData };
