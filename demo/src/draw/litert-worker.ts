/**
 * LiteRT-LM WASM Worker.
 * Runs inference in a Web Worker with synchronous OPFS file access.
 * Model bytes are served directly from disk via FileSystemSyncAccessHandle —
 * never loaded into JS heap memory.
 *
 * Communication with main thread via postMessage:
 *   Main → Worker: { type: "init", modelFileName: string }
 *   Main → Worker: { type: "generate", text: string }
 *   Worker → Main: { type: "status", text: string }
 *   Worker → Main: { type: "result", text: string, ttft?: number, tokPerSec?: number }
 *   Worker → Main: { type: "error", text: string }
 */

// ============================================================================
// Synchronous OPFS-backed WASI filesystem
// ============================================================================
class SyncOPFS {
  private handles = new Map<number, FileSystemSyncAccessHandle>();
  private sizes = new Map<number, number>();
  private positions = new Map<number, number>();
  private pathToHandle = new Map<string, FileSystemSyncAccessHandle>();
  private pathToSize = new Map<string, number>();
  private nextFd = 10;
  private preadCount = 0;

  async registerFile(path: string): Promise<void> {
    const root = await navigator.storage.getDirectory();
    const fileHandle = await root.getFileHandle(path);
    const accessHandle = await fileHandle.createSyncAccessHandle();
    const size = accessHandle.getSize();
    this.pathToHandle.set(path, accessHandle);
    this.pathToHandle.set("/" + path, accessHandle);
    this.pathToSize.set(path, size);
    this.pathToSize.set("/" + path, size);
  }

  pathOpen(
    pathPtr: number, pathLen: number, fdPtr: number,
    mem: DataView, memBytes: Uint8Array,
  ): number {
    const pathStr = new TextDecoder().decode(memBytes.subarray(pathPtr, pathPtr + pathLen));
    console.log(`[VFS] path_open: "${pathStr}"`);
    const handle = this.pathToHandle.get(pathStr) ||
                   this.pathToHandle.get(pathStr.replace(/^\/+/, ""));
    if (!handle) {
      console.warn(`[VFS] path_open: NOT FOUND: "${pathStr}" (have: ${[...this.pathToHandle.keys()].join(', ')})`);
      return 44; // ENOENT
    }
    const fd = this.nextFd++;
    const size = this.pathToSize.get(pathStr) || this.pathToSize.get(pathStr.replace(/^\/+/, "")) || 0;
    console.log(`[VFS] path_open: OK fd=${fd} size=${size}`);
    this.handles.set(fd, handle);
    this.sizes.set(fd, size);
    this.positions.set(fd, 0);
    mem.setUint32(fdPtr, fd, true);
    return 0;
  }

  fdPread(
    fd: number, iovsPtr: number, iovsLen: number, offset: bigint,
    nreadPtr: number, mem: DataView, memBytes: Uint8Array,
  ): number {
    const handle = this.handles.get(fd);
    if (!handle) return 8; // EBADF
    const fileSize = this.sizes.get(fd) || 0;
    let totalRead = 0;
    let fileOffset = Number(offset);
    for (let i = 0; i < iovsLen; i++) {
      const bufPtr = mem.getUint32(iovsPtr + i * 8, true);
      const bufLen = mem.getUint32(iovsPtr + i * 8 + 4, true);
      const available = Math.min(bufLen, fileSize - fileOffset);
      if (available > 0) {
        const chunk = new Uint8Array(available);
        const bytesRead = handle.read(chunk, { at: fileOffset });
        memBytes.set(chunk.subarray(0, bytesRead), bufPtr);
        fileOffset += bytesRead;
        totalRead += bytesRead;
      }
    }
    if (this.preadCount < 5) {
      console.log(`[VFS] fd_pread: fd=${fd} offset=${offset} read=${totalRead} fileSize=${fileSize}`);
    }
    this.preadCount++;
    mem.setUint32(nreadPtr, totalRead, true);
    return 0;
  }

  fdRead(
    fd: number, iovsPtr: number, iovsLen: number,
    nreadPtr: number, mem: DataView, memBytes: Uint8Array,
  ): number {
    const handle = this.handles.get(fd);
    if (!handle) return 8;
    const pos = this.positions.get(fd) || 0;
    const fileSize = this.sizes.get(fd) || 0;
    let totalRead = 0;
    let fileOffset = pos;
    for (let i = 0; i < iovsLen; i++) {
      const bufPtr = mem.getUint32(iovsPtr + i * 8, true);
      const bufLen = mem.getUint32(iovsPtr + i * 8 + 4, true);
      const available = Math.min(bufLen, fileSize - fileOffset);
      if (available > 0) {
        const chunk = new Uint8Array(available);
        const bytesRead = handle.read(chunk, { at: fileOffset });
        memBytes.set(chunk.subarray(0, bytesRead), bufPtr);
        fileOffset += bytesRead;
        totalRead += bytesRead;
      }
    }
    this.positions.set(fd, fileOffset);
    mem.setUint32(nreadPtr, totalRead, true);
    return 0;
  }

  fdSeek(fd: number, offset: bigint, whence: number, newOffsetPtr: number, mem: DataView): number {
    const fileSize = this.sizes.get(fd) || 0;
    let pos = this.positions.get(fd) || 0;
    if (whence === 0) pos = Number(offset);
    else if (whence === 1) pos += Number(offset);
    else if (whence === 2) pos = fileSize + Number(offset);
    this.positions.set(fd, pos);
    mem.setBigUint64(newOffsetPtr, BigInt(pos), true);
    return 0;
  }

  fdFilestatGet(fd: number, bufPtr: number, mem: DataView): number {
    const size = this.sizes.get(fd);
    if (size === undefined) return 8;
    for (let i = 0; i < 64; i += 4) mem.setUint32(bufPtr + i, 0, true);
    mem.setUint8(bufPtr + 16, 4); // regular file
    mem.setBigUint64(bufPtr + 24, 1n, true); // nlink
    mem.setBigUint64(bufPtr + 32, BigInt(size), true);
    return 0;
  }

  fdClose(fd: number): number {
    this.handles.delete(fd);
    this.sizes.delete(fd);
    this.positions.delete(fd);
    return 0;
  }

  close(): void {
    for (const handle of this.pathToHandle.values()) {
      handle.close();
    }
  }
}

// ============================================================================
// WASM instantiation + inference loop
// ============================================================================
interface LiteRtExports {
  memory: WebAssembly.Memory;
  wasm_malloc(size: number): number;
  wasm_free(ptr: number): void;
  litert_lm_engine_settings_create(p: number, b: number, v: number, a: number): number;
  litert_lm_engine_settings_delete(s: number): void;
  litert_lm_engine_settings_set_max_num_tokens(s: number, n: number): void;
  litert_lm_engine_settings_enable_benchmark(s: number): void;
  litert_lm_engine_create(s: number): number;
  litert_lm_engine_delete(e: number): void;
  litert_lm_engine_create_session(e: number, c: number): number;
  litert_lm_session_config_create(): number;
  litert_lm_session_config_delete(c: number): void;
  litert_lm_session_config_set_max_output_tokens(c: number, n: number): void;
  litert_lm_session_delete(s: number): void;
  litert_lm_session_generate_content(s: number, i: number, n: number): number;
  litert_lm_responses_delete(r: number): void;
  litert_lm_responses_get_response_text_at(r: number, i: number): number;
  litert_lm_session_get_benchmark_info(s: number): number;
  litert_lm_benchmark_info_delete(b: number): void;
  litert_lm_benchmark_info_get_time_to_first_token(b: number): number;
  litert_lm_benchmark_info_get_decode_tokens_per_sec_at(b: number, i: number): number;
}

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

let wasm: LiteRtExports;
let vfs: SyncOPFS;
let session = 0;

async function init(modelFileName: string) {
  vfs = new SyncOPFS();
  await vfs.registerFile(modelFileName);
  postMessage({ type: "status", text: "Model file opened. Loading WASM..." });

  const envCalled = new Set<string>();
  const envProxy = new Proxy({} as Record<string, Function>, {
    get(_target, prop: string) {
      return (..._args: unknown[]) => {
        if (!envCalled.has(prop)) {
          envCalled.add(prop);
          console.log(`[env] ${prop}`);
        }
        return 0;
      };
    },
  });

  const wasiImports = {
    clock_time_get(_id: number, _prec: bigint, time_ptr: number): number {
      new DataView(wasm.memory.buffer).setBigUint64(
        time_ptr, BigInt(Math.floor(performance.now() * 1e6)), true);
      return 0;
    },
    environ_get() { return 0; },
    environ_sizes_get(cp: number, sp: number): number {
      const v = new DataView(wasm.memory.buffer);
      v.setUint32(cp, 0, true); v.setUint32(sp, 0, true);
      return 0;
    },
    fd_close(fd: number) { return vfs.fdClose(fd); },
    fd_fdstat_get() { return 0; },
    fd_fdstat_set_flags() { return 0; },
    fd_filestat_get(fd: number, buf: number) {
      return vfs.fdFilestatGet(fd, buf, new DataView(wasm.memory.buffer));
    },
    fd_filestat_set_size() { return 0; },
    fd_pread(fd: number, iovs: number, iovsLen: number, offset: bigint, nread: number) {
      return vfs.fdPread(fd, iovs, iovsLen, offset, nread,
        new DataView(wasm.memory.buffer), new Uint8Array(wasm.memory.buffer));
    },
    fd_prestat_get(fd: number, bufPtr: number): number {
      // Preopened directory fd=3 → "/" (root)
      // prestat struct: { tag: u8, dir_name_len: u32 } at offset 0,4
      if (fd === 3) {
        const v = new DataView(wasm.memory.buffer);
        v.setUint8(bufPtr, 0); // tag = __WASI_PREOPENTYPE_DIR
        v.setUint32(bufPtr + 4, 1, true); // dir_name_len = 1 ("/" length)
        return 0;
      }
      return 8; // EBADF for other fds
    },
    fd_prestat_dir_name(fd: number, pathPtr: number, pathLen: number): number {
      if (fd === 3) {
        new Uint8Array(wasm.memory.buffer)[pathPtr] = 47; // '/'
        return 0;
      }
      return 8;
    },
    fd_read(fd: number, iovs: number, iovsLen: number, nread: number) {
      return vfs.fdRead(fd, iovs, iovsLen, nread,
        new DataView(wasm.memory.buffer), new Uint8Array(wasm.memory.buffer));
    },
    fd_seek(fd: number, offset: bigint, whence: number, newOffset: number) {
      return vfs.fdSeek(fd, offset, whence, newOffset, new DataView(wasm.memory.buffer));
    },
    fd_write(_fd: number, iovsPtr: number, iovsLen: number, nwrittenPtr: number): number {
      const v = new DataView(wasm.memory.buffer);
      const mem = new Uint8Array(wasm.memory.buffer);
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
      _dirfd: number, _dirflags: number, pathPtr: number, pathLen: number,
      _oflags: number, _fsRightsBase: bigint, _fsRightsInheriting: bigint,
      _fdflags: number, fdPtr: number,
    ) {
      return vfs.pathOpen(pathPtr, pathLen, fdPtr,
        new DataView(wasm.memory.buffer), new Uint8Array(wasm.memory.buffer));
    },
    path_filestat_get() { return 8; },
    poll_oneoff() { return 0; },
    proc_exit(code: number) { throw new Error(`WASM exit: ${code}`); },
    sched_yield() { return 0; },
  };

  const response = await fetch("/turboquant-litert.wasm");
  const module = await WebAssembly.compileStreaming(response);
  const instance = await WebAssembly.instantiate(module, {
    env: envProxy,
    wasi_snapshot_preview1: wasiImports,
  });
  wasm = instance.exports as unknown as LiteRtExports;
  postMessage({ type: "status", text: "WASM loaded. Creating engine..." });

  // Create engine
  const modelPath = `/${modelFileName}`;
  console.log("[worker] Creating engine with model:", modelPath);

  const pathPtr = writeString(wasm, modelPath);
  const cpuPtr = writeString(wasm, "cpu");
  console.log("[worker] engine_settings_create...");
  const settings = wasm.litert_lm_engine_settings_create(pathPtr, cpuPtr, 0, 0);
  console.log("[worker] settings =", settings);
  if (!settings) {
    postMessage({ type: "error", text: "engine_settings_create returned null" });
    return;
  }
  wasm.litert_lm_engine_settings_set_max_num_tokens(settings, 2048);
  wasm.litert_lm_engine_settings_enable_benchmark(settings);
  wasm.wasm_free(pathPtr);
  wasm.wasm_free(cpuPtr);

  // Capture C++ error messages during engine_create
  let wasmErrors: string[] = [];
  const origFdWrite = wasiImports.fd_write;
  wasiImports.fd_write = (_fd: number, iovsPtr: number, iovsLen: number, nwrittenPtr: number): number => {
    const v = new DataView(wasm.memory.buffer);
    const mem = new Uint8Array(wasm.memory.buffer);
    let written = 0;
    for (let i = 0; i < iovsLen; i++) {
      const ptr = v.getUint32(iovsPtr + i * 8, true);
      const len = v.getUint32(iovsPtr + i * 8 + 4, true);
      const text = new TextDecoder().decode(mem.subarray(ptr, ptr + len));
      wasmErrors.push(text);
      console.log("[wasm]", text);
      written += len;
    }
    v.setUint32(nwrittenPtr, written, true);
    return 0;
  };

  console.log("[worker] engine_create...");
  const engine = wasm.litert_lm_engine_create(settings);
  console.log("[worker] engine =", engine);
  wasiImports.fd_write = origFdWrite;

  if (!engine) {
    const errMsg = wasmErrors.join("").trim() || "engine_create returned null (no error message from C++)";
    postMessage({ type: "error", text: errMsg });
    return;
  }

  const config = wasm.litert_lm_session_config_create();
  wasm.litert_lm_session_config_set_max_output_tokens(config, 512);
  session = wasm.litert_lm_engine_create_session(engine, config);
  wasm.litert_lm_session_config_delete(config);
  wasm.litert_lm_engine_settings_delete(settings);

  postMessage({ type: "status", text: "Gemma 4 E2B ready" });
}

function generate(text: string) {
  if (!session) {
    postMessage({ type: "error", text: "Session not initialized" });
    return;
  }

  const textPtr = writeString(wasm, text);
  const textLen = new TextEncoder().encode(text).length;
  const structPtr = wasm.wasm_malloc(12);
  const v = new DataView(wasm.memory.buffer);
  v.setInt32(structPtr, 0, true);
  v.setUint32(structPtr + 4, textPtr, true);
  v.setUint32(structPtr + 8, textLen, true);

  const responses = wasm.litert_lm_session_generate_content(session, structPtr, 1);
  wasm.wasm_free(structPtr);
  wasm.wasm_free(textPtr);

  if (responses) {
    const resultPtr = wasm.litert_lm_responses_get_response_text_at(responses, 0);
    const result = resultPtr ? readString(wasm, resultPtr) : "";

    let ttft = 0, tokPerSec = 0;
    const bench = wasm.litert_lm_session_get_benchmark_info(session);
    if (bench) {
      ttft = wasm.litert_lm_benchmark_info_get_time_to_first_token(bench);
      tokPerSec = wasm.litert_lm_benchmark_info_get_decode_tokens_per_sec_at(bench, 0);
      wasm.litert_lm_benchmark_info_delete(bench);
    }
    wasm.litert_lm_responses_delete(responses);

    postMessage({ type: "result", text: result, ttft, tokPerSec });
  } else {
    postMessage({ type: "error", text: "No response from model" });
  }
}

// Message handler
self.onmessage = async (e: MessageEvent) => {
  try {
    if (e.data.type === "init") {
      await init(e.data.modelFileName);
    } else if (e.data.type === "generate") {
      generate(e.data.text);
    }
  } catch (err) {
    postMessage({ type: "error", text: (err as Error).message });
  }
};
