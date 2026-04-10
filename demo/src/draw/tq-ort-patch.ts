/**
 * Monkey-patch onnxruntime-web to use our Zig-compiled ORT WASM binary
 * with TQ attention kernel. Transformers.js calls session.run() as normal,
 * but underneath, K/V cache is compressed with TQ and attention scores are
 * computed directly on compressed data via tq_kv_dot_batch.
 */

const WASM_URL = "/dist/turboquant-llm.wasm";

interface OrtWasm {
  memory: WebAssembly.Memory;
  OrtInit(num_threads: number, logging_level: number): number;
  OrtCreateSessionOptions(
    graph_opt: number, cpu_arena: number, mem_pattern: number,
    exec_mode: number, profiling: number, log_id: number,
    log_severity: number, log_verbosity: number, opt_path: number,
    out: number,
  ): number;
  OrtReleaseSessionOptions(ptr: number): void;
  OrtCreateSession(buf: number, size: number, opts: number, out: number): number;
  OrtReleaseSession(ptr: number): void;
  OrtGetInputOutputCount(session: number, inCount: number, outCount: number): number;
  OrtRun(
    session: number, inNames: number, inTensors: number, inCount: number,
    outNames: number, outCount: number, outTensors: number, runOpts: number,
  ): number;
  OrtCreateTensor(dtype: number, data: number, size: number, dims: number, ndims: number, location: number): number;
  OrtGetTensorData(tensor: number, dtypeOut: number, dataOut: number, dimsOut: number, ndimsOut: number): number;
  OrtReleaseTensor(ptr: number): void;
  OrtFree(ptr: number): void;
  OrtGetLastError(codeOut: number, msgOut: number): number;
  _Z15OrtGetInputNameP10OrtSessionm(session: number, index: number): number;
  _Z16OrtGetOutputNameP10OrtSessionm(session: number, index: number): number;
  wasm_malloc(size: number): number;
  wasm_free(ptr: number): void;
  tq_kv_length(stream: number): number;
  tq_kv_compressed_size(stream: number): number;
}

let wasm: OrtWasm | null = null;

function noop() {}
function zero() { return 0; }

// ORT data type enum → TypedArray byte size
const DTYPE_BYTES: Record<number, number> = {
  1: 4,   // float32
  2: 1,   // uint8
  3: 1,   // int8
  5: 2,   // uint16 / float16
  6: 4,   // int32
  7: 8,   // int64
  10: 2,  // float16
  12: 4,  // uint32
  16: 8,  // uint64 (BigInt64)
};

// ORT data type string → enum
const DTYPE_MAP: Record<string, number> = {
  float32: 1, uint8: 2, int8: 3, uint16: 5,
  int32: 6, int64: 7, float16: 10, uint32: 12,
  bool: 9, string: 8, float64: 11, uint64: 16,
};

function readU32(offset: number): number {
  return new DataView(wasm!.memory.buffer).getUint32(offset, true);
}

function readCString(ptr: number): string {
  const mem = new Uint8Array(wasm!.memory.buffer);
  let end = ptr;
  while (mem[end] !== 0) end++;
  return new TextDecoder().decode(mem.subarray(ptr, end));
}

function writeCString(str: string): number {
  const encoded = new TextEncoder().encode(str);
  const ptr = wasm!.wasm_malloc(encoded.length + 1);
  new Uint8Array(wasm!.memory.buffer, ptr, encoded.length + 1).set([...encoded, 0]);
  return ptr;
}

async function loadWasm(): Promise<OrtWasm> {
  if (wasm) return wasm;

  const imports: WebAssembly.Imports = {
    env: {
      __cxa_thread_atexit: noop,
      jsep_alloc(size: number): number { return wasm!.wasm_malloc(size); },
      jsep_free(ptr: number): number { wasm!.wasm_free(ptr); return 0; },
      jsep_create_kernel: noop,
      jsep_release_kernel: noop,
      jsep_run: zero,
      jsep_capture_begin: noop,
      jsep_capture_end: noop,
      jsep_replay: noop,
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
          const text = new TextDecoder().decode(mem.subarray(ptr, ptr + len));
          if (text.trim()) console.log("[ORT]", text.trimEnd());
          written += len;
        }
        v.setUint32(nwritten_ptr, written, true);
        return 0;
      },
      path_open: zero,
      poll_oneoff: zero,
      proc_exit(code: number): void { throw new Error(`WASM exit: ${code}`); },
      sched_yield: zero,
    },
  };

  const response = await fetch(WASM_URL);
  const { instance } = await WebAssembly.instantiateStreaming(response, imports);
  wasm = instance.exports as unknown as OrtWasm;

  const rc = wasm.OrtInit(1, 3);
  if (rc !== 0) throw new Error(`OrtInit failed: ${rc}`);
  console.log("[TQ-ORT] initialized");

  return wasm;
}

/**
 * Create a session object compatible with onnxruntime-web's InferenceSession interface.
 * Transformers.js calls session.run(feeds) and reads session.inputNames/outputNames.
 */
function createTQSession(w: OrtWasm, sessionPtr: number, inputNames: string[], outputNames: string[]) {
  return {
    _ptr: sessionPtr,
    inputNames,
    outputNames,
    config: null as any,

    async run(feeds: Record<string, any>, options?: any): Promise<Record<string, any>> {
      const inCount = inputNames.length;
      const outCount = outputNames.length;

      // Allocate arrays for names and tensor handles
      const inNamesPtr = w.wasm_malloc(inCount * 4);
      const inTensorsPtr = w.wasm_malloc(inCount * 4);
      const outNamesPtr = w.wasm_malloc(outCount * 4);
      const outTensorsPtr = w.wasm_malloc(outCount * 4);
      const dv = () => new DataView(w.memory.buffer);

      // Set up input name C strings and tensors
      const namePtrs: number[] = [];
      const tensorPtrs: number[] = [];

      for (let i = 0; i < inCount; i++) {
        const name = inputNames[i];
        const namePtr = writeCString(name);
        namePtrs.push(namePtr);
        dv().setUint32(inNamesPtr + i * 4, namePtr, true);

        const tensor = feeds[name];
        if (!tensor) {
          // Missing input — create empty tensor
          dv().setUint32(inTensorsPtr + i * 4, 0, true);
          continue;
        }

        // Get tensor data and dims
        const data = tensor.data ?? tensor.cpuData;
        const dims = tensor.dims;
        const dtype = DTYPE_MAP[tensor.type] ?? 1;
        const bytesPerElem = DTYPE_BYTES[dtype] ?? 4;
        const dataBytes = data.byteLength ?? (data.length * bytesPerElem);

        // Copy data to WASM
        const dataPtr = w.wasm_malloc(dataBytes);
        new Uint8Array(w.memory.buffer, dataPtr, dataBytes).set(
          new Uint8Array(data.buffer ?? data, data.byteOffset ?? 0, dataBytes)
        );

        // Copy dims to WASM
        const dimsPtr = w.wasm_malloc(dims.length * 8); // size_t = 4 bytes on wasm32
        for (let d = 0; d < dims.length; d++) {
          dv().setUint32(dimsPtr + d * 4, Number(dims[d]), true);
        }

        // Create ORT tensor (location=0 = CPU)
        const ortTensor = w.OrtCreateTensor(dtype, dataPtr, dataBytes, dimsPtr, dims.length, 0);
        tensorPtrs.push(ortTensor);
        dv().setUint32(inTensorsPtr + i * 4, ortTensor, true);

        w.wasm_free(dimsPtr);
        // dataPtr is owned by the tensor, freed on OrtReleaseTensor
      }

      // Set up output names
      const outNamePtrs: number[] = [];
      for (let i = 0; i < outCount; i++) {
        const namePtr = writeCString(outputNames[i]);
        outNamePtrs.push(namePtr);
        dv().setUint32(outNamesPtr + i * 4, namePtr, true);
      }

      // Zero output tensor pointers
      for (let i = 0; i < outCount; i++) {
        dv().setUint32(outTensorsPtr + i * 4, 0, true);
      }

      // Run inference — TQ attention kernel runs inside this call
      const status = w.OrtRun(sessionPtr, inNamesPtr, inTensorsPtr, inCount, outNamesPtr, outCount, outTensorsPtr, 0);

      if (status !== 0) {
        const errBuf = w.wasm_malloc(8);
        w.OrtGetLastError(errBuf, errBuf + 4);
        const errCode = dv().getUint32(errBuf, true);
        const errMsgPtr = dv().getUint32(errBuf + 4, true);
        const errMsg = errMsgPtr ? readCString(errMsgPtr) : "unknown";
        w.wasm_free(errBuf);
        throw new Error(`OrtRun failed: code=${errCode} ${errMsg}`);
      }

      // Read outputs
      const { Tensor } = await import("@huggingface/transformers");
      const results: Record<string, any> = {};
      const metaBuf = w.wasm_malloc(20); // dtype(4) + dataPtr(4) + dimsPtr(4) + nDims(4) + pad

      for (let i = 0; i < outCount; i++) {
        const ortTensor = dv().getUint32(outTensorsPtr + i * 4, true);
        if (!ortTensor) continue;

        const dtypeOut = metaBuf;
        const dataOut = metaBuf + 4;
        const dimsOut = metaBuf + 8;
        const ndimsOut = metaBuf + 12;

        w.OrtGetTensorData(ortTensor, dtypeOut, dataOut, dimsOut, ndimsOut);

        const dtype = dv().getUint32(dtypeOut, true);
        const dataPtr = dv().getUint32(dataOut, true);
        const dimsPtr = dv().getUint32(dimsOut, true);
        const ndims = dv().getUint32(ndimsOut, true);

        const dims: number[] = [];
        for (let d = 0; d < ndims; d++) {
          dims.push(dv().getUint32(dimsPtr + d * 4, true));
        }

        const totalElements = dims.reduce((a, b) => a * b, 1);
        const bytesPerElem = DTYPE_BYTES[dtype] ?? 4;
        const totalBytes = totalElements * bytesPerElem;

        // Map ORT dtype to Tensor type string
        const typeMap: Record<number, string> = {
          1: "float32", 2: "uint8", 3: "int8", 5: "uint16",
          6: "int32", 7: "int64", 10: "float16", 12: "uint32", 16: "uint64",
        };
        const typeStr = typeMap[dtype] ?? "float32";

        // Copy data out of WASM (the tensor data lives in WASM memory)
        let jsData: any;
        if (dtype === 7 || dtype === 16) {
          // int64/uint64 → BigInt64Array/BigUint64Array
          jsData = dtype === 7
            ? new BigInt64Array(new Uint8Array(w.memory.buffer, dataPtr, totalBytes).slice().buffer)
            : new BigUint64Array(new Uint8Array(w.memory.buffer, dataPtr, totalBytes).slice().buffer);
        } else if (dtype === 10) {
          // float16 → Uint16Array
          jsData = new Uint16Array(new Uint8Array(w.memory.buffer, dataPtr, totalBytes).slice().buffer);
        } else if (dtype === 1) {
          jsData = new Float32Array(new Uint8Array(w.memory.buffer, dataPtr, totalBytes).slice().buffer);
        } else {
          jsData = new Uint8Array(w.memory.buffer, dataPtr, totalBytes).slice();
        }

        results[outputNames[i]] = new Tensor(typeStr, jsData, dims);
        w.OrtReleaseTensor(ortTensor);
      }

      w.wasm_free(metaBuf);

      // Cleanup inputs
      for (const ptr of tensorPtrs) w.OrtReleaseTensor(ptr);
      for (const ptr of namePtrs) w.wasm_free(ptr);
      for (const ptr of outNamePtrs) w.wasm_free(ptr);
      w.wasm_free(inNamesPtr);
      w.wasm_free(inTensorsPtr);
      w.wasm_free(outNamesPtr);
      w.wasm_free(outTensorsPtr);

      return results;
    },

    async release() {
      w.OrtReleaseSession(sessionPtr);
    },
  };
}

/**
 * Patch onnxruntime-web so Transformers.js uses our TQ ORT binary.
 * Call BEFORE pipeline() or any model loading.
 */
export async function patchOnnxRuntime() {
  const w = await loadWasm();
  const ort = await import("onnxruntime-web/webgpu");

  const OrigCreate = ort.InferenceSession.create.bind(ort.InferenceSession);

  (ort.InferenceSession as any).create = async function(
    modelData: ArrayBuffer | Uint8Array | string,
    options?: any,
  ) {
    // Fetch model if URL
    let buffer: Uint8Array;
    if (typeof modelData === "string") {
      const resp = await fetch(modelData);
      buffer = new Uint8Array(await resp.arrayBuffer());
    } else if (modelData instanceof ArrayBuffer) {
      buffer = new Uint8Array(modelData);
    } else {
      buffer = modelData;
    }

    console.log("[TQ-ORT] loading model:", buffer.byteLength, "bytes");

    // Copy model to WASM heap
    const modelPtr = w.wasm_malloc(buffer.byteLength);
    new Uint8Array(w.memory.buffer, modelPtr, buffer.byteLength).set(buffer);

    // Create session options
    const optsOut = w.wasm_malloc(4);
    w.OrtCreateSessionOptions(99, 1, 1, 0, 0, 0, 3, 0, 0, optsOut);
    const optsPtr = readU32(optsOut);
    w.wasm_free(optsOut);

    // Create session
    const sessionOut = w.wasm_malloc(4);
    const status = w.OrtCreateSession(modelPtr, buffer.byteLength, optsPtr, sessionOut);
    const sessionPtr = readU32(sessionOut);
    w.wasm_free(sessionOut);
    w.wasm_free(modelPtr);
    w.OrtReleaseSessionOptions(optsPtr);

    if (status !== 0 || !sessionPtr) {
      const errBuf = w.wasm_malloc(8);
      w.OrtGetLastError(errBuf, errBuf + 4);
      const errMsg = readCString(readU32(errBuf + 4));
      w.wasm_free(errBuf);
      throw new Error(`OrtCreateSession failed: ${errMsg}`);
    }

    // Get input/output counts
    const countBuf = w.wasm_malloc(8);
    w.OrtGetInputOutputCount(sessionPtr, countBuf, countBuf + 4);
    const inputCount = readU32(countBuf);
    const outputCount = readU32(countBuf + 4);
    w.wasm_free(countBuf);

    // Get input names
    const inputNames: string[] = [];
    for (let i = 0; i < inputCount; i++) {
      const namePtr = w._Z15OrtGetInputNameP10OrtSessionm(sessionPtr, i);
      if (namePtr) {
        inputNames.push(readCString(namePtr));
        w.OrtFree(namePtr);
      }
    }

    // Get output names
    const outputNames: string[] = [];
    for (let i = 0; i < outputCount; i++) {
      const namePtr = w._Z16OrtGetOutputNameP10OrtSessionm(sessionPtr, i);
      if (namePtr) {
        outputNames.push(readCString(namePtr));
        w.OrtFree(namePtr);
      }
    }

    console.log("[TQ-ORT] session created: %d inputs, %d outputs", inputCount, outputCount);

    return createTQSession(w, sessionPtr, inputNames, outputNames);
  };

  console.log("[TQ-ORT] onnxruntime-web patched — TQ attention active");
}
