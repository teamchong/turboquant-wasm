/**
 * LiteRT-LM + TurboQuant unified WASM glue.
 * Provides WASI filesystem, env imports, and C API wrapper for the
 * Zig-compiled LiteRT-LM binary (9.5 MB, 1313 objects).
 *
 * 89 imports total: 19 WASI + 70 env.
 */

const WASM_URL = "/dist/turboquant-litert.wasm";

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
  // Generic no-op for env imports (GPU, NPU, compiler, etc.)
  function trap(): never { throw new Error("Unreachable WASM import called"); }
  function noop() {}
  function zero() { return 0; }
  function zeroN(..._args: unknown[]) { return 0; }

  const imports: WebAssembly.Imports = {
    env: {
      // Abseil threading/sync — single-threaded WASM
      AbslInternalPerThreadSemPost_lts_20250814: noop,
      AbslInternalPerThreadSemWait_lts_20250814: zeroN,

      // Abseil time — UTC only
      '_ZN4absl12lts_2025081413time_internal4cctz12TimeZoneInfo3UTCEv': zero,
      '_ZN4absl12lts_2025081413time_internal4cctz12TimeZoneInfo4MakeERKNSt3__112basic_stringIcNS4_11char_traitsIcEENS4_9allocatorIcEEEE': zero,
      '_ZN4absl12lts_2025081415random_internal21InverseNormalSurvivalEd': () => 0.0,

      // UTF-8 validation — return "valid"
      utf8_range: zeroN,
      utf8_lemire: zeroN,
      utf8_range2: zeroN,

      // LiteRT accelerator registration — CPU only on WASM
      '_ZN6litert39TriggerAcceleratorAutomaticRegistrationER18LiteRtEnvironmentT': noop,

      // GPU options — all no-ops (GPU not available)
      '_ZN6litert10GpuOptions6CreateEv': zero,
      '_ZN6litert10GpuOptions27EnableConstantTensorSharingEb': noop,
      '_ZN6litert10GpuOptions12SetPrecisionENS0_9PrecisionE': noop,
      '_ZN6litert10GpuOptions23SetPreferTextureWeightsEb': noop,
      '_ZN6litert10GpuOptions16SetModelCacheKeyEPKc': noop,
      '_ZN6litert10GpuOptions19SetSerializationDirEPKc': noop,
      '_ZN6litert10GpuOptions27SetSerializeExternalTensorsEb': noop,
      '_ZN6litert10GpuOptions17SetProgramCacheFdEi': noop,
      '_ZN6litert10GpuOptions24SetSerializeProgramCacheEb': noop,
      '_ZN6litert10GpuOptions26EnableInfiniteFloatCappingEb': noop,
      '_ZN6litert10GpuOptions25CacheCompiledProgramsOnlyEb': noop,
      '_ZN6litert10GpuOptions25EnableExternalTensorsModeEb': noop,
      '_ZN6litert10GpuOptions24AddExternalTensorPatternEPKc': noop,
      '_ZN6litert10GpuOptions29AddBufferStorageTensorPatternEPKc': noop,
      '_ZN6litert10GpuOptions37SetHintFullyDelegatedToSingleDelegateEb': noop,
      '_ZN6litert10GpuOptions31SetMadviseOriginalSharedTensorsEb': noop,
      '_ZN6litert10GpuOptions22SetConvertWeightsOnGpuEb': noop,
      '_ZN6litert10GpuOptions32EnableAllowSrcQuantizedFcConvOpsEb': noop,
      '_ZN6litert10GpuOptions24HintWaitingForCompletionEb': noop,
      '_ZN6litert10GpuOptions28SetSyncExecutionModeWaitTypeENS0_25SyncExecutionModeWaitTypeE': noop,
      '_ZN6litert10GpuOptions32WaitForWeightsConversionCompleteEb': noop,
      '_ZN6litert10GpuOptions11SetPriorityENS0_8PriorityE': noop,
      '_ZN6litert10GpuOptions24SetPreferredDeviceSubstrEPKc': noop,
      '_ZN6litert10GpuOptions25DisableShaderOptimizationEb': noop,
      '_ZN6litert10GpuOptions38SetNumStepsOfCommandBufferPreparationsEi': noop,
      '_ZN6litert10GpuOptions21SetNumThreadsToUploadEi': noop,
      '_ZN6litert10GpuOptions22SetNumThreadsToCompileEi': noop,
      LrtGetOpaqueGpuOptionsData: zero,
      LrtGetOpaqueCompilerOptionsData: zero,
      LrtDestroyGpuOptions: noop,
      LrtDestroyCompilerOptions: noop,

      // Compiler options — not available on WASM
      '_ZN6litert15CompilerOptions6CreateEv': zero,

      // LiteRT internal — platform-unavailable features
      '_ZN6litert8internal14GpuEnvironmentD1Ev': noop,
      '_ZN6litert8internal14SerializeModelEO12LiteRtModelTm': zero,
      '_ZN6litert8internal14LoadBinaryFileENSt3__117basic_string_viewIcNS1_11char_traitsIcEEEE': zero,
      '_ZN6litert8internal20GetDispatchOpOptionsENS_9BufferRefIhEE': zero,
      '_ZN6litert8internal18CustomOpDispatcherC1ERKN14LiteRtOptionsT14CustomOpOptionE': noop,
      '_ZN6litert8internal18CustomOpDispatcherD1Ev': noop,
      '_ZN6litert8internal23DispatchDelegateOptions6CreateEv': zero,
      '_ZN6litert8internal23DispatchDelegateOptions12SetAllocBaseEPKv': noop,
      '_ZN6litert8internal23DispatchDelegateOptions14SetAllocBaseFdEi': noop,

      // Weight loader
      '_ZN13weight_loader24CreateLiteRtWeightLoaderEP20LiteRtRuntimeContextPKN6tflite5ModelENSt3__18optionalINS6_12basic_stringIcNS6_11char_traitsIcEENS6_9allocatorIcEEEEEENS6_10unique_ptrIN6litert18ScopedWeightSourceENS6_14default_deleteISH_EEEE': zero,

      // LiteRT-LM: constrained decoding (needs Rust llguidance)
      '_ZN6litert2lm24CreateConstraintProviderERKNSt3__17variantIJNS0_24ExternalConstraintConfigENS0_16LlGuidanceConfigENS0_9FstConfigEEEERKNS0_9TokenizerERKNS1_6vectorINSC_IiNS1_9allocatorIiEEEENSD_ISF_EEEE': zero,
      LiteRtLmGemmaModelConstraintProvider_Create: zero,
      LiteRtLmGemmaModelConstraintProvider_Destroy: noop,
      LiteRtLmGemmaModelConstraintProvider_CreateConstraintFromTools: zero,

      // LiteRT-LM: channel content, tool use, LoRA
      '_ZN6litert2lm21ExtractChannelContentERKNSt3__16vectorINS0_7ChannelENS1_9allocatorIS3_EEEERNS0_9ResponsesE': zero,
      '_ZN6litert2lm31InsertChannelContentIntoMessageERKN4absl12lts_2025081413flat_hash_mapINSt3__112basic_stringIcNS4_11char_traitsIcEENS4_9allocatorIcEEEESA_NS2_18container_internal10StringHashENSB_8StringEqENS8_INS4_4pairIKSA_SA_EEEEEERN8nlohmann16json_abi_v3_12_010basic_jsonINSM_11ordered_mapENS4_6vectorESA_bxydS8_NSM_14adl_serializerENSP_IhNS8_IhEEEEvEE': zero,
      '_ZN6litert2lm15FormatValueAsFcERKN8nlohmann16json_abi_v3_12_010basic_jsonINS2_11ordered_mapENSt3__16vectorENS5_12basic_stringIcNS5_11char_traitsIcEENS5_9allocatorIcEEEEbxydSA_NS2_14adl_serializerENS6_IhNSA_IhEEEEvEENS5_17basic_string_viewIcS9_EE': zero,
      '_ZN6litert2lm14FormatToolAsFcERKN8nlohmann16json_abi_v3_12_010basic_jsonINS2_11ordered_mapENSt3__16vectorENS5_12basic_stringIcNS5_11char_traitsIcEENS5_9allocatorIcEEEEbxydSA_NS2_14adl_serializerENS6_IhNSA_IhEEEEvEENS5_17basic_string_viewIcS9_EE': zero,
      '_ZN6litert2lm13GetSyntaxTypeENSt3__117basic_string_viewIcNS1_11char_traitsIcEEEE': zero,
      '_ZN6litert2lm21ParseTextAndToolCallsENSt3__117basic_string_viewIcNS1_11char_traitsIcEEEES5_S5_NS0_10SyntaxTypeEbS5_': zero,
      '_ZN6litert2lm8LoraData20CreateFromScopedFileENSt3__110shared_ptrIKNS_10ScopedFileEEE': zero,
      '_ZN6litert2lm15IsLoRAInputNameENSt3__117basic_string_viewIcNS1_11char_traitsIcEEEE': zero,

      // LiteRT-LM: Gemma data processors, model resources, logging
      '_ZN6litert2lm19Gemma3DataProcessor6CreateENS0_25Gemma3DataProcessorConfigENSt3__18optionalINS3_7variantIJNS0_11JsonPrefaceEEEEEEPKNS0_9TokenizerERKNS3_6vectorINSC_IiNS3_9allocatorIiEEEENSD_ISF_EEEEb': zero,
      '_ZN6litert2lm19Gemma4DataProcessor6CreateENS0_25Gemma4DataProcessorConfigENSt3__18optionalINS3_7variantIJNS0_11JsonPrefaceEEEEEEPKNS0_9TokenizerERKNS3_6vectorINSC_IiNS3_9allocatorIiEEEENSD_ISF_EEEEb': zero,
      '_ZN6litert2lm18ModelResourcesTask6CreateENSt3__110unique_ptrINS0_25ModelAssetBundleResourcesENS2_14default_deleteIS4_EEEE': zero,
      '_ZN6litert2lm9LogTensorERNS_12TensorBufferEmNSt3__117basic_string_viewIcNS3_11char_traitsIcEEEE': noop,
      '_ZN6litert2lmlsERNSt3__113basic_ostreamIcNS1_11char_traitsIcEEEERKNS_12TensorBufferE': zero,

      // LiteRT-LM: model loading (critical — needs VFS)
      '_ZN6litert2lm6schema22ReadHeaderFromLiteRTLMEPvmPNS1_14LitertlmHeaderE': zero,
      '_ZN6litert2lm6schema14DecompressDataEPKhmPNSt3__16vectorIhNS4_9allocatorIhEEEE': zero,
      '_ZN6litert2lm23ExtractFilesfromZipFileENSt3__117basic_string_viewIcNS1_11char_traitsIcEEEE': zero,

      // std::filesystem::path::__parent_path
      '_ZNKSt3__14__fs10filesystem4path13__parent_pathEv': zero,
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
      fd_close(fd: number): number { return vfs.fdClose(fd); },
      fd_fdstat_get: zero,
      fd_fdstat_set_flags: zero,
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
      poll_oneoff: zero,
      proc_exit(code: number): void { throw new Error(`WASM exit: ${code}`); },
      sched_yield: zero,
    },
  };

  const response = await fetch(WASM_URL);
  const { instance } = await WebAssembly.instantiateStreaming(response, imports);
  wasm = instance.exports as unknown as LiteRtExports;

  return wasm;
}

export function getWasm(): LiteRtExports {
  if (!wasm) throw new Error("LiteRT not initialized");
  return wasm;
}

export { writeString, readString, writeInputData };
