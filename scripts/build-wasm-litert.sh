#!/bin/bash
# Build LiteRT + LiteRT-LM + TurboQuant unified WASM binary.
# All C/C++ compiled with zig c++/cc. Zero Emscripten.
#
# Output: dist/turboquant-litert.wasm
#
# Usage:
#   bash scripts/build-wasm-litert.sh          # full build
#   bash scripts/build-wasm-litert.sh --clean  # wipe cache and rebuild
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LITERT="$ROOT/vendor/litert"
LITERT_LM="$ROOT/vendor/litert-lm"
XNNPACK="$ROOT/vendor/xnnpack"
SHIMS="$ROOT/vendor/ort-shims"
OUT="$ROOT/dist"
CACHE="$ROOT/.build-cache-litert"

if [ "$1" = "--clean" ]; then
  echo "Cleaning build cache..."
  rm -rf "$CACHE"
fi

mkdir -p "$OUT" "$CACHE/tflite" "$CACHE/xnnpack" "$CACHE/litert-lm" "$CACHE/deps" "$CACHE/extra"

# --------------------------------------------------------------------------
# Compiler flags
# --------------------------------------------------------------------------
CXX="zig c++ -target wasm32-wasi -std=c++17 -Os -fvisibility=default -Wno-deprecated-declarations"
CC="zig cc -target wasm32-wasi -Os -fvisibility=default"

ORT_EXT="$ROOT/vendor/onnxruntime/cmake/external"
TFLITE_INCLUDES=(
  -I "$SHIMS"
  -I "$LITERT"
  -I "$LITERT/tflite"
  -I "$LITERT/tflite/kernels"
  -I "$ROOT/vendor/flatbuffers/include"
  -I "$ORT_EXT/abseil-cpp"
  -I "$ROOT/vendor/eigen"
  -I "$ROOT/vendor/gemmlowp"
  -I "$ROOT/vendor/ruy"
  -I "$ROOT/vendor/fp16/include"
  -I "$ROOT/vendor/cpuinfo/include"
  -I "$ROOT/vendor/pthreadpool/include"
  -I "$XNNPACK"
  -I "$XNNPACK/include"
  -I "$XNNPACK/src"
)

TFLITE_DEFINES=(
  -D__wasm__
  -D_WASI_EMULATED_MMAN
  -DTFLITE_ENABLE_XNNPACK=1
  -DTFLITE_ENABLE_MMAP=0
  -DEIGEN_DONT_PARALLELIZE
  -DEIGEN_MPL2_ONLY
  -DTFL_STATIC_LIBRARY_BUILD
  -DXNN_ENABLE_ASSEMBLY=0
  -DXNN_ENABLE_MEMOPT=1
  -DXNN_ENABLE_SPARSE=1
  -DXNN_ENABLE_WASM_REVECTORIZE=0
  -DPTHREADPOOL_NO_DEPRECATED_API=1
  -DXNN_LOG_LEVEL=0
)

TFLITE_FORCE_INCLUDES=(
  -include "$SHIMS/mutex"
  -include "$SHIMS/shared_mutex"
  -include "$SHIMS/condition_variable"
  -include "$SHIMS/thread"
  -include "$SHIMS/wasm_compat.h"
)

# --------------------------------------------------------------------------
# Compile functions — path-based .o naming, no collisions
# --------------------------------------------------------------------------
compile_cc() {
  local src="$1" outdir="$2" base_path="$3"
  shift 3
  local rel="${src#$base_path/}"
  local oname="${rel//\//__}"
  oname="${oname%.cc}"
  oname="${oname%.cpp}"
  oname="${oname%.c}"
  oname="${oname}.o"
  local obj="$outdir/$oname"
  if [ -f "$obj" ] && [ "$obj" -nt "$src" ]; then
    echo "$obj"
    return 0
  fi
  $CXX "$@" -c "$src" -o "$obj" 2>/dev/null || { rm -f "$obj"; return 1; }
  echo "$obj"
  return 0
}

compile_c() {
  local src="$1" outdir="$2" base_path="$3"
  shift 3
  local rel="${src#$base_path/}"
  local oname="${rel//\//__}"
  oname="${oname%.c}"
  oname="${oname}.o"
  local obj="$outdir/$oname"
  if [ -f "$obj" ] && [ "$obj" -nt "$src" ]; then
    echo "$obj"
    return 0
  fi
  $CC "$@" -c "$src" -o "$obj" 2>/dev/null || { rm -f "$obj"; return 1; }
  echo "$obj"
  return 0
}

# --------------------------------------------------------------------------
# 1. TFLite core
# --------------------------------------------------------------------------
echo "=== TFLite core ==="
TFLITE_SOURCES=()

for dir in \
  "$LITERT/tflite" \
  "$LITERT/tflite/core" \
  "$LITERT/tflite/core/api" \
  "$LITERT/tflite/core/async" \
  "$LITERT/tflite/core/async/c" \
  "$LITERT/tflite/core/async/interop" \
  "$LITERT/tflite/core/async/interop/c" \
  "$LITERT/tflite/core/c" \
  "$LITERT/tflite/core/kernels" \
  "$LITERT/tflite/core/tools" \
  "$LITERT/tflite/c" \
  "$LITERT/tflite/delegates" \
  "$LITERT/tflite/delegates/xnnpack" \
  "$LITERT/tflite/experimental/resource" \
  "$LITERT/tflite/experimental/genai" \
  "$LITERT/tflite/kernels" \
  "$LITERT/tflite/kernels/internal" \
  "$LITERT/tflite/kernels/internal/reference" \
  "$LITERT/tflite/profiling" \
  "$LITERT/tflite/schema"; do
  for f in "$dir"/*.cc; do
    [ -f "$f" ] || continue
    echo "$f" | grep -qiE "test|benchmark|_main\.|tflite_with_xnnpack|with_selected_ops|tensorflow_profiler" && continue
    echo "$f" | grep -qiE "minimal_logging_android|minimal_logging_ios" && continue
    # Skip files needing XLA/TSL headers or POSIX (audio/FFT, Eigen conv, file I/O — not needed for LLM)
    echo "$f" | grep -qiE "audio_spectrogram|irfft2d|rfft2d|spectrogram|eigen_support|lsh_projection" && continue
    echo "$f" | grep -qiE "file_util\.cc|mmap_handle" && continue
    # conv.cc needs XLA/TSL eigen_spatial_convolutions (XNNPack handles conv at runtime)
    basename "$f" | grep -qE "^conv\.cc$" && continue
    # Skip profiling files that need RE2 regex
    echo "$f" | grep -qiE "profile_summarizer|profile_summary_formatter|model_runtime_info" && continue
    # Skip serialization (needs filesystem)
    echo "$f" | grep -qiE "delegates/serialization|genai_ops_wrapper|variable_ops_wrapper" && continue
    # Skip random_ops (needs platform RNG)
    echo "$f" | grep -qiE "random_ops" && continue
    TFLITE_SOURCES+=("$f")
  done
done

echo "  ${#TFLITE_SOURCES[@]} source files"

TFLITE_OBJECTS=()
compiled=0
failed=0
for src in "${TFLITE_SOURCES[@]}"; do
  if obj=$(compile_cc "$src" "$CACHE/tflite" "$LITERT" \
    "${TFLITE_FORCE_INCLUDES[@]}" "${TFLITE_INCLUDES[@]}" "${TFLITE_DEFINES[@]}"); then
    TFLITE_OBJECTS+=("$obj")
    compiled=$((compiled + 1))
  else
    echo "  FAIL: ${src#$LITERT/}"
    failed=$((failed + 1))
  fi
done
echo "  Compiled: $compiled, Failed: $failed"

# --------------------------------------------------------------------------
# 2. XNNPack (WASM SIMD production kernels + core)
# --------------------------------------------------------------------------
echo ""
echo "=== XNNPack ==="
XNNPACK_SOURCES=()

# Core sources (operators, configs, tables, subgraph)
for dir in \
  "$XNNPACK/src/configs" \
  "$XNNPACK/src/enums" \
  "$XNNPACK/src/operators" \
  "$XNNPACK/src/subgraph" \
  "$XNNPACK/src/tables"; do
  for f in "$dir"/*.c; do
    [ -f "$f" ] || continue
    XNNPACK_SOURCES+=("$f")
  done
done

# Production WASM SIMD microkernels
while IFS= read -r line; do
  f="$XNNPACK/$line"
  [ -f "$f" ] && XNNPACK_SOURCES+=("$f")
done < <(sed -n '/^SET(PROD_WASMSIMD_MICROKERNEL_SRCS/,/^SET(NON_PROD/p' \
  "$XNNPACK/cmake/gen/wasmsimd_microkernels.cmake" | grep "\.c" | sed 's/[[:space:]]*//' | sed 's/)$//')

# Production WASM relaxed SIMD microkernels
while IFS= read -r line; do
  f="$XNNPACK/$line"
  [ -f "$f" ] && XNNPACK_SOURCES+=("$f")
done < <(sed -n '/^SET(PROD_WASMRELAXEDSIMD_MICROKERNEL_SRCS/,/^SET(NON_PROD/p' \
  "$XNNPACK/cmake/gen/wasmrelaxedsimd_microkernels.cmake" | grep "\.c" | sed 's/[[:space:]]*//' | sed 's/)$//')

# Production scalar microkernels (fallback for ops without WASM SIMD variants)
while IFS= read -r line; do
  f="$XNNPACK/$line"
  [ -f "$f" ] && XNNPACK_SOURCES+=("$f")
done < <(sed -n '/^SET(PROD_SCALAR_MICROKERNEL_SRCS/,/^SET(NON_PROD/p' \
  "$XNNPACK/cmake/gen/scalar_microkernels.cmake" | grep "\.c" | sed 's/[[:space:]]*//' | sed 's/)$//')

# XNNPack runtime sources
for f in "$XNNPACK/src"/*.c; do
  [ -f "$f" ] || continue
  XNNPACK_SOURCES+=("$f")
done

echo "  ${#XNNPACK_SOURCES[@]} source files"

XNNPACK_OBJECTS=()
compiled=0
failed=0
for src in "${XNNPACK_SOURCES[@]}"; do
  if obj=$(compile_c "$src" "$CACHE/xnnpack" "$XNNPACK" \
    "${TFLITE_INCLUDES[@]}" "${TFLITE_DEFINES[@]}" \
    -I "$SHIMS" -I "$XNNPACK" -I "$XNNPACK/src" \
    -msimd128 -mrelaxed-simd -fno-lax-vector-conversions); then
    XNNPACK_OBJECTS+=("$obj")
    compiled=$((compiled + 1))
  else
    echo "  FAIL: ${src#$XNNPACK/}"
    failed=$((failed + 1))
  fi
done
echo "  Compiled: $compiled, Failed: $failed"

# --------------------------------------------------------------------------
# 3. Dependencies (pthreadpool, cpuinfo, farmhash, ruy)
# --------------------------------------------------------------------------
echo ""
echo "=== Dependencies ==="
DEP_OBJECTS=()

# pthreadpool (single-threaded WASM — portable-api + shim + memory)
for f in "$ROOT/vendor/pthreadpool/src/portable-api.c" \
         "$ROOT/vendor/pthreadpool/src/shim.c" \
         "$ROOT/vendor/pthreadpool/src/memory.c"; do
  [ -f "$f" ] || continue
  base=$(basename "$f" .c)
  obj="$CACHE/deps/pthreadpool_${base}.o"
  $CC -I "$ROOT/vendor/pthreadpool/include" -I "$ROOT/vendor/pthreadpool/src" \
    -I "$ROOT/vendor/cpuinfo/include" \
    -DPTHREADPOOL_NO_DEPRECATED_API=1 \
    -c "$f" -o "$obj" 2>/dev/null && DEP_OBJECTS+=("$obj")
done

# farmhash
if [ -f "$ROOT/vendor/farmhash/src/farmhash.cc" ]; then
  obj="$CACHE/deps/farmhash.o"
  $CXX -I "$ROOT/vendor/farmhash/src" \
    -c "$ROOT/vendor/farmhash/src/farmhash.cc" -o "$obj" 2>/dev/null && DEP_OBJECTS+=("$obj")
fi

# cpuinfo (WASM variant)
for f in "$ROOT/vendor/cpuinfo/src/init.c" \
         "$ROOT/vendor/cpuinfo/src/api.c"; do
  [ -f "$f" ] || continue
  base=$(basename "$f" .c)
  obj="$CACHE/deps/cpuinfo_${base}.o"
  $CC -I "$ROOT/vendor/cpuinfo/include" -I "$ROOT/vendor/cpuinfo/src" \
    -D__wasm__ \
    -c "$f" -o "$obj" 2>/dev/null && DEP_OBJECTS+=("$obj")
done

# ruy (matrix multiplication library used by TFLite quantized kernels)
echo "  Compiling ruy..."
RUY="$ROOT/vendor/ruy"
RUY_OBJECTS=()
for f in "$RUY/ruy/allocator.cc" \
         "$RUY/ruy/apply_multiplier.cc" \
         "$RUY/ruy/block_map.cc" \
         "$RUY/ruy/blocking_counter.cc" \
         "$RUY/ruy/context.cc" \
         "$RUY/ruy/context_get_ctx.cc" \
         "$RUY/ruy/cpuinfo.cc" \
         "$RUY/ruy/ctx.cc" \
         "$RUY/ruy/denormal.cc" \
         "$RUY/ruy/frontend.cc" \
         "$RUY/ruy/pmu.cc" \
         "$RUY/ruy/prepacked_cache.cc" \
         "$RUY/ruy/prepare_packed_matrices.cc" \
         "$RUY/ruy/system_aligned_alloc.cc" \
         "$RUY/ruy/thread_pool.cc" \
         "$RUY/ruy/trmul.cc" \
         "$RUY/ruy/tune.cc" \
         "$RUY/ruy/wait.cc"; do
  [ -f "$f" ] || continue
  base=$(basename "$f" .cc)
  obj="$CACHE/deps/ruy_${base}.o"
  if [ ! -f "$obj" ] || [ "$f" -nt "$obj" ]; then
    $CXX "${TFLITE_FORCE_INCLUDES[@]}" \
      -I "$RUY" -I "$ROOT/vendor/cpuinfo/include" \
      "${TFLITE_DEFINES[@]}" \
      -DRUY_PLATFORM_DETECTED=1 -DRUY_DONOTUSEDIRECTLY_ARCH_WASM=1 \
      -c "$f" -o "$obj" 2>/dev/null && RUY_OBJECTS+=("$obj") || echo "  ruy FAIL: $(basename $f)"
  else
    RUY_OBJECTS+=("$obj")
  fi
done
DEP_OBJECTS+=("${RUY_OBJECTS[@]}")
echo "  ruy: ${#RUY_OBJECTS[@]} compiled"

# Abseil containers (hash tables used by TFLite op registry)
echo "  Compiling abseil..."
ABSL="$ORT_EXT/abseil-cpp"
ABSL_OBJECTS=()
for f in "$ABSL/absl/container/internal/raw_hash_set.cc" \
         "$ABSL/absl/container/internal/hashtablez_sampler.cc" \
         "$ABSL/absl/container/internal/hashtablez_sampler_force_weak_definition.cc"; do
  [ -f "$f" ] || continue
  base=$(basename "$f" .cc)
  obj="$CACHE/deps/absl_${base}.o"
  if [ ! -f "$obj" ] || [ "$f" -nt "$obj" ]; then
    $CXX "${TFLITE_FORCE_INCLUDES[@]}" \
      -I "$ABSL" "${TFLITE_DEFINES[@]}" \
      -c "$f" -o "$obj" 2>/dev/null && ABSL_OBJECTS+=("$obj") || echo "  absl FAIL: $(basename $f)"
  else
    ABSL_OBJECTS+=("$obj")
  fi
done
DEP_OBJECTS+=("${ABSL_OBJECTS[@]}")
echo "  abseil: ${#ABSL_OBJECTS[@]} compiled"

# TFLite schema_utils (GetBuiltinCode)
echo "  Compiling schema_utils..."
SCHEMA_SRC="$LITERT/tflite/converter/schema/schema_utils.cc"
if [ -f "$SCHEMA_SRC" ]; then
  obj="$CACHE/deps/schema_utils.o"
  $CXX "${TFLITE_FORCE_INCLUDES[@]}" "${TFLITE_INCLUDES[@]}" "${TFLITE_DEFINES[@]}" \
    -I "$LITERT" \
    -c "$SCHEMA_SRC" -o "$obj" 2>/dev/null && DEP_OBJECTS+=("$obj") || echo "  FAIL: schema_utils"
fi

# XNNPack fingerprint_check
echo "  Compiling xnn_fingerprint_check..."
FP_SRC="$XNNPACK/src/xnnpack/fingerprint_check.c"
if [ -f "$FP_SRC" ]; then
  obj="$CACHE/deps/xnn_fingerprint_check.o"
  $CC "${TFLITE_INCLUDES[@]}" "${TFLITE_DEFINES[@]}" \
    -I "$XNNPACK" -I "$XNNPACK/include" -I "$XNNPACK/src" \
    -msimd128 -mrelaxed-simd \
    -c "$FP_SRC" -o "$obj" 2>/dev/null && DEP_OBJECTS+=("$obj") || echo "  FAIL: fingerprint_check"
fi

# XNNPack reference C++ sources (packing, binary/unary elementwise, pack-lh)
echo "  Compiling XNNPack reference sources..."
XNN_REF_OBJECTS=()
for f in "$XNNPACK/src/reference/packing.cc" \
         "$XNNPACK/src/reference/binary-elementwise.cc" \
         "$XNNPACK/src/reference/unary-elementwise.cc" \
         "$XNNPACK/src/pack-lh.cc"; do
  [ -f "$f" ] || continue
  base=$(basename "$f" .cc)
  obj="$CACHE/deps/xnn_ref_${base}.o"
  if [ ! -f "$obj" ] || [ "$f" -nt "$obj" ]; then
    $CXX "${TFLITE_FORCE_INCLUDES[@]}" "${TFLITE_INCLUDES[@]}" "${TFLITE_DEFINES[@]}" \
      -I "$XNNPACK" -I "$XNNPACK/include" -I "$XNNPACK/src" \
      -msimd128 -mrelaxed-simd \
      -c "$f" -o "$obj" 2>/dev/null && XNN_REF_OBJECTS+=("$obj") || echo "  xnn_ref FAIL: $(basename $f)"
  else
    XNN_REF_OBJECTS+=("$obj")
  fi
done
DEP_OBJECTS+=("${XNN_REF_OBJECTS[@]}")
echo "  xnn reference: ${#XNN_REF_OBJECTS[@]} compiled"

# TFLite XNNPack file I/O and mmap (compiled with no-mmap debug path for WASM)
echo "  Compiling XNNPack file_util and mmap_handle..."
for f in "$LITERT/tflite/delegates/xnnpack/file_util.cc" \
         "$LITERT/tflite/delegates/xnnpack/mmap_handle.cc"; do
  [ -f "$f" ] || continue
  base=$(basename "$f" .cc)
  obj="$CACHE/deps/xnnpack_${base}.o"
  if [ ! -f "$obj" ] || [ "$f" -nt "$obj" ]; then
    $CXX "${TFLITE_FORCE_INCLUDES[@]}" "${TFLITE_INCLUDES[@]}" "${TFLITE_DEFINES[@]}" \
      -DXNNPACK_CACHE_NO_FILE_MAPPING_FOR_DEBUG=1 \
      -c "$f" -o "$obj" 2>/dev/null && DEP_OBJECTS+=("$obj") || echo "  FAIL: $(basename $f)"
  else
    DEP_OBJECTS+=("$obj")
  fi
done

echo "  ${#DEP_OBJECTS[@]} total dep objects"

# --------------------------------------------------------------------------
# 4. Platform implementations (WASM shims)
# --------------------------------------------------------------------------
echo ""
echo "=== Platform implementations ==="
SHIM_OBJECTS=()

# C implementations (pthread, abseil LowLevelAlloc, madvise, cxa_throw)
$CC -c "$SHIMS/missing_syms.c" -o "$CACHE/extra/missing_syms.o"
SHIM_OBJECTS+=("$CACHE/extra/missing_syms.o")

# C++ implementations (TFLite ops, telemetry, ErrorReporter, kernel registrations, etc.)
$CXX "${TFLITE_FORCE_INCLUDES[@]}" "${TFLITE_INCLUDES[@]}" "${TFLITE_DEFINES[@]}" \
  -c "$SHIMS/tflite_wasm_impls.cc" -o "$CACHE/extra/tflite_wasm_impls.o"
SHIM_OBJECTS+=("$CACHE/extra/tflite_wasm_impls.o")

# Abseil WASM implementations (Mutex, Notification, Duration, RawLog — for ruy)
$CXX "${TFLITE_FORCE_INCLUDES[@]}" "${TFLITE_DEFINES[@]}" \
  -c "$SHIMS/abseil_wasm.cc" -o "$CACHE/extra/abseil_wasm.o"
SHIM_OBJECTS+=("$CACHE/extra/abseil_wasm.o")

echo "  ${#SHIM_OBJECTS[@]} compiled"

# --------------------------------------------------------------------------
# 5. Protobuf runtime (v33 — matches generated .pb.cc files)
# --------------------------------------------------------------------------
echo ""
echo "=== Protobuf runtime ==="
PROTOBUF="$ROOT/vendor/protobuf"
PB_OBJECTS=()
mkdir -p "$CACHE/protobuf"

PB_INCLUDES=(
  -I "$PROTOBUF/src"
  -I "$PROTOBUF/third_party/utf8_range"
  -I "$ORT_EXT/abseil-cpp"
)
PB_DEFINES=(
  -DHAVE_ZLIB=0
  -DPROTOBUF_USE_DLLS=0
  -DABSL_MIN_LOG_LEVEL=4
)

# Core protobuf sources and subdirectories (non-compiler, non-test)
for subdir in "" "/io"; do
  for f in "$PROTOBUF/src/google/protobuf${subdir}"/*.cc; do
    [ -f "$f" ] || continue
    echo "$f" | grep -qiE "test|unittest|mock|compiler|_test\." && continue
    rel="${f#$PROTOBUF/}"
    oname="${rel//\//__}"
    oname="${oname%.cc}.o"
    obj="$CACHE/protobuf/$oname"
    if [ ! -f "$obj" ] || [ "$f" -nt "$obj" ]; then
      $CXX "${TFLITE_FORCE_INCLUDES[@]}" "${PB_INCLUDES[@]}" "${PB_DEFINES[@]}" "${TFLITE_DEFINES[@]}" \
        -c "$f" -o "$obj" 2>/dev/null && PB_OBJECTS+=("$obj") || echo "  FAIL: ${f#$PROTOBUF/}"
    else
      PB_OBJECTS+=("$obj")
    fi
  done
done

# UTF8 range library (required by protobuf v33+)
for f in "$PROTOBUF/third_party/utf8_range"/*.c; do
  [ -f "$f" ] || continue
  base=$(basename "$f" .c)
  obj="$CACHE/protobuf/utf8_${base}.o"
  if [ ! -f "$obj" ] || [ "$f" -nt "$obj" ]; then
    $CC "${PB_INCLUDES[@]}" -c "$f" -o "$obj" 2>/dev/null && PB_OBJECTS+=("$obj") || echo "  FAIL: utf8/$(basename $f)"
  else
    PB_OBJECTS+=("$obj")
  fi
done

echo "  ${#PB_OBJECTS[@]} compiled"

# --------------------------------------------------------------------------
# 6. SentencePiece tokenizer
# --------------------------------------------------------------------------
echo ""
echo "=== SentencePiece ==="
SP="$ROOT/vendor/sentencepiece"
SP_OBJECTS=()
mkdir -p "$CACHE/sentencepiece"

SP_INCLUDES=(
  -I "$SP/src"
  -I "$SP/src/builtin_pb"
  -I "$SP"
  -I "$SP/third_party"
  -I "$PROTOBUF/src"
)

for f in "$SP/src"/*.cc; do
  [ -f "$f" ] || continue
  echo "$f" | grep -qiE "test|_main\.|trainer" && continue
  base=$(basename "$f" .cc)
  obj="$CACHE/sentencepiece/sp_${base}.o"
  if [ ! -f "$obj" ] || [ "$f" -nt "$obj" ]; then
    $CXX "${TFLITE_FORCE_INCLUDES[@]}" "${SP_INCLUDES[@]}" "${PB_INCLUDES[@]}" "${PB_DEFINES[@]}" "${TFLITE_DEFINES[@]}" \
      -c "$f" -o "$obj" 2>/dev/null && SP_OBJECTS+=("$obj") || echo "  FAIL: $(basename $f)"
  else
    SP_OBJECTS+=("$obj")
  fi
done

echo "  ${#SP_OBJECTS[@]} compiled"

# --------------------------------------------------------------------------
# 7. LiteRT-LM runtime
# --------------------------------------------------------------------------
echo ""
echo "=== LiteRT-LM runtime ==="
LM="$ROOT/vendor/litert-lm"
LM_OBJECTS=()
mkdir -p "$CACHE/litert-lm"

LM_INCLUDES=(
  -I "$LM"
  -I "$LM/runtime"
  -I "$LITERT"
  -I "$SHIMS"
  "${TFLITE_INCLUDES[@]}"
  "${PB_INCLUDES[@]}"
  -I "$SP/src"
  -I "$ROOT/vendor/nlohmann-json/include"
)

LM_DEFINES=(
  "${TFLITE_DEFINES[@]}"
  "${PB_DEFINES[@]}"
  -DENABLE_SENTENCEPIECE_TOKENIZER=1
  -DLITERTLM_USE_STD_THREAD=1
)

# Skip list: files that need unavailable platform features
LM_SKIP_PATTERNS="test|benchmark|_main\.|worker_thread_pthread|memory_mapped_file_win|audio_preprocessor_miniaudio|npu_compiled_model|huggingface_tokenizer|gemma_model_constraint_provider\.cc"

LM_SOURCES=()
for dir in \
  "$LM/c" \
  "$LM/runtime/components" \
  "$LM/runtime/components/constrained_decoding" \
  "$LM/runtime/components/embedding_lookup" \
  "$LM/runtime/components/preprocessor" \
  "$LM/runtime/components/tool_use" \
  "$LM/runtime/conversation" \
  "$LM/runtime/conversation/model_data_processor" \
  "$LM/runtime/core" \
  "$LM/runtime/engine" \
  "$LM/runtime/executor" \
  "$LM/runtime/executor/litert" \
  "$LM/runtime/framework" \
  "$LM/runtime/framework/resource_management" \
  "$LM/runtime/framework/resource_management/utils" \
  "$LM/runtime/util"; do
  for f in "$dir"/*.cc; do
    [ -f "$f" ] || continue
    echo "$f" | grep -qiE "$LM_SKIP_PATTERNS" && continue
    LM_SOURCES+=("$f")
  done
done

# Generated proto sources
for f in "$LM/runtime/proto"/*.pb.cc "$LM/runtime/util"/*.pb.cc "$LM/runtime/executor/proto"/*.pb.cc; do
  [ -f "$f" ] && LM_SOURCES+=("$f")
done

echo "  ${#LM_SOURCES[@]} source files"

compiled=0
failed=0
for src in "${LM_SOURCES[@]}"; do
  rel="${src#$LM/}"
  oname="${rel//\//__}"
  oname="${oname%.cc}.o"
  obj="$CACHE/litert-lm/$oname"
  if [ -f "$obj" ] && [ "$obj" -nt "$src" ]; then
    LM_OBJECTS+=("$obj")
    compiled=$((compiled + 1))
    continue
  fi
  if $CXX "${TFLITE_FORCE_INCLUDES[@]}" "${LM_INCLUDES[@]}" "${LM_DEFINES[@]}" \
    -c "$src" -o "$obj" 2>/dev/null; then
    LM_OBJECTS+=("$obj")
    compiled=$((compiled + 1))
  else
    echo "  FAIL: ${src#$LM/}"
    failed=$((failed + 1))
  fi
done
echo "  Compiled: $compiled, Failed: $failed"

# --------------------------------------------------------------------------
# 8. TQ bridge (Zig)
# --------------------------------------------------------------------------
echo ""
echo "=== TQ bridge ==="
TQ_OBJ="$CACHE/extra/tq_bridge.o"
zig build-obj -target wasm32-wasi -O ReleaseSmall -fstrip "$ROOT/src/tq_bridge.zig" --name tq_bridge 2>/dev/null
mv tq_bridge.o "$TQ_OBJ" 2>/dev/null || true
echo "  compiled"

# --------------------------------------------------------------------------
# 9. Link
# --------------------------------------------------------------------------
echo ""
echo "=== Linking ==="

# System libraries from Zig
ZIGCXX_LIBS=$(echo "int main(){return 0;}" > /tmp/_p.cpp && zig c++ -target wasm32-wasi /tmp/_p.cpp -o /tmp/_p.wasm -lc++ -v 2>&1 | grep -oE '/[^ ]+\.a' | tr '\n' ' ' && rm -f /tmp/_p.cpp /tmp/_p.wasm)

ALL_OBJECTS=(
  "${SHIM_OBJECTS[@]}"
  "$TQ_OBJ"
  "${TFLITE_OBJECTS[@]}"
  "${XNNPACK_OBJECTS[@]}"
  "${DEP_OBJECTS[@]}"
  "${PB_OBJECTS[@]}"
  "${SP_OBJECTS[@]}"
  "${LM_OBJECTS[@]}"
)

echo "  ${#ALL_OBJECTS[@]} total objects"

wasm-ld --no-entry --export-dynamic \
  --export=wasm_malloc --export=wasm_free \
  --error-limit=0 \
  "${ALL_OBJECTS[@]}" \
  $ZIGCXX_LIBS \
  -o "$OUT/turboquant-litert-raw.wasm" 2>&1 | tee /tmp/litert-link-errors.txt | grep "error:" | head -5

UNDEF=$(grep "undefined symbol" /tmp/litert-link-errors.txt | sed 's/.*undefined symbol: //' | sort -u | wc -l | tr -d ' ')
DUPS=$(grep "duplicate symbol" /tmp/litert-link-errors.txt | wc -l | tr -d ' ')

echo "  Undefined: $UNDEF, Duplicates: $DUPS"

if [ -f "$OUT/turboquant-litert-raw.wasm" ]; then
  RAW_SIZE=$(ls -lh "$OUT/turboquant-litert-raw.wasm" | awk '{print $5}')
  echo "  Raw: $RAW_SIZE"

  echo "  Optimizing..."
  wasm-opt -Oz --strip-debug --strip-producers --low-memory-unused \
    --skip-pass=remove-unused-module-elements \
    "$OUT/turboquant-litert-raw.wasm" -o "$OUT/turboquant-litert.wasm"
  rm "$OUT/turboquant-litert-raw.wasm"

  OPT_SIZE=$(ls -lh "$OUT/turboquant-litert.wasm" | awk '{print $5}')
  echo ""
  echo "=== SUCCESS ==="
  echo "  $OUT/turboquant-litert.wasm"
  echo "  Size: $OPT_SIZE (optimized)"
else
  echo ""
  echo "=== LINK FAILED ==="
  echo "  $UNDEF undefined symbols, $DUPS duplicate symbols"
  echo "  See /tmp/litert-link-errors.txt"
  exit 1
fi
