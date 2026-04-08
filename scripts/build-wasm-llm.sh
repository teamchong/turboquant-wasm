#!/bin/bash
# Build ORT + TurboQuant unified WASM binary.
# All C++ compiled with zig c++. Zero Emscripten.
#
# Output: dist/turboquant-llm.wasm (~14 MB optimized)
#
# Usage:
#   bash scripts/build-wasm-llm.sh          # full build
#   bash scripts/build-wasm-llm.sh --clean  # wipe cache and rebuild
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ORT="$ROOT/vendor/onnxruntime"
SHIMS="$ROOT/vendor/ort-shims"
PB_SRC="$ORT/cmake/external/protobuf/src"
ZIG_LIB="$(dirname $(which zig))/../lib"
OUT="$ROOT/dist"
CACHE="$ROOT/.build-cache"

if [ "$1" = "--clean" ]; then
  echo "Cleaning build cache..."
  rm -rf "$CACHE"
fi

mkdir -p "$OUT" "$CACHE/ort" "$CACHE/pb" "$CACHE/extra"

# --------------------------------------------------------------------------
# Compiler flags
# --------------------------------------------------------------------------
ORT_INCLUDES=(
  -I "$SHIMS"
  -I "$ORT/include/onnxruntime" -I "$ORT/include" -I "$ORT/onnxruntime"
  -I "$ORT/include/onnxruntime/core/session"
  -I "$ORT/cmake/external/abseil-cpp"
  -I "$ORT/cmake/external/microsoft_gsl/include"
  -I "$ORT/cmake/external/flatbuffers/include"
  -I "$ORT/cmake/external/safeint"
  -I "$ORT/cmake/external/onnx"
  -I "$PB_SRC"
  -I "$ORT/cmake/external/mp11/include"
  -I "$ORT/cmake/external/date/include"
  -I "$ORT/cmake/external/json/include"
  -I "$ORT/cmake/external/eigen"
  -I "$ORT/onnxruntime/core/mlas/inc"
  -I "$ORT/onnxruntime/core/mlas/lib"
)

ORT_FORCE_INCLUDES=(
  -include "$SHIMS/fstream"
  -include "$SHIMS/mutex"
  -include "$SHIMS/shared_mutex"
  -include "$SHIMS/condition_variable"
  -include "$SHIMS/thread"
  -include "$SHIMS/wasm_compat.h"
  -include "$SHIMS/thread_stream.h"
)

ORT_DEFINES=(
  -DONNX_ML -DONNX_NAMESPACE=onnx
  -D__wasm__ -DORT_API_MANUAL_INIT -DDISABLE_FLOAT8_T
  -DUSE_JSEP -DMLAS_NO_ONNXRUNTIME_THREADPOOL
)

CXX="zig c++ -target wasm32-wasi -std=c++17 -O2 -fvisibility=default -Wno-deprecated-declarations"
CC="zig cc -target wasm32-wasi -O2"

# --------------------------------------------------------------------------
# Compile function — path-based .o naming, no collisions
# --------------------------------------------------------------------------
# Usage: compile_cc <source.cc> <output_dir> <base_path> [extra_flags...]
# .o name = path relative to base_path with / replaced by __
compile_cc() {
  local src="$1" outdir="$2" base_path="$3"
  shift 3
  local rel="${src#$base_path/}"
  local oname="${rel//\//__}"
  oname="${oname%.cc}.o"
  oname="${oname%.cpp}.o"
  local obj="$outdir/$oname"

  if [ -f "$obj" ] && [ "$obj" -nt "$src" ]; then
    echo "$obj"
    return 0
  fi

  $CXX "$@" -c "$src" -o "$obj" 2>/dev/null || { rm -f "$obj"; return 1; }
  echo "$obj"
  return 0
}

# --------------------------------------------------------------------------
# 1. ORT core C++ sources
# --------------------------------------------------------------------------
echo "=== ORT core ==="
ORT_SOURCES=()

# Collect source files
for dir in \
  "$ORT/onnxruntime/core/common" \
  "$ORT/onnxruntime/core/common/logging" \
  "$ORT/onnxruntime/core/common/logging/sinks" \
  "$ORT/onnxruntime/core/framework" \
  "$ORT/onnxruntime/core/graph" \
  "$ORT/onnxruntime/core/graph/contrib_ops" \
  "$ORT/onnxruntime/core/flatbuffers" \
  "$ORT/onnxruntime/core/session" \
  "$ORT/onnxruntime/core/providers/js" \
  "$ORT/onnxruntime/core/providers/js/operators" \
  "$ORT/onnxruntime/core/optimizer" \
  "$ORT/onnxruntime/core/providers/cpu" \
  "$ORT/onnxruntime/core/platform" \
  "$ORT/onnxruntime/core/platform/logging" \
  "$ORT/onnxruntime/core/platform/posix"; do
  for f in "$dir"/*.cc; do
    [ -f "$f" ] || continue
    echo "$f" | grep -qiE "test|benchmark|vitisai" && continue
    ORT_SOURCES+=("$f")
  done
done

# Subdirectories of optimizer and cpu provider
for dir in $(find "$ORT/onnxruntime/core/optimizer" -mindepth 1 -type d 2>/dev/null); do
  for f in "$dir"/*.cc; do
    [ -f "$f" ] || continue
    echo "$f" | grep -qiE "test|benchmark" && continue
    ORT_SOURCES+=("$f")
  done
done
for dir in $(find "$ORT/onnxruntime/core/providers/cpu" -mindepth 1 -type d 2>/dev/null); do
  for f in "$dir"/*.cc; do
    [ -f "$f" ] || continue
    echo "$f" | grep -qiE "test|benchmark" && continue
    ORT_SOURCES+=("$f")
  done
done

# JS contrib ops
for dir in \
  "$ORT/onnxruntime/contrib_ops/js" \
  "$ORT/onnxruntime/contrib_ops/js/bert" \
  "$ORT/onnxruntime/contrib_ops/js/quantization"; do
  for f in "$dir"/*.cc; do
    [ -f "$f" ] || continue
    ORT_SOURCES+=("$f")
  done
done

# CPU contrib ops
for dir in \
  "$ORT/onnxruntime/contrib_ops/cpu" \
  "$ORT/onnxruntime/contrib_ops/cpu/bert" \
  "$ORT/onnxruntime/contrib_ops/cpu/quantization"; do
  for f in "$dir"/*.cc; do
    [ -f "$f" ] || continue
    echo "$f" | grep -qiE "test|benchmark" && continue
    ORT_SOURCES+=("$f")
  done
done

# WASM API
ORT_SOURCES+=("$ORT/onnxruntime/wasm/api.cc")

echo "  ${#ORT_SOURCES[@]} source files"

ORT_OBJECTS=()
compiled=0
failed=0
for src in "${ORT_SOURCES[@]}"; do
  if obj=$(compile_cc "$src" "$CACHE/ort" "$ORT" \
    "${ORT_FORCE_INCLUDES[@]}" "${ORT_INCLUDES[@]}" "${ORT_DEFINES[@]}"); then
    ORT_OBJECTS+=("$obj")
    compiled=$((compiled + 1))
  else
    echo "  FAIL: ${src#$ORT/}"
    failed=$((failed + 1))
  fi
done
echo "  Compiled: $compiled, Failed: $failed"

# --------------------------------------------------------------------------
# 2. ONNX defs + shape inference + checker
# --------------------------------------------------------------------------
echo ""
echo "=== ONNX ==="
ONNX_SOURCES=()
for dir in \
  "$ORT/cmake/external/onnx/onnx" \
  "$ORT/cmake/external/onnx/onnx/defs" \
  "$ORT/cmake/external/onnx/onnx/defs/math" \
  "$ORT/cmake/external/onnx/onnx/defs/tensor" \
  "$ORT/cmake/external/onnx/onnx/defs/nn" \
  "$ORT/cmake/external/onnx/onnx/defs/sequence" \
  "$ORT/cmake/external/onnx/onnx/defs/logical" \
  "$ORT/cmake/external/onnx/onnx/defs/reduction" \
  "$ORT/cmake/external/onnx/onnx/defs/traditionalml" \
  "$ORT/cmake/external/onnx/onnx/defs/object_detection" \
  "$ORT/cmake/external/onnx/onnx/defs/quantization" \
  "$ORT/cmake/external/onnx/onnx/defs/optional" \
  "$ORT/cmake/external/onnx/onnx/defs/generator" \
  "$ORT/cmake/external/onnx/onnx/defs/controlflow" \
  "$ORT/cmake/external/onnx/onnx/defs/experiments" \
  "$ORT/cmake/external/onnx/onnx/defs/rnn" \
  "$ORT/cmake/external/onnx/onnx/shape_inference" \
  "$ORT/cmake/external/onnx/onnx/common" \
  "$ORT/cmake/external/onnx/onnx/version_converter"; do
  for f in "$dir"/*.cc; do
    [ -f "$f" ] || continue
    echo "$f" | grep -qiE "test|benchmark" && continue
    ONNX_SOURCES+=("$f")
  done
done

ONNX_OBJECTS=()
compiled=0
for src in "${ONNX_SOURCES[@]}"; do
  # ONNX checker needs external data checks disabled (uses filesystem::is_symlink)
  extra_flags=()
  echo "$src" | grep -q "checker.cc" && extra_flags+=("-DONNX_DISABLE_EXTERNAL_DATA_CHECKS=1")

  if obj=$(compile_cc "$src" "$CACHE/ort" "$ORT/cmake/external" \
    "${ORT_FORCE_INCLUDES[@]}" "${ORT_INCLUDES[@]}" "${ORT_DEFINES[@]}" "${extra_flags[@]}"); then
    ONNX_OBJECTS+=("$obj")
    compiled=$((compiled + 1))
  fi
done
echo "  $compiled compiled"

# --------------------------------------------------------------------------
# 3. Protobuf (runtime only, no compiler)
# --------------------------------------------------------------------------
echo ""
echo "=== Protobuf ==="
PB_SOURCES=()
for dir in $(find "$PB_SRC/google/protobuf" -type d ! -path "*/compiler*" 2>/dev/null); do
  for f in "$dir"/*.cc; do
    [ -f "$f" ] || continue
    echo "$f" | grep -qiE "test|mock|unittest|_testing|benchmark" && continue
    PB_SOURCES+=("$f")
  done
done

PB_OBJECTS=()
compiled=0
for src in "${PB_SOURCES[@]}"; do
  # Protobuf only needs the mutex shim, not all ORT force-includes
  if obj=$(compile_cc "$src" "$CACHE/pb" "$PB_SRC" \
    -include "$SHIMS/mutex" -I "$PB_SRC" -DONNX_ML -DONNX_NAMESPACE=onnx); then
    PB_OBJECTS+=("$obj")
    compiled=$((compiled + 1))
  fi
done
echo "  $compiled compiled"

# --------------------------------------------------------------------------
# 4. Abseil
# --------------------------------------------------------------------------
echo ""
echo "=== Abseil ==="
AB_SOURCES=()
for dir in base base/internal strings strings/internal hash hash/internal \
  container container/internal numeric types status synchronization \
  synchronization/internal time time/internal debugging profiling/internal \
  random random/internal; do
  for f in "$ORT/cmake/external/abseil-cpp/absl/$dir"/*.cc; do
    [ -f "$f" ] || continue
    echo "$f" | grep -qiE "test|benchmark|_testing|print_hash_of|gentables" && continue
    AB_SOURCES+=("$f")
  done
done

AB_OBJECTS=()
compiled=0
for src in "${AB_SOURCES[@]}"; do
  if obj=$(compile_cc "$src" "$CACHE/ort" "$ORT/cmake/external" \
    "${ORT_FORCE_INCLUDES[@]}" -I "$SHIMS" -I "$ORT/cmake/external/abseil-cpp" \
    -D__wasm__ -DHAVE_PTHREAD=0); then
    AB_OBJECTS+=("$obj")
    compiled=$((compiled + 1))
  fi
done
echo "  $compiled compiled"

# --------------------------------------------------------------------------
# 5. MLAS (WASM-compatible only)
# --------------------------------------------------------------------------
echo ""
echo "=== MLAS ==="
MLAS_OBJECTS=()
compiled=0
for f in "$ORT/onnxruntime/core/mlas/lib"/*.cpp "$ORT/onnxruntime/core/mlas/lib/wasm"/*.cpp; do
  [ -f "$f" ] || continue
  echo "$f" | grep -qiE "neon|avx|sse|amx|lsx|kai_|test|benchmark|q4_dq_cli" && continue
  if obj=$(compile_cc "$f" "$CACHE/ort" "$ORT" \
    "${ORT_FORCE_INCLUDES[@]}" "${ORT_INCLUDES[@]}" "${ORT_DEFINES[@]}" \
    -msimd128 -mrelaxed-simd); then
    MLAS_OBJECTS+=("$obj")
    compiled=$((compiled + 1))
  fi
done
echo "  $compiled compiled"

# --------------------------------------------------------------------------
# 6. libc++ filesystem (path parsing for ORT model loading)
# --------------------------------------------------------------------------
echo ""
echo "=== libc++ filesystem ==="
FS_OBJECTS=()
for f in path; do
  src="$ZIG_LIB/libcxx/src/filesystem/${f}.cpp"
  [ -f "$src" ] || continue
  obj="$CACHE/extra/libcxx_fs_${f}.o"
  if [ ! -f "$obj" ] || [ "$src" -nt "$obj" ]; then
    $CXX -D_LIBCPP_HAS_FILESYSTEM=1 -D_LIBCPP_BUILDING_LIBRARY \
      -I "$ZIG_LIB/libcxx/include" -I "$ZIG_LIB/libcxx/src" \
      -c "$src" -o "$obj" 2>/dev/null
  fi
  [ -f "$obj" ] && FS_OBJECTS+=("$obj")
done
echo "  ${#FS_OBJECTS[@]} compiled"

# --------------------------------------------------------------------------
# 7. Platform shims (our implementations)
# --------------------------------------------------------------------------
echo ""
echo "=== Platform shims ==="
SHIM_OBJECTS=()

# C shims
$CC -c "$SHIMS/missing_syms.c" -o "$CACHE/extra/missing_syms.o"
SHIM_OBJECTS+=("$CACHE/extra/missing_syms.o")

# C++ shims (need ORT headers)
for f in "$SHIMS/wasm_env.cc" "$SHIMS/wasm_data_transfer.cc" "$SHIMS/missing_kernels.cc" "$SHIMS/wasm_cxa.cc"; do
  [ -f "$f" ] || continue
  base=$(basename "$f" .cc)
  obj="$CACHE/extra/${base}.o"
  $CXX "${ORT_FORCE_INCLUDES[@]}" "${ORT_INCLUDES[@]}" "${ORT_DEFINES[@]}" \
    -c "$f" -o "$obj" 2>/dev/null && SHIM_OBJECTS+=("$obj")
done
echo "  ${#SHIM_OBJECTS[@]} compiled"

# --------------------------------------------------------------------------
# 8. TQ bridge (Zig)
# --------------------------------------------------------------------------
echo ""
echo "=== TQ bridge ==="
TQ_OBJ="$CACHE/extra/tq_bridge.o"
zig build-obj -target wasm32-wasi -O ReleaseFast -fstrip "$ROOT/src/tq_bridge.zig" --name tq_bridge
mv tq_bridge.o "$TQ_OBJ"
echo "  compiled"

# --------------------------------------------------------------------------
# 9. Link
# --------------------------------------------------------------------------
echo ""
echo "=== Linking ==="

# System libraries from Zig's cache
echo "int main(){return 0;}" > /tmp/_zig_probe.c
ZIG_LIBS=$($CC /tmp/_zig_probe.c -o /tmp/_zig_probe.wasm -lc -v 2>&1 | grep -oE '/[^ ]+\.a' | tr '\n' ' ')
rm -f /tmp/_zig_probe.c /tmp/_zig_probe.wasm

echo "int main(){return 0;}" > /tmp/_zig_probe.cpp
ZIGCXX_LIBS=$(zig c++ -target wasm32-wasi /tmp/_zig_probe.cpp -o /tmp/_zig_probe.wasm -lc++ -v 2>&1 | grep -oE '/[^ ]+\.a' | tr '\n' ' ')
rm -f /tmp/_zig_probe.cpp /tmp/_zig_probe.wasm

# Allowed JS imports (JSEP, WASI, threading)
ALLOWED="$CACHE/allowed-imports.txt"
cat > "$ALLOWED" << 'IMPORTS'
jsep_alloc
jsep_free
jsep_create_kernel
jsep_release_kernel
jsep_run
jsep_capture_begin
jsep_capture_end
jsep_replay
__cxa_thread_atexit
pthread_self
pthread_getspecific
AbslInternalPerThreadSemPost_lts_20250814
AbslInternalPerThreadSemWait_lts_20250814
_ZN4absl12lts_2025081424synchronization_internal20CreateThreadIdentityEv
_ZN7OrtApis17CreateLoraAdapterEPKcP12OrtAllocatorPP14OrtLoraAdapter
_ZN7OrtApis26CreateLoraAdapterFromArrayEPKvmP12OrtAllocatorPP14OrtLoraAdapter
_ZN7OrtApis18ReleaseLoraAdapterEP14OrtLoraAdapter
_ZN11onnxruntime25IExecutionProviderFactory14CreateProviderERK17OrtSessionOptionsRK9OrtLogger
_ZTIN11onnxruntime25IExecutionProviderFactoryE
IMPORTS

ALL_OBJECTS=(
  "${SHIM_OBJECTS[@]}"
  "$TQ_OBJ"
  "${FS_OBJECTS[@]}"
  "${ORT_OBJECTS[@]}"
  "${ONNX_OBJECTS[@]}"
  "${PB_OBJECTS[@]}"
  "${AB_OBJECTS[@]}"
  "${MLAS_OBJECTS[@]}"
)

echo "  ${#ALL_OBJECTS[@]} total objects"

wasm-ld --no-entry --export-dynamic \
  --allow-undefined-file="$ALLOWED" \
  --error-limit=0 \
  "${ALL_OBJECTS[@]}" \
  $ZIGCXX_LIBS \
  -o "$OUT/turboquant-llm-raw.wasm" 2>&1 | tee /tmp/link-errors.txt | grep "error:" | head -5

UNDEF=$(grep "undefined symbol" /tmp/link-errors.txt | sed 's/.*undefined symbol: //' | sort -u | wc -l | tr -d ' ')
DUPS=$(grep "duplicate symbol" /tmp/link-errors.txt | wc -l | tr -d ' ')

echo "  Undefined: $UNDEF, Duplicates: $DUPS"

if [ -f "$OUT/turboquant-llm-raw.wasm" ]; then
  RAW_SIZE=$(ls -lh "$OUT/turboquant-llm-raw.wasm" | awk '{print $5}')
  echo "  Raw: $RAW_SIZE"

  echo "  Optimizing..."
  wasm-opt -O3 --strip-debug "$OUT/turboquant-llm-raw.wasm" -o "$OUT/turboquant-llm.wasm"
  rm "$OUT/turboquant-llm-raw.wasm"

  OPT_SIZE=$(ls -lh "$OUT/turboquant-llm.wasm" | awk '{print $5}')
  IMPORTS=$(wasm-objdump -j Import -x "$OUT/turboquant-llm.wasm" 2>/dev/null | grep "func\[" | wc -l | tr -d ' ')
  echo ""
  echo "=== SUCCESS ==="
  echo "  $OUT/turboquant-llm.wasm"
  echo "  Size: $OPT_SIZE (optimized)"
  echo "  Imports: $IMPORTS"
else
  echo ""
  echo "=== LINK FAILED ==="
  echo "  $UNDEF undefined symbols, $DUPS duplicate symbols"
  echo "  See /tmp/link-errors.txt"
  exit 1
fi
