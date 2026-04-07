#!/bin/bash
# Build ORT + TurboQuant unified WASM binary.
# Compiles all ORT C++ with zig c++, links with Zig TQ code.
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ORT="$ROOT/vendor/onnxruntime"
SHIMS="$ROOT/vendor/ort-shims"
OUT="$ROOT/dist"
mkdir -p "$OUT" "$ROOT/.build-cache/obj"

# Compiler flags
INCLUDES=(
  -I "$SHIMS"
  -I "$ORT/include/onnxruntime"
  -I "$ORT/include"
  -I "$ORT/onnxruntime"
  -I "$ORT/include/onnxruntime/core/session"
  -I "$ORT/cmake/external/abseil-cpp"
  -I "$ORT/cmake/external/microsoft_gsl/include"
  -I "$ORT/cmake/external/flatbuffers/include"
  -I "$ORT/cmake/external/safeint"
  -I "$ORT/cmake/external/onnx"
  -I "$ORT/cmake/external/protobuf/src"
  -I "$ORT/cmake/external/mp11/include"
  -I "$ORT/cmake/external/date/include"
  -I "$ORT/cmake/external/json/include"
  -I "$ORT/cmake/external/eigen"
  -I "$ORT/onnxruntime/core/mlas/inc"
  -I "$ORT/onnxruntime/core/mlas/lib"
)

FORCE_INCLUDES=(
  -include "$SHIMS/fstream"
  -include "$SHIMS/mutex"
  -include "$SHIMS/shared_mutex"
  -include "$SHIMS/condition_variable"
  -include "$SHIMS/thread"
  -include "$SHIMS/wasm_compat.h"
  -include "$SHIMS/thread_stream.h"
)

DEFINES=(
  -DONNX_ML -DONNX_NAMESPACE=onnx
  -D__wasm__ -DORT_API_MANUAL_INIT -DDISABLE_FLOAT8_T
  -DUSE_JSEP -DMLAS_NO_ONNXRUNTIME_THREADPOOL
)

CXXFLAGS="-target wasm32-wasi -std=c++17 -O2 -fvisibility=default -Wno-deprecated-declarations"

# Collect ORT source files
echo "Collecting source files..."
ORT_SOURCES=()

# Core components
for dir in common framework graph session; do
  for f in "$ORT/onnxruntime/core/$dir"/*.cc; do
    [ -f "$f" ] || continue
    echo "$f" | grep -q "vitisai" && continue
    echo "$f" | grep -qi "test" && continue
    ORT_SOURCES+=("$f")
  done
done

# JSEP provider
for f in "$ORT/onnxruntime/core/providers/js"/*.cc; do
  [ -f "$f" ] || continue
  ORT_SOURCES+=("$f")
done

# JSEP operators
for f in "$ORT/onnxruntime/core/providers/js/operators"/*.cc; do
  [ -f "$f" ] || continue
  ORT_SOURCES+=("$f")
done

# WASM API
ORT_SOURCES+=("$ORT/onnxruntime/wasm/api.cc")

echo "  ${#ORT_SOURCES[@]} ORT source files"

# Compile ORT C++ to .o files (parallel)
echo "Compiling ORT C++..."
OBJECTS=()
FAILED=0
COMPILED=0

compile_one() {
  local src="$1"
  local base=$(basename "$src" .cc)
  base=$(basename "$base" .cpp)
  local obj="$ROOT/.build-cache/obj/${base}.o"

  # Skip if already compiled and newer than source
  if [ -f "$obj" ] && [ "$obj" -nt "$src" ]; then
    echo "$obj"
    return 0
  fi

  if zig c++ $CXXFLAGS \
    "${FORCE_INCLUDES[@]}" "${INCLUDES[@]}" "${DEFINES[@]}" \
    -c "$src" -o "$obj" 2>/dev/null; then
    echo "$obj"
    return 0
  else
    return 1
  fi
}

for src in "${ORT_SOURCES[@]}"; do
  obj=$(compile_one "$src")
  if [ $? -eq 0 ]; then
    OBJECTS+=("$obj")
    COMPILED=$((COMPILED + 1))
  else
    echo "  FAILED: $(basename "$src")"
    FAILED=$((FAILED + 1))
  fi
done

echo "  Compiled: $COMPILED, Failed: $FAILED"

if [ $FAILED -gt 0 ]; then
  echo "ERROR: $FAILED files failed to compile"
  exit 1
fi

echo "Build complete. ${#OBJECTS[@]} object files in .build-cache/obj/"
echo ""

echo "Compiling protobuf..."
PB_DIR="$ORT/cmake/external/protobuf/src/google/protobuf"
PB_COMPILED=0
PB_FILES=$(find "$PB_DIR" -name "*.cc" -not -name "*test*" -not -name "*mock*" -not -name "*compiler*" -not -path "*testing*" | sort)
for f in $PB_FILES; do
  [ -f "$f" ] || continue
  dir_part=$(basename $(dirname "$f"))
  base=$(basename "$f" .cc)
  obj="$ROOT/.build-cache/obj/pb_${dir_part}_${base}.o"
  if [ -f "$obj" ] && [ "$obj" -nt "$f" ]; then
    OBJECTS+=("$obj")
    PB_COMPILED=$((PB_COMPILED + 1))
    continue
  fi
  if zig c++ $CXXFLAGS "${FORCE_INCLUDES[@]}" "${INCLUDES[@]}" "${DEFINES[@]}" -c "$f" -o "$obj" 2>/dev/null; then
    OBJECTS+=("$obj")
    PB_COMPILED=$((PB_COMPILED + 1))
  fi
done
echo "  protobuf: $PB_COMPILED compiled"

echo "Compiling ONNX protobuf..."
ONNX_PB=0
for f in "$ORT/cmake/external/onnx/onnx"/*.pb.cc; do
  [ -f "$f" ] || continue
  base=$(basename "$f" .cc)
  obj="$ROOT/.build-cache/obj/onnx_${base}.o"
  if zig c++ $CXXFLAGS "${FORCE_INCLUDES[@]}" "${INCLUDES[@]}" "${DEFINES[@]}" -c "$f" -o "$obj" 2>/dev/null; then
    OBJECTS+=("$obj")
    ONNX_PB=$((ONNX_PB + 1))
  fi
done
echo "  onnx protobuf: $ONNX_PB compiled"

echo "Compiling abseil..."
AB_COMPILED=0
for dir in base base/internal strings strings/internal hash hash/internal container container/internal numeric types status; do
  for f in "$ORT/cmake/external/abseil-cpp/absl/$dir"/*.cc; do
    [ -f "$f" ] || continue
    echo "$f" | grep -qE "test|benchmark|_testing" && continue
    base=$(basename "$f" .cc)
    obj="$ROOT/.build-cache/obj/absl_${dir//\//_}_${base}.o"
    if [ -f "$obj" ] && [ "$obj" -nt "$f" ]; then
      OBJECTS+=("$obj")
      AB_COMPILED=$((AB_COMPILED + 1))
      continue
    fi
    if zig c++ $CXXFLAGS "${FORCE_INCLUDES[@]}" -I "$SHIMS" -I "$ORT/cmake/external/abseil-cpp" -D__wasm__ -DHAVE_PTHREAD=0 -c "$f" -o "$obj" 2>/dev/null; then
      OBJECTS+=("$obj")
      AB_COMPILED=$((AB_COMPILED + 1))
    fi
  done
done
echo "  abseil: $AB_COMPILED compiled"

echo ""
echo "Total objects: ${#OBJECTS[@]}"
echo "Build complete."

# Compile ONNX operator definitions
echo "Compiling ONNX defs..."
ONNX_COMPILED=0
for f in "$ORT/cmake/external/onnx/onnx/defs"/*.cc "$ORT/cmake/external/onnx/onnx/defs"/**/*.cc "$ORT/cmake/external/onnx/onnx"/*.cc "$ORT/cmake/external/onnx/onnx/shape_inference"/*.cc; do
  [ -f "$f" ] || continue
  echo "$f" | grep -qE "test|benchmark" && continue
  base=$(basename "$f" .cc)
  dir_part=$(basename $(dirname "$f"))
  obj="$ROOT/.build-cache/obj/onnxdefs_${dir_part}_${base}.o"
  if [ -f "$obj" ] && [ "$obj" -nt "$f" ]; then
    OBJECTS+=("$obj")
    ONNX_COMPILED=$((ONNX_COMPILED + 1))
    continue
  fi
  if zig c++ $CXXFLAGS "${FORCE_INCLUDES[@]}" "${INCLUDES[@]}" "${DEFINES[@]}" -c "$f" -o "$obj" 2>/dev/null; then
    OBJECTS+=("$obj")
    ONNX_COMPILED=$((ONNX_COMPILED + 1))
  fi
done
echo "  onnx defs: $ONNX_COMPILED compiled"

# Compile ORT optimizer
echo "Compiling optimizer..."
OPT_COMPILED=0
for f in "$ORT/onnxruntime/core/optimizer"/*.cc "$ORT/onnxruntime/core/optimizer"/**/*.cc; do
  [ -f "$f" ] || continue
  echo "$f" | grep -qE "test|benchmark" && continue
  base=$(basename "$f" .cc)
  dir_part=$(basename $(dirname "$f"))
  obj="$ROOT/.build-cache/obj/opt_${dir_part}_${base}.o"
  if [ -f "$obj" ] && [ "$obj" -nt "$f" ]; then
    OBJECTS+=("$obj")
    OPT_COMPILED=$((OPT_COMPILED + 1))
    continue
  fi
  if zig c++ $CXXFLAGS "${FORCE_INCLUDES[@]}" "${INCLUDES[@]}" "${DEFINES[@]}" -c "$f" -o "$obj" 2>/dev/null; then
    OBJECTS+=("$obj")
    OPT_COMPILED=$((OPT_COMPILED + 1))
  fi
done
echo "  optimizer: $OPT_COMPILED compiled"

# Compile CPU provider (needed for fallback ops)
echo "Compiling CPU provider..."
CPU_COMPILED=0
for f in "$ORT/onnxruntime/core/providers/cpu"/*.cc "$ORT/onnxruntime/core/providers/cpu"/**/*.cc; do
  [ -f "$f" ] || continue
  echo "$f" | grep -qE "test|benchmark" && continue
  base=$(basename "$f" .cc)
  dir_part=$(basename $(dirname "$f"))
  obj="$ROOT/.build-cache/obj/cpu_${dir_part}_${base}.o"
  if [ -f "$obj" ] && [ "$obj" -nt "$f" ]; then
    OBJECTS+=("$obj")
    CPU_COMPILED=$((CPU_COMPILED + 1))
    continue
  fi
  if zig c++ $CXXFLAGS "${FORCE_INCLUDES[@]}" "${INCLUDES[@]}" "${DEFINES[@]}" -c "$f" -o "$obj" 2>/dev/null; then
    OBJECTS+=("$obj")
    CPU_COMPILED=$((CPU_COMPILED + 1))
  fi
done
echo "  cpu provider: $CPU_COMPILED compiled"

# Compile MLAS (WASM-compatible files only)
echo "Compiling MLAS..."
MLAS_COMPILED=0
for f in "$ORT/onnxruntime/core/mlas/lib"/*.cpp; do
  [ -f "$f" ] || continue
  # Skip architecture-specific kernels
  echo "$f" | grep -qiE "neon|avx|sse|amx|lsx|kai_|test|benchmark" && continue
  base=$(basename "$f" .cpp)
  obj="$ROOT/.build-cache/obj/mlas_${base}.o"
  if [ -f "$obj" ] && [ "$obj" -nt "$f" ]; then
    OBJECTS+=("$obj")
    MLAS_COMPILED=$((MLAS_COMPILED + 1))
    continue
  fi
  if zig c++ $CXXFLAGS "${FORCE_INCLUDES[@]}" "${INCLUDES[@]}" "${DEFINES[@]}" -msimd128 -mrelaxed-simd -c "$f" -o "$obj" 2>/dev/null; then
    OBJECTS+=("$obj")
    MLAS_COMPILED=$((MLAS_COMPILED + 1))
  fi
done
# WASM-specific MLAS
for f in "$ORT/onnxruntime/core/mlas/lib/wasm"/*.cpp; do
  [ -f "$f" ] || continue
  base=$(basename "$f" .cpp)
  obj="$ROOT/.build-cache/obj/mlas_wasm_${base}.o"
  if zig c++ $CXXFLAGS "${FORCE_INCLUDES[@]}" "${INCLUDES[@]}" "${DEFINES[@]}" -msimd128 -mrelaxed-simd -c "$f" -o "$obj" 2>/dev/null; then
    OBJECTS+=("$obj")
    MLAS_COMPILED=$((MLAS_COMPILED + 1))
  fi
done
echo "  mlas: $MLAS_COMPILED compiled"

# Compile ORT contrib ops (JSEP needs them)
echo "Compiling contrib ops..."
CONTRIB_COMPILED=0
for f in "$ORT/onnxruntime/contrib_ops/js"/*.cc; do
  [ -f "$f" ] || continue
  base=$(basename "$f" .cc)
  obj="$ROOT/.build-cache/obj/contrib_${base}.o"
  if zig c++ $CXXFLAGS "${FORCE_INCLUDES[@]}" "${INCLUDES[@]}" "${DEFINES[@]}" -c "$f" -o "$obj" 2>/dev/null; then
    OBJECTS+=("$obj")
    CONTRIB_COMPILED=$((CONTRIB_COMPILED + 1))
  fi
done
echo "  contrib ops: $CONTRIB_COMPILED compiled"

echo ""
echo "=== FINAL TOTALS ==="
echo "Total objects: ${#OBJECTS[@]}"
du -sh "$ROOT/.build-cache/obj/"

# Compile CXA runtime
echo "Compiling CXA runtime..."
zig c++ -target wasm32-wasi -std=c++17 -O2 -c "$SHIMS/wasm_cxa.cc" -o "$ROOT/.build-cache/obj/wasm_cxa.o"
OBJECTS+=("$ROOT/.build-cache/obj/wasm_cxa.o")
echo "  cxa runtime compiled"

# Link
echo ""
echo "=== Linking ==="
zig c++ -target wasm32-wasi -O2 \
  -Wl,--no-entry -Wl,--export-dynamic \
  "${OBJECTS[@]}" \
  -o "$OUT/turboquant-llm.wasm" \
  -lc++ 2>&1 | grep "undefined symbol" | sed 's/.*undefined symbol: //' | sort -u | wc -l
echo "undefined symbols remaining (see above)"

if [ -f "$OUT/turboquant-llm.wasm" ]; then
  echo ""
  echo "SUCCESS: $(ls -lh "$OUT/turboquant-llm.wasm" | awk '{print $5}') WASM binary"
fi
