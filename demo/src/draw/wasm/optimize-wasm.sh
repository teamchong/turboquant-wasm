#!/bin/bash
# Post-build WASM optimization using Binaryen tools.
# Reduces binary size by ~13% (629KB → 546KB).
#
# Usage: ./optimize-wasm.sh [input.wasm] [output.wasm]

set -e

INPUT="${1:-zig-out/bin/drawmode.wasm}"
OUTPUT="${2:-$INPUT}"

if ! command -v wasm-opt &>/dev/null; then
  echo "wasm-opt not found — skipping optimization (install binaryen)"
  exit 0
fi

if [ ! -f "$INPUT" ]; then
  echo "WASM file not found: $INPUT"
  exit 1
fi

BEFORE=$(wc -c < "$INPUT")

# Binaryen optimization: -Oz for maximum size reduction.
# Feature flags match what Zig emits for wasm32-wasi with musl libc:
#   bulk-memory: memcpy/memset intrinsics
#   nontrapping-float-to-int: trunc_sat instructions (Graphviz uses extensively)
#   sign-ext: sign-extension operators
wasm-opt -Oz \
  --enable-bulk-memory \
  --enable-nontrapping-float-to-int \
  --enable-sign-ext \
  "$INPUT" -o "$OUTPUT"

AFTER=$(wc -c < "$OUTPUT")
echo "wasm-opt: ${BEFORE} → ${AFTER} bytes ($(( (BEFORE - AFTER) * 100 / BEFORE ))% reduction)"
