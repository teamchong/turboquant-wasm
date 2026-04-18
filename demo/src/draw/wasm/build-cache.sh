#!/bin/bash
# Hash-based WASM build cache. Skips rebuild if sources unchanged.
#
# Usage: ./build-cache.sh
#   - Computes SHA-256 of all Zig + C sources
#   - Compares with stored hash in .build-hash
#   - Skips rebuild if unchanged, otherwise builds + optimizes
#
# Force rebuild: rm wasm/.build-hash

set -e
cd "$(dirname "$0")"

HASH_FILE=".build-hash"
WASM_OUT="zig-out/bin/drawmode.wasm"

# Compute hash of all source files that affect the WASM output
compute_hash() {
  find src/ vendor/graphviz/ -type f \( -name "*.zig" -o -name "*.c" -o -name "*.h" \) \
    | sort \
    | xargs shasum -a 256 \
    | shasum -a 256 \
    | cut -d' ' -f1
}

CURRENT_HASH=$(compute_hash)

# Check if we can skip
if [ -f "$HASH_FILE" ] && [ -f "$WASM_OUT" ]; then
  STORED_HASH=$(cat "$HASH_FILE")
  if [ "$CURRENT_HASH" = "$STORED_HASH" ]; then
    echo "build-cache: sources unchanged, skipping WASM build"
    exit 0
  fi
fi

echo "build-cache: sources changed, rebuilding WASM..."
zig build
./optimize-wasm.sh

# Store hash after successful build
echo "$CURRENT_HASH" > "$HASH_FILE"
echo "build-cache: hash saved"
