#!/bin/bash
# Fetch ORT's FetchContent dependencies at their pinned versions.
# Run once after cloning, or after updating the ORT submodule.
set -e

ORT_DIR="$(cd "$(dirname "$0")/../vendor/onnxruntime" && pwd)"
EXT="$ORT_DIR/cmake/external"

fetch() {
  local name=$1 url=$2 sha=$3
  if [ -d "$EXT/$name" ] && [ -f "$EXT/$name/.fetched" ]; then
    echo "  $name: cached"
    return
  fi
  echo "  $name: fetching..."
  rm -rf "$EXT/$name"
  local tmp="/tmp/ort-dep-$name.zip"
  curl -sL "$url" -o "$tmp"
  local actual_sha=$(shasum -a 1 "$tmp" | cut -d' ' -f1)
  if [ "$actual_sha" != "$sha" ]; then
    echo "  ERROR: SHA1 mismatch for $name (expected $sha, got $actual_sha)"
    exit 1
  fi
  unzip -qo "$tmp" -d "/tmp/ort-dep-$name-extract"
  mv "/tmp/ort-dep-$name-extract"/* "$EXT/$name"
  rm -rf "/tmp/ort-dep-$name-extract" "$tmp"
  touch "$EXT/$name/.fetched"
  echo "  $name: ok"
}

echo "Fetching ORT dependencies..."

fetch abseil-cpp \
  "https://github.com/abseil/abseil-cpp/archive/refs/tags/20250814.0.zip" \
  "a9eb1d648cbca4d4d788737e971a6a7a63726b07"

fetch protobuf \
  "https://github.com/protocolbuffers/protobuf/archive/refs/tags/v21.12.zip" \
  "7cf2733949036c7d52fda017badcab093fe73bfa"

fetch flatbuffers \
  "https://github.com/google/flatbuffers/archive/refs/tags/v23.5.26.zip" \
  "59422c3b5e573dd192fead2834d25951f1c1670c"

fetch date \
  "https://github.com/HowardHinnant/date/archive/refs/tags/v3.0.1.zip" \
  "2dac0c81dc54ebdd8f8d073a75c053b04b56e159"

fetch json \
  "https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.zip" \
  "5e88795165cc8590138d1f47ce94ee567b85b4d6"

fetch mp11 \
  "https://github.com/boostorg/mp11/archive/refs/tags/boost-1.82.0.zip" \
  "9bc9e01dffb64d9e0773b2e44d2f22c51aace063"

fetch eigen \
  "https://github.com/eigen-mirror/eigen/archive/1d8b82b0740839c0de7f1242a3585e3390ff5f33/eigen-1d8b82b0740839c0de7f1242a3585e3390ff5f33.zip" \
  "05b19b49e6fbb91246be711d801160528c135e34"

fetch safeint \
  "https://github.com/dcleblanc/SafeInt/archive/refs/tags/3.0.28.zip" \
  "23f252040ff6cb9f1fd18575b32fa8fb5928daac"

fetch microsoft_gsl \
  "https://github.com/microsoft/GSL/archive/refs/tags/v4.0.0.zip" \
  "cf368104cd22a87b4dd0c80228919bb2df3e2a14"

echo ""
echo "Pre-generating protobuf headers..."
PROTOC_URL="https://github.com/protocolbuffers/protobuf/releases/download/v21.12/protoc-21.12-osx-universal_binary.zip"
if [ "$(uname)" = "Linux" ]; then
  PROTOC_URL="https://github.com/protocolbuffers/protobuf/releases/download/v21.12/protoc-21.12-linux-x86_64.zip"
fi
PROTOC_DIR="/tmp/protoc-21.12"
if [ ! -f "$PROTOC_DIR/bin/protoc" ]; then
  curl -sL "$PROTOC_URL" -o /tmp/protoc.zip
  unzip -qo /tmp/protoc.zip -d "$PROTOC_DIR"
  rm /tmp/protoc.zip
fi
cd "$EXT/onnx"
"$PROTOC_DIR/bin/protoc" -I. --cpp_out=. onnx/onnx-ml.proto onnx/onnx-operators-ml.proto onnx/onnx-data.proto
echo "  onnx protobuf headers generated"

echo ""
echo "Applying WASM patches..."
cd "$ORT_DIR"
git apply "$(cd "$(dirname "$0")/.." && pwd)/vendor/ort-patches/wasm-compat.patch" 2>/dev/null || echo "  patches already applied"

echo ""
echo "Done. All ORT dependencies ready."
