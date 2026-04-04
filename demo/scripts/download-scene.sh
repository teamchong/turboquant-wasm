#!/bin/bash
# Download a demo 3DGS scene (.ply with SH coefficients)
# Usage: bash scripts/download-scene.sh

set -e
cd "$(dirname "$0")/.."

DATA_DIR="data"
OUTPUT="$DATA_DIR/scene.ply"

if [ -f "$OUTPUT" ]; then
  echo "Scene already exists at $OUTPUT"
  echo "Delete it to re-download: rm $OUTPUT"
  exit 0
fi

mkdir -p "$DATA_DIR"

# Luma AI "garden" splat from the GaussianSplats3D demo collection
# This is a compact scene with SH coefficients suitable for the demo.
# Alternative: download from HuggingFace or train your own.
# Default: "train" scene from Tanks and Temples, ~668K Gaussians, 158MB
# Source: camenduru/gaussian-splatting on HuggingFace (no auth required)
SCENE_URL="${SCENE_URL:-https://huggingface.co/camenduru/gaussian-splatting/resolve/main/train/point_cloud/iteration_7000/point_cloud.ply}"

echo "Downloading scene from $SCENE_URL..."
curl -L -C - -o "$OUTPUT" "$SCENE_URL"

echo ""
echo "Downloaded to $OUTPUT"
ls -lh "$OUTPUT"
echo ""
echo "Next step: encode with TurboQuant"
echo "  bun run encode -- --input data/scene.ply --output data/scene.tqply"
