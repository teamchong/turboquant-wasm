# TurboQuant 3DGS Demo

Side-by-side comparison of original 3D Gaussian Splatting scene vs TurboQuant-compressed version.

## Quick Start

```bash
cd demo

# Install dependencies
bun install

# Download demo scene (~158 MB .ply from HuggingFace)
bash scripts/download-scene.sh

# Compress with TurboQuant
bun run encode -- --input data/scene.ply --output data/scene.tqply

# Start dev server
bun run dev
```

Open http://localhost:5173/ — you'll see side-by-side viewers with synced cameras.

## How It Works

1. **Offline**: The encoder CLI reads a 3DGS `.ply` file, extracts the 24 spherical harmonic (SH) coefficients per Gaussian, and compresses them with TurboQuant (~2x compression on SH data).

2. **Browser**: The viewer fetches the `.tqply` file, decompresses SH coefficients in real-time using `turboquant-wasm` (WASM + relaxed SIMD), reconstructs a valid PLY blob, and renders it with GaussianSplats3D.

3. **Result**: Visually identical quality at 2.4x smaller file size (158 MB → 65 MB).

## Custom Scenes

Use any 3DGS `.ply` file with SH coefficients:

```bash
bun run encode -- --input path/to/your/scene.ply --output data/scene.tqply --sh-degree 2 --seed 42
```

Or load via URL params: `?ply=URL&tqply=URL`

## Requirements

- Browser with WASM relaxed SIMD: Chrome 114+, Firefox 128+, Safari 18+
- Bun (for encoder CLI)
