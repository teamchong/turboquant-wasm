# TurboQuant WASM

[![CI](https://github.com/teamchong/turboquant-wasm/actions/workflows/ci.yml/badge.svg)](https://github.com/teamchong/turboquant-wasm/actions/workflows/ci.yml)
[![npm](https://img.shields.io/npm/v/turboquant-wasm)](https://www.npmjs.com/package/turboquant-wasm)
[![gzip size](https://img.shields.io/badge/gzip-~12kB-blue)](https://www.npmjs.com/package/turboquant-wasm)
[![license](https://img.shields.io/npm/l/turboquant-wasm)](https://github.com/teamchong/turboquant-wasm/blob/main/LICENSE)

Experimental WASM + relaxed SIMD build of [botirk38/turboquant](https://github.com/botirk38/turboquant) for browsers and Node.js.

Based on the paper ["TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"](https://arxiv.org/abs/2504.19874) (Google Research, ICLR 2026).

**[Live Demo](https://teamchong.github.io/turboquant-wasm/)** — vector search, image similarity, 3D Gaussian Splatting compression, **and a Gemma 4 E2B in-browser LLM whose KV cache is TurboQuant-compressed** — all in the browser.

## Two substrates, one algorithm

Most of this repo is **TurboQuant in WASM**: a Zig → WASM build with relaxed SIMD that encodes / decodes / scores vectors on the CPU. That's what the `turboquant-wasm` npm package ships — the vector-search, image-similarity, and 3DGS demos all load this module and call into it.

The **Prompt → Diagram demo is different**: it re-implements the same TurboQuant math (polar + QJL rotation) directly in WGSL compute shaders. An LLM KV cache is 140 layers × 35 heads × 30 tok/s worth of encode/decode/dot calls — only a GPU-native path hits real-time. So that demo is a showcase of the *algorithm*, not a consumer of the *package*.

Same math, two substrates: WASM for CPU vector-search workloads, WGSL for GPU LLM workloads. If you're looking for "what does TurboQuant compression look like on GPU", the draw demo's WGSL shaders in [`demo/src/draw/shaders/`](demo/src/draw/shaders/) are the reference.

## Why TurboQuant?

Float32 embedding indexes are large — 1M vectors × 384-dim = 1.5GB. They don't fit in mobile RAM, take minutes to download, and gzip only saves ~7% because float32 has high entropy.

TurboQuant compresses them 6x (1.5GB → 240MB) and searches directly on compressed data without decompressing. No training step — unlike PQ/OPQ, just `init({ dim, seed })` and encode any vector immediately.

## What this adds

- **npm package** with embedded WASM — `npm install turboquant-wasm`
- **WebGPU acceleration** — `dotBatch()` auto-detects WebGPU and dispatches a compute shader that scans compressed vectors directly on GPU, no decompression
- **CPU fallback** — transparent fallback to WASM relaxed SIMD when WebGPU is unavailable
- **Relaxed SIMD** — `@mulAdd` FMA maps to `f32x4.relaxed_madd`
- **SIMD-vectorized** QJL sign packing/unpacking and scaling
- **TypeScript API** — `TurboQuant.init()` / `encode()` / `decode()` / `dot()` / `dotBatch()`
- **Golden-value tests** — byte-identical output with the reference Zig implementation

## Browser Requirements

The WASM binary uses relaxed SIMD instructions:

| Runtime | Minimum Version |
|---------|----------------|
| Chrome | 114+ |
| Firefox | 128+ |
| Safari | 18+ |
| Node.js | 20+ |

## Quick Start

```ts
import { TurboQuant } from "turboquant-wasm";

const tq = await TurboQuant.init({ dim: 1024, seed: 42 });

// Compress a vector (~4.5 bits/dim, ~6x compression)
const compressed = tq.encode(myFloat32Array);

// Decode back
const decoded = tq.decode(compressed);

// Fast dot product without decoding
const score = tq.dot(queryVector, compressed);

// Batch search: auto-uses WebGPU when available, falls back to WASM SIMD
const allCompressed = new Uint8Array(/* concatenated compressed vectors */);
const scores = await tq.dotBatch(queryVector, allCompressed, bytesPerVector);

tq.destroy();
```

## API

```ts
class TurboQuant {
  static async init(config: { dim: number; seed: number }): Promise<TurboQuant>;
  encode(vector: Float32Array): Uint8Array;
  decode(compressed: Uint8Array): Float32Array;
  dot(query: Float32Array, compressed: Uint8Array): number;
  dotBatch(query: Float32Array, compressedConcat: Uint8Array, bytesPerVector: number): Promise<Float32Array>;
  rotateQuery(query: Float32Array): Float32Array;
  destroy(): void;
}
```

`dotBatch()` transparently uses WebGPU when available (Chrome 113+, Edge 113+). The GPU compute shader reads compressed data directly — no decompression step. Falls back to WASM SIMD on devices without WebGPU.

## Building

```bash
# Run tests
zig test -target aarch64-macos src/turboquant.zig

# Full npm build (zig -> wasm-opt -> base64 embed -> bun + tsc)
bun run build

# Build WASM only
bun run build:zig
```

Requires Zig 0.15.2 and Bun.

## Quality

Encoding preserves inner products — verified by golden-value tests and distortion bounds:

- **MSE** decreases with dimension (unit vectors)
- **Bits/dim** is ~4.5 (payload only, excluding 22-byte header)
- **Dot product preservation** — mean absolute error < 1.0 for unit vectors at dim=128
- **Bit-identical** output with [botirk38/turboquant](https://github.com/botirk38/turboquant) for same input + seed

## When to use TurboQuant (and when not to)

| | TurboQuant | PQ/OPQ (FAISS, ScaNN) |
|---|---|---|
| Compression | ~4.5 bits/dim (6x) | ~1-2 bits/dim (16-32x) |
| Query speed | Slower (float decode per pair) | Faster (integer codebook lookup) |
| Training | None — encode any vector immediately | Required — must train codebook on dataset |
| Streaming data | Yes — each vector is self-contained | Degrades if data distribution shifts |
| Setup | `npm install` + 3 lines of code | Dataset-dependent configuration |

**Use TurboQuant when:** vectors arrive continuously (LLM KV cache, real-time indexing), you can't pause to train, you need simple deployment (browser, edge), or you want a single npm package with no dependencies.

**Use PQ/OPQ when:** you have a static dataset, can afford a training step, and need maximum compression + fastest query speed. PQ is the better tool for traditional batch vector search.

## Credits

- [botirk38/turboquant](https://github.com/botirk38/turboquant) — original Zig implementation
- [TurboQuant paper](https://arxiv.org/abs/2504.19874) (Google Research, ICLR 2026) — algorithm design

## License

MIT
