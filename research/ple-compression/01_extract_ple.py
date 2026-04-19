#!/usr/bin/env python3
"""
Step 1: Extract per_layer_token_embd from the Gemma 4 E2B GGUF file as
a float16 numpy array. Dequantizes from Q6_K (or whatever quant the
tensor is stored in) to f16 so the downstream analysis / codebook
training sees the baseline reconstruction values.

Usage:
    python3 01_extract_ple.py <path-to-gguf> [--out <path-to-npy>]

Default output path: ./ple.f16.npy

Expected size of output: 262144 tokens * 35 layers * 256 dims * 2 bytes
  = 4.5 GB. Larger than the source GGUF because we're storing at f16
  instead of the Q6_K on-disk form. Keep it on an SSD with headroom.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFReader


PLE_TENSOR_NAME = "per_layer_token_embd.weight"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf", type=Path, help="path to the GGUF file")
    parser.add_argument("--out", type=Path, default=Path("ple.f16.npy"))
    args = parser.parse_args()

    if not args.gguf.exists():
        print(f"no such file: {args.gguf}", file=sys.stderr)
        return 1

    print(f"[extract] opening {args.gguf} ({args.gguf.stat().st_size / 1e9:.2f} GB)")
    reader = GGUFReader(args.gguf)

    ple = None
    for t in reader.tensors:
        if t.name == PLE_TENSOR_NAME:
            ple = t
            break

    if ple is None:
        names = [t.name for t in reader.tensors]
        prefix = sorted({n.rsplit(".", 1)[0] for n in names})
        print(f"[extract] tensor {PLE_TENSOR_NAME!r} not found.", file=sys.stderr)
        print(f"[extract] tensor name prefixes: {prefix[:20]}", file=sys.stderr)
        return 2

    # GGUFReader.data is a memory-mapped view of the quantized bytes.
    # .shape is (num_elements_per_row, num_rows) in GGML convention.
    # For per_layer_token_embd we expect shape reported as (8960, 262144).
    print(f"[extract] tensor: {ple.name}")
    print(f"[extract]   shape (ggml convention): {list(ple.shape)}")
    print(f"[extract]   dtype: {ple.tensor_type.name} ({ple.tensor_type})")
    print(f"[extract]   raw bytes: {ple.data.nbytes / 1e6:.1f} MB")
    print(f"[extract]   n_elements: {ple.n_elements:,}")

    # Dequantize to f32 via gguf's built-in dequant (works for every K-quant).
    # Then cast to f16 to halve the on-disk size of the extracted copy.
    from gguf.quants import dequantize

    print(f"[extract] dequantizing to f32 ...")
    f32 = dequantize(ple.data, ple.tensor_type)
    print(f"[extract]   dequantized shape: {f32.shape} dtype={f32.dtype}")
    print(f"[extract]   min={f32.min():.6g} max={f32.max():.6g} mean={f32.mean():.6g} std={f32.std():.6g}")

    # Reshape to (vocab=262144, rows=35*256=8960). GGML stores
    # "row-major with the first dim as the innermost", which translates
    # to numpy shape (outer_dim, inner_dim) where outer = vocab_size.
    # Validate before casting.
    expected_cols = 35 * 256
    if f32.size != 262144 * expected_cols:
        print(
            f"[extract] unexpected size: got {f32.size}, "
            f"expected {262144 * expected_cols}",
            file=sys.stderr,
        )
        print(f"[extract]   actual shape from dequantize: {f32.shape}", file=sys.stderr)
    vocab_rows = f32.reshape(-1, expected_cols)
    print(f"[extract] reshaped to (vocab, layers*dim) = {vocab_rows.shape}")

    f16 = vocab_rows.astype(np.float16)
    print(f"[extract] casting to f16, saving to {args.out} ({f16.nbytes / 1e6:.1f} MB)")
    np.save(args.out, f16)
    print(f"[extract] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
