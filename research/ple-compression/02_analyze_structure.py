#!/usr/bin/env python3
"""
Step 2: Read the extracted PLE tensor (from 01_extract_ple.py) and
produce the structural stats that decide compression strategy.

What we want to know:
- Per-layer value distribution: is each layer's 256-dim block roughly
  i.i.d. Gaussian, or heavy-tailed? (Affects whether uniform
  quantization or codebook is better.)
- Global rank: SVD on a subsample. If the 9.17M vectors live on a
  low-dim manifold, Tucker / low-rank beats codebook. If full-rank
  but clusterable, codebook wins.
- Cross-layer correlation: is PLE[layer=L][token=T] correlated with
  PLE[layer=L+1][token=T]? If yes, we can share codebooks / indices
  across layers.
- Sub-vector structure: split each 256-dim vector into N sub-vectors,
  measure per-sub-vector rank. Product quantization needs roughly
  uniform per-sub-vector variance.

Output: a markdown report to stdout + optional plots to ./out/.

Usage:
    python3 02_analyze_structure.py [--ple ple.f16.npy] [--plots]
"""

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ple", type=Path, default=Path("ple.f16.npy"))
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--subsample", type=int, default=50_000,
                        help="number of (token, layer) vectors to sample for SVD")
    args = parser.parse_args()

    print(f"[analyze] loading {args.ple}")
    ple = np.load(args.ple)
    print(f"[analyze] shape: {ple.shape} dtype: {ple.dtype}")

    VOCAB, FLAT = ple.shape
    assert FLAT == 35 * 256, f"expected inner dim 8960, got {FLAT}"
    NUM_LAYERS = 35
    DIM = 256

    # Reshape into per-(token, layer, dim) view: (vocab, layer, dim).
    ple3d = ple.reshape(VOCAB, NUM_LAYERS, DIM)
    print(f"[analyze] reshaped to (vocab, layers, dim) = {ple3d.shape}")

    # ------------------------------------------------------------------
    # Global value distribution
    # ------------------------------------------------------------------
    print("\n## global distribution")
    print(f"  min / max / mean / std: "
          f"{ple.min():.4f} / {ple.max():.4f} / {ple.mean():.4f} / {ple.std():.4f}")
    q = np.quantile(ple.astype(np.float32), [0.01, 0.25, 0.5, 0.75, 0.99])
    print(f"  quantiles (1/25/50/75/99): {q}")

    # ------------------------------------------------------------------
    # Per-layer stats: variance, sparsity
    # ------------------------------------------------------------------
    print("\n## per-layer statistics")
    print("  layer  std      mean     |max|    active%")
    for L in range(NUM_LAYERS):
        block = ple3d[:, L, :].astype(np.float32)
        s = block.std()
        m = block.mean()
        am = np.abs(block).max()
        active = (np.abs(block) > 1e-3).mean() * 100
        print(f"  {L:3d}    {s:.5f}  {m:+.5f}  {am:.4f}  {active:5.1f}%")

    # ------------------------------------------------------------------
    # Subsample + SVD: global rank
    # ------------------------------------------------------------------
    print(f"\n## SVD on {args.subsample} random (token, layer) vectors")
    # Treat each (token, layer) row as one vector of dim 256.
    all_vecs = ple3d.reshape(-1, DIM).astype(np.float32)  # (vocab*layers, 256)
    N = all_vecs.shape[0]
    idx = np.random.default_rng(42).choice(N, size=args.subsample, replace=False)
    sample = all_vecs[idx]
    print(f"  sample shape: {sample.shape}")
    _, S, _ = np.linalg.svd(sample, full_matrices=False)
    energy = (S ** 2).cumsum() / (S ** 2).sum()
    ranks = [np.searchsorted(energy, f) for f in (0.5, 0.75, 0.9, 0.95, 0.99)]
    print(f"  rank for 50%/75%/90%/95%/99% energy: {ranks}")
    print(f"  top 10 singular values: {S[:10]}")

    # ------------------------------------------------------------------
    # Per-layer rank (each layer's 256-dim space separately)
    # ------------------------------------------------------------------
    print("\n## per-layer rank (95% energy)")
    print("  layer  rank_95  top_sv   cond_number")
    for L in range(NUM_LAYERS):
        sub = ple3d[idx % VOCAB, L, :].astype(np.float32)
        _, Sl, _ = np.linalg.svd(sub, full_matrices=False)
        el = (Sl ** 2).cumsum() / (Sl ** 2).sum()
        rl = int(np.searchsorted(el, 0.95))
        cond = Sl[0] / max(Sl[-1], 1e-8)
        print(f"  {L:3d}    {rl:4d}     {Sl[0]:.3f}  {cond:.2e}")

    # ------------------------------------------------------------------
    # Cross-layer correlation: for a random set of tokens, measure how
    # much layer L looks like layer L+1 for the same token.
    # ------------------------------------------------------------------
    print("\n## cross-layer similarity (same token, adjacent layers)")
    sample_tokens = np.random.default_rng(1).choice(VOCAB, size=2000, replace=False)
    print("  pair  mean_cos    frac_within_0.1")
    for L in range(NUM_LAYERS - 1):
        a = ple3d[sample_tokens, L, :].astype(np.float32)
        b = ple3d[sample_tokens, L + 1, :].astype(np.float32)
        # Cosine similarity row-wise.
        na = np.linalg.norm(a, axis=1) + 1e-9
        nb = np.linalg.norm(b, axis=1) + 1e-9
        cos = (a * b).sum(axis=1) / (na * nb)
        print(f"  {L:2d}→{L+1:2d}  {cos.mean():+.4f}     {(np.abs(1 - cos) < 0.1).mean() * 100:.1f}%")

    # ------------------------------------------------------------------
    # Same-token-across-all-layers coherence: a strong result here would
    # say "store one vector per token + a per-layer delta instead of a
    # full 35-copy".
    # ------------------------------------------------------------------
    print("\n## per-token variance across layers (low → layers are ~copies)")
    # For N sampled tokens, compute std of each dim across layers, average.
    tokens = np.random.default_rng(2).choice(VOCAB, size=5000, replace=False)
    per_token_stack = ple3d[tokens].astype(np.float32)  # (5000, 35, 256)
    token_mean = per_token_stack.mean(axis=1)            # (5000, 256)
    token_std = per_token_stack.std(axis=1)              # (5000, 256)
    print(f"  mean across tokens of per-dim std-across-layers: {token_std.mean():.5f}")
    print(f"  mean across tokens of per-dim mean: {np.abs(token_mean).mean():.5f}")
    ratio = token_std.mean() / max(np.abs(token_mean).mean(), 1e-9)
    print(f"  ratio (std / |mean|): {ratio:.3f}")
    print("  interpretation: <<1 means layers are near-identical copies of a")
    print("                  per-token embedding; in that case a factored")
    print("                  storage (per-token base + small per-layer delta)")
    print("                  beats any codebook approach.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
