#!/usr/bin/env python3
"""
Step 3: Train Product Quantization (PQ) codebooks on the extracted PLE
vectors. PQ was picked from the step-2 analysis:

- PLE vectors are full-rank (≥237 dims of 256 for 95% energy) → rules
  out low-rank methods.
- Cross-layer cosine ≈ 0 → rules out factored storage.
- Tight Gaussian distribution (std=0.064) → k-means finds clean
  clusters quickly.
- 9.17M vectors to compress → PQ trains in minutes, RVQ would need an
  iterative residual pipeline that's comparably slow.

PQ splits each 256-dim vector into M sub-vectors of dim=256/M and
fits an independent k-means codebook of K centroids on each sub-space.
Storage per vector: M × log2(K) bits. Reconstruction: concatenate
codebook[m][idx_m] across all M.

Configurations trained + evaluated for reconstruction MSE:
    M=16 × K=256      (128 bits/vec → ~147 MB)
    M=16 × K=16       (64  bits/vec → ~73  MB)
    M=8  × K=256      (64  bits/vec → ~73  MB; bigger sub-vec)
    M=32 × K=256      (256 bits/vec → ~294 MB; finer grid)
    M=64 × K=16       (256 bits/vec → ~294 MB)

The relative MSE printed here compares PQ reconstruction to the
already-dequantized f16 PLE tensor (i.e. against whatever Q5_K
produced as "ground truth" for the current model). A rel MSE of e.g.
0.01 means PQ recovers 99% of the per-element variance of that f16
reference. Whether that's acceptable is a downstream question —
validated in step 4 by comparing decoder logits end-to-end.

Usage:
    python3 03_train_pq.py [--ple ple.f16.npy] [--subsample 500000]

The subsample parameter controls how many training vectors we fit
each codebook on. k-means on ALL 9.17M vectors is possible but slow;
500k typically converges to within 0.5% of full-data reconstruction.
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans


@dataclass
class PQResult:
    name: str
    M: int  # number of sub-vectors
    K: int  # codebook size per sub-vector
    bits_per_vec: int
    fit_time_s: float
    codebooks: np.ndarray  # (M, K, sub_dim)
    mse: float             # mean squared error vs original
    per_elem_mse: float    # mse / mean(|vec|^2)

    def storage_mb(self, n_vectors: int) -> float:
        index_mb = n_vectors * self.bits_per_vec / 8 / 1e6
        codebook_mb = self.codebooks.nbytes / 1e6
        return index_mb + codebook_mb


def train_pq(
    vectors: np.ndarray,
    M: int,
    K: int,
    max_iter: int = 50,
    batch_size: int = 8192,
    random_state: int = 42,
) -> PQResult:
    """Fit M independent k-means codebooks, each K centroids, on
    M sub-vector slices of `vectors`.
    """
    assert vectors.ndim == 2
    N, D = vectors.shape
    assert D % M == 0, f"dim {D} not divisible by M={M}"
    sub_dim = D // M
    print(f"[pq] training M={M} K={K} sub_dim={sub_dim} on {N:,} vectors")

    codebooks = np.empty((M, K, sub_dim), dtype=np.float32)
    t0 = time.time()
    for m in range(M):
        slice_ = vectors[:, m * sub_dim:(m + 1) * sub_dim]
        km = MiniBatchKMeans(
            n_clusters=K,
            batch_size=batch_size,
            max_iter=max_iter,
            n_init=3,
            random_state=random_state + m,
            verbose=0,
        )
        km.fit(slice_.astype(np.float32))
        codebooks[m] = km.cluster_centers_
        if m == 0 or (m + 1) % max(1, M // 8) == 0:
            print(f"  sub {m+1}/{M} inertia={km.inertia_:.3f}")
    fit_s = time.time() - t0

    # Evaluate reconstruction MSE on the same sample.
    # (Full-corpus MSE would be a separate, bigger pass — done in step 4.)
    t0 = time.time()
    indices = np.empty((N, M), dtype=np.uint16)
    for m in range(M):
        slice_ = vectors[:, m * sub_dim:(m + 1) * sub_dim].astype(np.float32)
        # Argmin L2 distance to each centroid.
        # Use (x - c)^2 = x^2 - 2xc + c^2; drop constant x^2 term.
        cb = codebooks[m]
        cn = (cb ** 2).sum(axis=1)  # (K,)
        # Chunk to avoid NxK=500k*256 matrix all at once.
        CHUNK = 16384
        for start in range(0, N, CHUNK):
            end = min(start + CHUNK, N)
            xc = slice_[start:end] @ cb.T
            dist = -2 * xc + cn[None, :]
            indices[start:end, m] = dist.argmin(axis=1)

    # Reconstruct.
    recon = np.empty_like(vectors, dtype=np.float32)
    for m in range(M):
        recon[:, m * sub_dim:(m + 1) * sub_dim] = codebooks[m][indices[:, m]]

    orig = vectors.astype(np.float32)
    diff = orig - recon
    mse = (diff ** 2).mean()
    per_elem_mse = mse / (orig ** 2).mean()
    enc_s = time.time() - t0
    print(f"[pq]   fit={fit_s:.1f}s encode+recon={enc_s:.1f}s mse={mse:.6e} rel={per_elem_mse:.4f}")

    bits_per_vec = M * int(np.ceil(np.log2(K)))
    return PQResult(
        name=f"PQ-{M}x{K}",
        M=M, K=K, bits_per_vec=bits_per_vec,
        fit_time_s=fit_s,
        codebooks=codebooks,
        mse=mse,
        per_elem_mse=per_elem_mse,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ple", type=Path, default=Path("/tmp/ple-research/ple.f16.npy"))
    parser.add_argument("--subsample", type=int, default=500_000,
                        help="how many random (token, layer) vectors to train on")
    args = parser.parse_args()

    print(f"[pq] loading {args.ple}")
    ple = np.load(args.ple, mmap_mode="r")
    VOCAB, FLAT = ple.shape
    NUM_LAYERS, DIM = 35, 256
    assert FLAT == NUM_LAYERS * DIM

    # Subsample (token, layer) pairs, read each 256-dim slice from the
    # mmap'd PLE. Direct indexing of the mmap is much smaller than
    # pulling whole 8960-dim rows into memory for each pair.
    total = VOCAB * NUM_LAYERS
    rng = np.random.default_rng(0)
    idx = rng.choice(total, size=args.subsample, replace=False)
    idx.sort()

    vocab_idx = idx // NUM_LAYERS
    layer_idx = idx % NUM_LAYERS
    print(f"[pq] gathering {len(idx):,} vectors from mmap ...")
    t0 = time.time()
    vectors = np.empty((len(idx), DIM), dtype=np.float16)
    for i in range(len(idx)):
        v = int(vocab_idx[i])
        L = int(layer_idx[i])
        vectors[i] = ple[v, L * DIM:(L + 1) * DIM]
    print(f"[pq]   gathered in {time.time() - t0:.1f}s shape={vectors.shape}")
    print(f"[pq]   std={vectors.std():.5f} |max|={np.abs(vectors).max():.4f}")

    CONFIGS = [
        (16, 256),
        (16, 16),
        (8, 256),
        (32, 256),
        (64, 16),
        (4, 4096),
    ]

    results = []
    for M, K in CONFIGS:
        if DIM % M != 0:
            continue
        r = train_pq(vectors, M=M, K=K, max_iter=30, batch_size=8192)
        results.append(r)

    n_total = VOCAB * NUM_LAYERS
    print(f"\n## Summary (storage for full {n_total:,} PLE vectors)")
    print(f"  baseline Q5_K on-disk (current): 1615 MB")
    print(f"  config          bits/vec   MB       ratio    rel_MSE   fit_s")
    for r in results:
        mb = r.storage_mb(n_total)
        ratio = 1615 / mb
        print(f"  {r.name:14s}  {r.bits_per_vec:3d}        {mb:6.1f}   {ratio:5.1f}×   {r.per_elem_mse:.4f}  {r.fit_time_s:.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
