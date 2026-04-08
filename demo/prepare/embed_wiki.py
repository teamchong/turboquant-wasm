#!/usr/bin/env python3
"""
Fetch Wikipedia sentences and embed with all-MiniLM-L6-v2.
Much faster than Transformers.js — uses Python sentence-transformers.

Usage:
  python prepare/embed_wiki.py --count 50000 --out public/data/wiki-50k

Requires: pip install datasets sentence-transformers
Output: passages.json, embeddings.bin
"""

import argparse
import json
import struct
import os
import time
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50000)
    parser.add_argument("--out", type=str, default="public/data/wiki-50k")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    passages_path = os.path.join(args.out, "passages.json")
    emb_path = os.path.join(args.out, "embeddings.bin")

    # --- Fetch passages ---
    if os.path.exists(passages_path):
        with open(passages_path) as f:
            passages = json.load(f)
        if len(passages) >= args.count:
            print(f"Loaded cached {len(passages):,} passages from {passages_path}")
            passages = passages[:args.count]
        else:
            print(f"Cached {len(passages):,} < target {args.count:,}, re-fetching...")
            passages = None
    else:
        passages = None

    if passages is None:
        print(f"Fetching {args.count:,} Wikipedia sentences...")
        # Stream to avoid downloading entire dataset
        ds = load_dataset(
            "sentence-transformers/wikipedia-en-sentences",
            split="train",
            streaming=True,
        )
        passages = []
        for row in ds:
            s = row["sentence"]
            if s and 30 < len(s) < 500:
                passages.append(s)
            if len(passages) >= args.count:
                break
            if len(passages) % 10000 == 0 and len(passages) > 0:
                print(f"  {len(passages):,} / {args.count:,}")

        print(f"Got {len(passages):,} passages")
        with open(passages_path, "w") as f:
            json.dump(passages, f)
        print(f"Saved {passages_path}")

    # --- Embed ---
    if os.path.exists(emb_path):
        existing_count = os.path.getsize(emb_path) // (384 * 4)
        if existing_count >= len(passages):
            print(f"Embeddings already cached ({existing_count:,} vectors), skipping.")
            return
        print(f"Cached {existing_count:,} < {len(passages):,}, re-embedding...")

    print(f"Loading all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Embedding {len(passages):,} passages (batch_size={args.batch_size})...")
    start = time.time()
    embeddings = model.encode(
        passages,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    elapsed = time.time() - start
    print(f"Embedded in {elapsed:.1f}s ({len(passages) / elapsed:.0f} vec/s)")

    # Write as flat float32
    emb_f32 = embeddings.astype(np.float32)
    emb_f32.tofile(emb_path)
    size_mb = os.path.getsize(emb_path) / 1e6
    print(f"Saved {emb_path} ({len(passages):,} x 384, {size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
