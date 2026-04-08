#!/usr/bin/env node
/**
 * Generate text embeddings using all-MiniLM-L6-v2 via Transformers.js.
 * Fetches Wikipedia sentences via HuggingFace Datasets API and embeds them.
 *
 * Usage:
 *   node prepare/embed.mjs                     # 5K vectors → data/
 *   node prepare/embed.mjs --count 50000 --out public/data/wiki-50k
 *
 * Output: passages.json, embeddings.bin
 */

import { pipeline } from "@huggingface/transformers";
import { writeFileSync, mkdirSync, existsSync, readFileSync } from "fs";

const args = process.argv.slice(2);
function getArg(name, fallback) {
  const idx = args.indexOf(name);
  return idx >= 0 && args[idx + 1] ? args[idx + 1] : fallback;
}

const NUM_PASSAGES = parseInt(getArg("--count", "5000"));
const DATA_DIR = getArg("--out", "data");
const BATCH_SIZE = 64;
const DIM = 384;

async function fetchWithRetry(url, maxRetries = 5) {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    const resp = await fetch(url);
    if (resp.ok) return resp;
    if (resp.status === 429) {
      const wait = Math.min(2 ** attempt * 1000, 30000);
      console.log(`  Rate limited, waiting ${(wait / 1000).toFixed(0)}s...`);
      await new Promise((r) => setTimeout(r, wait));
      continue;
    }
    throw new Error(`HuggingFace API error: ${resp.status} ${resp.statusText}`);
  }
  throw new Error("Max retries exceeded");
}

async function fetchPassages() {
  const passagesPath = `${DATA_DIR}/passages.json`;
  if (existsSync(passagesPath)) {
    const cached = JSON.parse(readFileSync(passagesPath, "utf-8"));
    if (cached.length >= NUM_PASSAGES) {
      console.log(`Loading cached passages from ${passagesPath} (${cached.length.toLocaleString()})`);
      return cached.slice(0, NUM_PASSAGES);
    }
    console.log(`Cached passages (${cached.length}) < target (${NUM_PASSAGES}), re-fetching...`);
  }

  console.log(
    `Fetching ${NUM_PASSAGES.toLocaleString()} Wikipedia sentences via HuggingFace Datasets API...`,
  );

  const passages = [];
  const batchSize = 100;
  let offset = 0;

  while (passages.length < NUM_PASSAGES) {
    const url = `https://datasets-server.huggingface.co/rows?dataset=sentence-transformers/wikipedia-en-sentences&config=default&split=train&offset=${offset}&length=${batchSize}`;
    const resp = await fetchWithRetry(url);
    const data = await resp.json();
    if (!data.rows || data.rows.length === 0) break;

    for (const row of data.rows) {
      const sentence = row.row.sentence;
      if (sentence && sentence.length > 30 && sentence.length < 500) {
        passages.push(sentence);
      }
      if (passages.length >= NUM_PASSAGES) break;
    }

    offset += batchSize;
    if (passages.length % 1000 === 0) {
      console.log(
        `  Fetched ${passages.length.toLocaleString()} / ${NUM_PASSAGES.toLocaleString()} passages`,
      );
    }
  }

  console.log(`Got ${passages.length.toLocaleString()} passages`);
  return passages;
}

async function main() {
  mkdirSync(DATA_DIR, { recursive: true });

  const passages = await fetchPassages();
  console.log(
    `Embedding ${passages.length.toLocaleString()} passages with all-MiniLM-L6-v2...`,
  );

  writeFileSync(`${DATA_DIR}/passages.json`, JSON.stringify(passages));
  console.log(`Saved ${DATA_DIR}/passages.json`);

  // Check for cached embeddings
  const embPath = `${DATA_DIR}/embeddings.bin`;
  if (existsSync(embPath)) {
    const existing = readFileSync(embPath);
    const existingCount = existing.byteLength / (DIM * 4);
    if (existingCount >= passages.length) {
      console.log(
        `Embeddings already cached (${existingCount.toLocaleString()} vectors), skipping embedding step.`,
      );
      return;
    }
    console.log(
      `Cached embeddings (${existingCount.toLocaleString()}) < passages (${passages.length.toLocaleString()}), re-embedding...`,
    );
  }

  const embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");

  const flatData = new Float32Array(passages.length * DIM);
  const start = performance.now();
  for (let i = 0; i < passages.length; i += BATCH_SIZE) {
    const batch = passages.slice(i, i + BATCH_SIZE);
    const output = await embedder(batch, { pooling: "mean", normalize: true });

    for (let j = 0; j < batch.length; j++) {
      const off = (i + j) * DIM;
      for (let d = 0; d < DIM; d++) {
        flatData[off + d] = output.data[j * DIM + d];
      }
    }

    const done = Math.min(i + BATCH_SIZE, passages.length);
    if (done % 1000 === 0 || done === passages.length) {
      const elapsed = (performance.now() - start) / 1000;
      const rate = done / elapsed;
      const remaining = ((passages.length - done) / rate).toFixed(0);
      console.log(
        `  Embedded ${done.toLocaleString()} / ${passages.length.toLocaleString()} (${rate.toFixed(0)} vec/s, ~${remaining}s remaining)`,
      );
    }
  }

  writeFileSync(`${DATA_DIR}/embeddings.bin`, Buffer.from(flatData.buffer));

  const sizeMB = (flatData.byteLength / 1e6).toFixed(1);
  const elapsed = ((performance.now() - start) / 1000).toFixed(1);
  console.log(
    `Saved ${DATA_DIR}/embeddings.bin (${passages.length.toLocaleString()} vectors, ${sizeMB} MB, ${elapsed}s)`,
  );
}

main().catch((err) => {
  console.error("Error:", err);
  process.exit(1);
});
