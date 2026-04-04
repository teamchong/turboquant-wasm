#!/usr/bin/env node
/**
 * Generate text embeddings using all-MiniLM-L6-v2 via Transformers.js.
 * Fetches Wikipedia sentences via HuggingFace Datasets API and embeds them.
 *
 * Usage: node prepare/embed.mjs
 * Output: data/passages.json, data/embeddings.bin
 */

import { pipeline } from "@xenova/transformers";
import { writeFileSync, mkdirSync, existsSync, readFileSync } from "fs";

const DATA_DIR = "data";
const NUM_PASSAGES = 5000;
const BATCH_SIZE = 64;
const DIM = 384;

async function fetchPassages() {
  const passagesPath = `${DATA_DIR}/passages.json`;
  if (existsSync(passagesPath)) {
    console.log("Loading cached passages from", passagesPath);
    return JSON.parse(readFileSync(passagesPath, "utf-8"));
  }

  console.log("Fetching Wikipedia sentences via HuggingFace Datasets API...");

  // Fetch rows via the datasets server API (returns JSON, no auth needed)
  const passages = [];
  const batchSize = 100;
  let offset = 0;

  while (passages.length < NUM_PASSAGES) {
    const url = `https://datasets-server.huggingface.co/rows?dataset=sentence-transformers/wikipedia-en-sentences&config=default&split=train&offset=${offset}&length=${batchSize}`;
    const resp = await fetch(url);
    if (!resp.ok) {
      throw new Error(`HuggingFace API error: ${resp.status} ${resp.statusText}`);
    }

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
    if (passages.length % 500 === 0) {
      console.log(`  Fetched ${passages.length} / ${NUM_PASSAGES} passages`);
    }
  }

  console.log(`Got ${passages.length} passages`);
  return passages;
}

async function main() {
  mkdirSync(DATA_DIR, { recursive: true });

  const passages = await fetchPassages();
  console.log(`Embedding ${passages.length} passages with all-MiniLM-L6-v2...`);

  writeFileSync(`${DATA_DIR}/passages.json`, JSON.stringify(passages));
  console.log(`Saved ${DATA_DIR}/passages.json`);

  const embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");

  const allVectors = [];
  for (let i = 0; i < passages.length; i += BATCH_SIZE) {
    const batch = passages.slice(i, i + BATCH_SIZE);
    const output = await embedder(batch, { pooling: "mean", normalize: true });

    for (let j = 0; j < batch.length; j++) {
      const vec = new Float32Array(DIM);
      for (let d = 0; d < DIM; d++) {
        vec[d] = output.data[j * DIM + d];
      }
      allVectors.push(vec);
    }

    if ((i / BATCH_SIZE) % 10 === 0) {
      console.log(`  Embedded ${Math.min(i + BATCH_SIZE, passages.length)} / ${passages.length}`);
    }
  }

  const flatData = new Float32Array(allVectors.length * DIM);
  for (let i = 0; i < allVectors.length; i++) {
    flatData.set(allVectors[i], i * DIM);
  }

  writeFileSync(`${DATA_DIR}/embeddings.bin`, Buffer.from(flatData.buffer));

  const sizeMB = (flatData.byteLength / 1e6).toFixed(1);
  console.log(`Saved ${DATA_DIR}/embeddings.bin (${allVectors.length} vectors, ${sizeMB} MB)`);
}

main().catch((err) => {
  console.error("Error:", err);
  process.exit(1);
});
