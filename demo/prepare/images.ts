#!/usr/bin/env bun
/**
 * Prepare image similarity demo data from Unsplash Lite dataset.
 *
 * Fetches image metadata (URLs, AI descriptions) from HuggingFace,
 * embeds descriptions with all-MiniLM-L6-v2 (384-dim) via Transformers.js,
 * then compresses with TurboQuant.
 *
 * Images with similar descriptions will have similar embeddings,
 * so clicking a "dog" image finds other animal/dog images.
 *
 * Usage: bun run prepare/images.ts [--count 1000]
 */

import { TurboQuant } from "turboquant-wasm";
import { pipeline } from "@huggingface/transformers";

const DIM = 384;
const SEED = 42;
const DEFAULT_COUNT = 1000;
const BATCH_SIZE = 64;

function parseArgs(): { count: number } {
  const args = process.argv.slice(2);
  let count = DEFAULT_COUNT;
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--count") count = parseInt(args[++i], 10);
  }
  return { count };
}

interface ImageMeta {
  id: string;
  url: string;
  desc: string;
}

async function fetchUnsplashMeta(count: number): Promise<ImageMeta[]> {
  console.log("Fetching Unsplash Lite metadata from HuggingFace...");
  const photos: ImageMeta[] = [];
  const batchSize = 100;

  for (let offset = 0; photos.length < count; offset += batchSize) {
    const url = `https://datasets-server.huggingface.co/rows?dataset=1aurent/unsplash-lite&config=default&split=train&offset=${offset}&length=${batchSize}`;
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HuggingFace API error: ${resp.status}`);
    const data = await resp.json();
    if (!data.rows?.length) break;

    for (const row of data.rows) {
      const photo = row.row.photo;
      const ai = row.row.ai;
      if (photo?.image_url && ai?.description && ai.description !== "nan") {
        photos.push({
          id: photo.id,
          url: photo.image_url,
          desc: ai.description,
        });
      }
      if (photos.length >= count) break;
    }
    if (offset % 500 === 0) console.log(`  Fetched ${photos.length} / ${count} photos`);
  }

  console.log(`Got ${photos.length} photos with URLs and descriptions`);
  return photos;
}

async function main() {
  const { count } = parseArgs();

  const meta = await fetchUnsplashMeta(count);
  const numImages = meta.length;

  await Bun.write("data/image_meta.json", JSON.stringify(meta));
  console.log(`Saved data/image_meta.json (${numImages} images)`);

  // Embed descriptions with all-MiniLM-L6-v2
  console.log("Loading embedding model (all-MiniLM-L6-v2)...");
  const embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");

  console.log(`Embedding ${numImages} descriptions...`);
  const embeddings = new Float32Array(numImages * DIM);

  for (let i = 0; i < numImages; i += BATCH_SIZE) {
    const batch = meta.slice(i, i + BATCH_SIZE).map((m) => m.desc);
    const output = await embedder(batch, { pooling: "mean", normalize: true });

    for (let j = 0; j < batch.length; j++) {
      for (let d = 0; d < DIM; d++) {
        embeddings[(i + j) * DIM + d] = output.data[j * DIM + d];
      }
    }
    if (i % (BATCH_SIZE * 5) === 0) {
      console.log(`  Embedded ${Math.min(i + BATCH_SIZE, numImages)} / ${numImages}`);
    }
  }

  await Bun.write(
    "data/image_embeddings.bin",
    new Uint8Array(embeddings.buffer, 0, numImages * DIM * 4),
  );
  console.log(`Saved data/image_embeddings.bin (${(numImages * DIM * 4 / 1e6).toFixed(1)} MB)`);

  // Compress with TurboQuant
  console.log("Compressing with TurboQuant...");
  const tq = await TurboQuant.init({ dim: DIM, seed: SEED });

  const firstBlob = tq.encode(embeddings.slice(0, DIM));
  const bytesPerVector = firstBlob.byteLength;
  console.log(
    `  ${bytesPerVector} bytes/vector (from ${DIM * 4} = ${((DIM * 4) / bytesPerVector).toFixed(1)}x)`,
  );

  const body = new Uint8Array(numImages * bytesPerVector);
  body.set(firstBlob, 0);
  for (let i = 1; i < numImages; i++) {
    body.set(tq.encode(embeddings.slice(i * DIM, (i + 1) * DIM)), i * bytesPerVector);
    if (i % 500 === 0) console.log(`  Encoded ${i} / ${numImages}`);
  }
  tq.destroy();

  // Write .tqv
  const header = new ArrayBuffer(17);
  const hView = new DataView(header);
  const hBytes = new Uint8Array(header);
  hBytes[0] = 0x54;
  hBytes[1] = 0x51;
  hBytes[2] = 0x56;
  hBytes[3] = 0x00;
  hView.setUint8(4, 1);
  hView.setUint32(5, numImages, true);
  hView.setUint16(9, DIM, true);
  hView.setUint32(11, SEED, true);
  hView.setUint16(15, bytesPerVector, true);

  const output = new Uint8Array(17 + body.byteLength);
  output.set(hBytes, 0);
  output.set(body, 17);
  await Bun.write("data/image_compressed.tqv", output);

  const rawMB = (numImages * DIM * 4) / 1e6;
  const compMB = output.byteLength / 1e6;
  console.log(`\n=== Results ===`);
  console.log(`  Images:     ${numImages}`);
  console.log(`  Dim:        ${DIM} (all-MiniLM-L6-v2 text embeddings of descriptions)`);
  console.log(`  Raw:        ${rawMB.toFixed(1)} MB`);
  console.log(`  Compressed: ${compMB.toFixed(1)} MB`);
  console.log(`  Ratio:      ${(rawMB / compMB).toFixed(1)}x`);
}

main().catch((err) => {
  console.error("Error:", err);
  process.exit(1);
});
