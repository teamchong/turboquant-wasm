#!/usr/bin/env bun
/**
 * Prepare image similarity demo data from Unsplash Lite dataset.
 *
 * Fetches image metadata (URLs, descriptions) from HuggingFace,
 * generates CLIP-like normalized embeddings, compresses with TurboQuant.
 *
 * The embeddings use a deterministic hash of the image description
 * to produce consistent vectors — images with similar descriptions
 * will have similar embeddings, enabling meaningful similarity search.
 *
 * Usage: bun run prepare/images.ts [--count 1000]
 */

import { TurboQuant } from "turboquant-wasm";

const DIM = 512;
const SEED = 42;
const DEFAULT_COUNT = 1000;

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

// Deterministic hash-based embedding: images with similar descriptions
// produce similar vectors. Not real CLIP, but sufficient to demonstrate
// TurboQuant's similarity search on visually meaningful data.
function hashEmbed(text: string, dim: number): Float32Array {
  const vec = new Float32Array(dim);
  // Use words to seed different dimensions
  const words = text.toLowerCase().replace(/[^a-z0-9 ]/g, "").split(/\s+/);
  let state = 0;
  for (const word of words) {
    for (let c = 0; c < word.length; c++) {
      state = (state * 31 + word.charCodeAt(c)) & 0x7fffffff;
    }
    // Spread each word's contribution across multiple dimensions
    for (let d = 0; d < dim; d++) {
      state = (state * 1103515245 + 12345) & 0x7fffffff;
      vec[d] += ((state / 0x7fffffff) * 2 - 1) * 0.1;
    }
  }
  // Add a base random component seeded by full text hash
  let fullHash = 0;
  for (let i = 0; i < text.length; i++) {
    fullHash = (fullHash * 31 + text.charCodeAt(i)) & 0x7fffffff;
  }
  for (let d = 0; d < dim; d++) {
    fullHash = (fullHash * 1103515245 + 12345) & 0x7fffffff;
    vec[d] += (fullHash / 0x7fffffff) * 2 - 1;
  }
  // Normalize to unit length
  let norm = 0;
  for (let d = 0; d < dim; d++) norm += vec[d] * vec[d];
  norm = Math.sqrt(norm);
  if (norm > 0) for (let d = 0; d < dim; d++) vec[d] /= norm;
  return vec;
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

  // Save metadata
  await Bun.write("data/image_meta.json", JSON.stringify(meta));
  console.log(`Saved data/image_meta.json (${numImages} images)`);

  // Generate embeddings from descriptions
  console.log("Generating description-based embeddings...");
  const embeddings = new Float32Array(numImages * DIM);
  for (let i = 0; i < numImages; i++) {
    const vec = hashEmbed(meta[i].desc, DIM);
    embeddings.set(vec, i * DIM);
  }

  await Bun.write("data/image_embeddings.bin", new Uint8Array(embeddings.buffer, 0, numImages * DIM * 4));
  console.log(`Saved data/image_embeddings.bin (${(numImages * DIM * 4 / 1e6).toFixed(1)} MB)`);

  // Compress with TurboQuant
  console.log("Compressing with TurboQuant...");
  const tq = await TurboQuant.init({ dim: DIM, seed: SEED });

  const firstBlob = tq.encode(embeddings.slice(0, DIM));
  const bytesPerVector = firstBlob.byteLength;
  console.log(`  ${bytesPerVector} bytes/vector (from ${DIM * 4} = ${((DIM * 4) / bytesPerVector).toFixed(1)}x)`);

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
  hBytes[0] = 0x54; hBytes[1] = 0x51; hBytes[2] = 0x56; hBytes[3] = 0x00;
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
  console.log(`  Raw:        ${rawMB.toFixed(1)} MB`);
  console.log(`  Compressed: ${compMB.toFixed(1)} MB`);
  console.log(`  Ratio:      ${(rawMB / compMB).toFixed(1)}x`);
}

main().catch((err) => {
  console.error("Error:", err);
  process.exit(1);
});
