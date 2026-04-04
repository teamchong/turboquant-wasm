#!/usr/bin/env bun
/**
 * Prepare image similarity demo data:
 * 1. Read precomputed CLIP ViT-B/32 embeddings (512-dim) from features.npy
 * 2. Read photo IDs from photo_ids.csv
 * 3. Sample N images
 * 4. Write: image_ids.json, image_embeddings.bin, image_compressed.tqv
 *
 * Usage: bun run prepare/images.ts [--count 2000]
 */

import { TurboQuant } from "turboquant-wasm";

const DIM = 512;
const SEED = 42;
const DEFAULT_COUNT = 2000;

function parseArgs(): { count: number } {
  const args = process.argv.slice(2);
  let count = DEFAULT_COUNT;
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--count") count = parseInt(args[++i], 10);
  }
  return { count };
}

function float16ToFloat32(h: number): number {
  const sign = (h >> 15) & 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) {
    // Subnormal or zero
    return (sign ? -1 : 1) * Math.pow(2, -14) * (frac / 1024);
  } else if (exp === 31) {
    return frac === 0 ? (sign ? -Infinity : Infinity) : NaN;
  }
  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

function readNpy(buffer: ArrayBuffer): { shape: number[]; data: Float32Array } {
  const bytes = new Uint8Array(buffer);
  if (bytes[0] !== 0x93 || bytes[1] !== 0x4e) {
    throw new Error("Not a .npy file");
  }
  const major = bytes[6];
  let headerLen: number;
  let dataOffset: number;
  if (major >= 2) {
    headerLen = bytes[8] | (bytes[9] << 8) | (bytes[10] << 16) | (bytes[11] << 24);
    dataOffset = 12 + headerLen;
  } else {
    headerLen = bytes[8] | (bytes[9] << 8);
    dataOffset = 10 + headerLen;
  }
  const headerStr = new TextDecoder().decode(bytes.slice(dataOffset - headerLen, dataOffset));

  const shapeMatch = headerStr.match(/\((\d+),\s*(\d+)\)/);
  if (!shapeMatch) throw new Error(`Cannot parse shape from: ${headerStr}`);
  const shape = [parseInt(shapeMatch[1]), parseInt(shapeMatch[2])];

  const descrMatch = headerStr.match(/'descr':\s*'([^']+)'/);
  const dtype = descrMatch ? descrMatch[1] : "<f4";

  if (dtype === "<f2") {
    // Float16 — convert to Float32
    const u16 = new Uint16Array(buffer, dataOffset, shape[0] * shape[1]);
    const f32 = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) {
      f32[i] = float16ToFloat32(u16[i]);
    }
    return { shape, data: f32 };
  }

  const data = new Float32Array(buffer, dataOffset, shape[0] * shape[1]);
  return { shape, data };
}

async function main() {
  const { count } = parseArgs();

  console.log("Reading features.npy...");
  const npyBuf = await Bun.file("data/features.npy").arrayBuffer();
  const { shape, data: allFeatures } = readNpy(npyBuf);
  const totalImages = shape[0];
  const dim = shape[1];
  console.log(`  ${totalImages} images, ${dim}-dim`);

  if (dim !== DIM) {
    throw new Error(`Expected ${DIM}-dim, got ${dim}-dim`);
  }

  console.log("Reading photo_ids.csv...");
  const csvText = await Bun.file("data/photo_ids.csv").text();
  const allIds = csvText.trim().split("\n").slice(1); // skip header
  console.log(`  ${allIds.length} photo IDs`);

  // Sample evenly spaced images for diversity
  const step = Math.floor(totalImages / count);
  const indices: number[] = [];
  for (let i = 0; i < count && i * step < totalImages; i++) {
    indices.push(i * step);
  }
  console.log(`Sampling ${indices.length} images (step=${step})...`);

  const ids: string[] = [];
  const embeddings = new Float32Array(indices.length * DIM);
  for (let i = 0; i < indices.length; i++) {
    const idx = indices[i];
    ids.push(allIds[idx]);
    embeddings.set(allFeatures.slice(idx * DIM, (idx + 1) * DIM), i * DIM);
  }

  // Write image IDs
  await Bun.write("data/image_ids.json", JSON.stringify(ids));
  console.log(`Saved data/image_ids.json (${ids.length} IDs)`);

  // Write raw embeddings
  await Bun.write("data/image_embeddings.bin", new Uint8Array(embeddings.buffer));
  console.log(`Saved data/image_embeddings.bin (${(embeddings.byteLength / 1e6).toFixed(1)} MB)`);

  // Compress with TurboQuant
  console.log("Compressing with TurboQuant...");
  const tq = await TurboQuant.init({ dim: DIM, seed: SEED });

  const firstBlob = tq.encode(embeddings.slice(0, DIM));
  const bytesPerVector = firstBlob.byteLength;
  console.log(`  ${bytesPerVector} bytes/vector (from ${DIM * 4} = ${((DIM * 4) / bytesPerVector).toFixed(1)}x)`);

  const body = new Uint8Array(indices.length * bytesPerVector);
  body.set(firstBlob, 0);
  for (let i = 1; i < indices.length; i++) {
    const vec = embeddings.slice(i * DIM, (i + 1) * DIM);
    body.set(tq.encode(vec), i * bytesPerVector);
    if (i % 500 === 0) console.log(`  Encoded ${i} / ${indices.length}`);
  }
  tq.destroy();

  // Write .tqv
  const header = new ArrayBuffer(17);
  const hView = new DataView(header);
  const hBytes = new Uint8Array(header);
  hBytes[0] = 0x54; hBytes[1] = 0x51; hBytes[2] = 0x56; hBytes[3] = 0x00;
  hView.setUint8(4, 1);
  hView.setUint32(5, indices.length, true);
  hView.setUint16(9, DIM, true);
  hView.setUint32(11, SEED, true);
  hView.setUint16(15, bytesPerVector, true);

  const output = new Uint8Array(17 + body.byteLength);
  output.set(hBytes, 0);
  output.set(body, 17);
  await Bun.write("data/image_compressed.tqv", output);

  const rawMB = (indices.length * DIM * 4) / 1e6;
  const compMB = output.byteLength / 1e6;
  console.log(`\n=== Results ===`);
  console.log(`  Images:     ${indices.length}`);
  console.log(`  Raw:        ${rawMB.toFixed(1)} MB`);
  console.log(`  Compressed: ${compMB.toFixed(1)} MB`);
  console.log(`  Ratio:      ${(rawMB / compMB).toFixed(1)}x`);

  // Cleanup large files
  console.log("\nYou can delete the large source files:");
  console.log("  rm data/features.npy data/photo_ids.csv");
}

main().catch((err) => {
  console.error("Error:", err);
  process.exit(1);
});
