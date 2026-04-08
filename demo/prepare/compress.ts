#!/usr/bin/env bun
/**
 * Compress raw float32 embeddings with TurboQuant.
 *
 * Usage:
 *   bun run prepare/compress.ts                                    # data/ → data/
 *   bun run prepare/compress.ts --dir public/data/wiki-50k         # custom dir
 *
 * Input:  <dir>/embeddings.bin (flat float32 array)
 * Output: <dir>/compressed.tqv (TQ-compressed vectors with header)
 *
 * .tqv format:
 *   Header (17 bytes):
 *     magic:          4 bytes  "TQV\0"
 *     version:        1 byte   (1)
 *     numVectors:     4 bytes  uint32 LE
 *     dim:            2 bytes  uint16 LE
 *     seed:           4 bytes  uint32 LE
 *     bytesPerVector: 2 bytes  uint16 LE
 *   Body:
 *     bytesPerVector * numVectors bytes of concatenated TQ blobs
 */

import { TurboQuant } from "turboquant-wasm";

const DIM = 384;
const SEED = 42;

const dirIdx = Bun.argv.indexOf("--dir");
const DATA_DIR = dirIdx >= 0 && Bun.argv[dirIdx + 1] ? Bun.argv[dirIdx + 1] : "data";

async function main() {
  console.log(`Reading ${DATA_DIR}/embeddings.bin...`);
  const file = Bun.file(`${DATA_DIR}/embeddings.bin`);
  const buffer = await file.arrayBuffer();
  const raw = new Float32Array(buffer);
  const numVectors = raw.length / DIM;
  console.log(`  ${numVectors.toLocaleString()} vectors, dim=${DIM}`);

  console.log("Initializing TurboQuant...");
  const tq = await TurboQuant.init({ dim: DIM, seed: SEED });

  // Encode first vector to determine compressed size
  const firstBlob = tq.encode(raw.slice(0, DIM));
  const bytesPerVector = firstBlob.byteLength;
  console.log(`  Compressed size per vector: ${bytesPerVector} bytes (from ${DIM * 4} bytes = ${((DIM * 4) / bytesPerVector).toFixed(1)}x)`);

  console.log("Encoding all vectors...");
  const body = new Uint8Array(numVectors * bytesPerVector);
  body.set(firstBlob, 0);

  const startTime = performance.now();
  for (let i = 1; i < numVectors; i++) {
    const vec = raw.slice(i * DIM, (i + 1) * DIM);
    const blob = tq.encode(vec);
    body.set(blob, i * bytesPerVector);

    if (i % 1000 === 0) {
      console.log(`  Encoded ${i.toLocaleString()} / ${numVectors.toLocaleString()}`);
    }
  }
  const encodeTime = performance.now() - startTime;
  console.log(`  Encoding done in ${(encodeTime / 1000).toFixed(2)}s`);

  tq.destroy();

  // Write .tqv file
  const header = new ArrayBuffer(17);
  const hView = new DataView(header);
  const hBytes = new Uint8Array(header);
  // Magic: "TQV\0"
  hBytes[0] = 0x54; // T
  hBytes[1] = 0x51; // Q
  hBytes[2] = 0x56; // V
  hBytes[3] = 0x00; // \0
  hView.setUint8(4, 1); // version
  hView.setUint32(5, numVectors, true);
  hView.setUint16(9, DIM, true);
  hView.setUint32(11, SEED, true);
  hView.setUint16(15, bytesPerVector, true);

  const output = new Uint8Array(17 + body.byteLength);
  output.set(hBytes, 0);
  output.set(body, 17);

  await Bun.write(`${DATA_DIR}/compressed.tqv`, output);

  const originalMB = (numVectors * DIM * 4) / 1e6;
  const compressedMB = output.byteLength / 1e6;
  console.log("\n=== Compression Results ===");
  console.log(`  Vectors:      ${numVectors.toLocaleString()} x ${DIM}-dim`);
  console.log(`  Original:     ${originalMB.toFixed(1)} MB`);
  console.log(`  Compressed:   ${compressedMB.toFixed(1)} MB`);
  console.log(`  Ratio:        ${(originalMB / compressedMB).toFixed(1)}x`);
  console.log(`  Output:       ${DATA_DIR}/compressed.tqv`);
}

main().catch((err) => {
  console.error("Error:", err);
  process.exit(1);
});
