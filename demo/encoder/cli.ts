#!/usr/bin/env bun
/**
 * Offline encoder: converts .ply → .tqply
 *
 * Usage:
 *   bun run encoder/cli.ts --input data/scene.ply --output data/scene.tqply [--seed 42] [--sh-degree 2]
 */

import { TurboQuant } from "turboquant-wasm";
import { parsePlyHeader, extractGaussians } from "../src/ply-parser.js";
import {
  writeTqply,
  TQPLY_VERSION,
  NON_SH_STRIDE,
  type TqplyHeader,
} from "../src/tqply-format.js";

function parseArgs() {
  const args = process.argv.slice(2);
  let input = "";
  let output = "";
  let seed = 42;
  let shDegree = 2;

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case "--input":
      case "-i":
        input = args[++i];
        break;
      case "--output":
      case "-o":
        output = args[++i];
        break;
      case "--seed":
        seed = parseInt(args[++i], 10);
        break;
      case "--sh-degree":
        shDegree = parseInt(args[++i], 10);
        break;
    }
  }

  if (!input || !output) {
    console.error(
      "Usage: bun run encoder/cli.ts --input <file.ply> --output <file.tqply> [--seed 42] [--sh-degree 2]",
    );
    process.exit(1);
  }

  return { input, output, seed, shDegree };
}

async function main() {
  const { input, output, seed, shDegree } = parseArgs();
  const startTime = performance.now();

  console.log(`Reading ${input}...`);
  const file = Bun.file(input);
  const buffer = await file.arrayBuffer();
  const originalSize = buffer.byteLength;
  console.log(`  Original size: ${(originalSize / 1024 / 1024).toFixed(2)} MB`);

  console.log("Parsing PLY...");
  const ply = parsePlyHeader(buffer);
  const gaussians = extractGaussians(ply, shDegree);
  const n = gaussians.numGaussians;
  console.log(`  ${n.toLocaleString()} Gaussians, ${gaussians.shRestCount} SH rest coefficients`);

  if (gaussians.shRestCount === 0) {
    console.error("Error: no SH rest coefficients found. Nothing to compress.");
    process.exit(1);
  }

  const dim = gaussians.shRestCount;
  // TurboQuant requires even dim
  if (dim % 2 !== 0) {
    console.error(`Error: SH rest count ${dim} is odd. TurboQuant requires even dimensions.`);
    process.exit(1);
  }

  console.log(`Initializing TurboQuant (dim=${dim}, seed=${seed})...`);
  const tq = await TurboQuant.init({ dim, seed });

  // Encode SH rest coefficients
  console.log("Encoding SH coefficients...");
  const encodeStart = performance.now();

  // First pass: encode one to determine blob size
  const firstVec = gaussians.shRest.slice(0, dim);
  const firstBlob = tq.encode(firstVec);
  const blobSize = firstBlob.byteLength;
  console.log(`  TQ blob size: ${blobSize} bytes (from ${dim * 4} bytes = ${(dim * 4 / blobSize).toFixed(2)}x on SH)`);

  const tqShBlock = new Uint8Array(n * blobSize);
  tqShBlock.set(firstBlob, 0);

  for (let i = 1; i < n; i++) {
    const vec = gaussians.shRest.slice(i * dim, (i + 1) * dim);
    const blob = tq.encode(vec);
    tqShBlock.set(blob, i * blobSize);

    if (i % 100000 === 0) {
      console.log(`  Encoded ${i.toLocaleString()} / ${n.toLocaleString()}`);
    }
  }

  const encodeTime = performance.now() - encodeStart;
  console.log(`  Encoding done in ${(encodeTime / 1000).toFixed(2)}s`);

  tq.destroy();

  // Build non-SH block: pos(3) + dc(3) + opacity(1) + scale(3) + rot(4) = 14 floats
  console.log("Building non-SH block...");
  const nonShBlock = new Uint8Array(n * NON_SH_STRIDE);
  const nonShView = new DataView(nonShBlock.buffer);

  for (let i = 0; i < n; i++) {
    const base = i * NON_SH_STRIDE;
    let off = 0;

    function writeFloat(val: number) {
      nonShView.setFloat32(base + off, val, true);
      off += 4;
    }

    // position
    writeFloat(gaussians.positions[i * 3 + 0]);
    writeFloat(gaussians.positions[i * 3 + 1]);
    writeFloat(gaussians.positions[i * 3 + 2]);

    // SH DC
    writeFloat(gaussians.shDC[i * 3 + 0]);
    writeFloat(gaussians.shDC[i * 3 + 1]);
    writeFloat(gaussians.shDC[i * 3 + 2]);

    // opacity
    writeFloat(gaussians.opacity[i]);

    // scale
    writeFloat(gaussians.scales[i * 3 + 0]);
    writeFloat(gaussians.scales[i * 3 + 1]);
    writeFloat(gaussians.scales[i * 3 + 2]);

    // rotation
    writeFloat(gaussians.rotations[i * 4 + 0]);
    writeFloat(gaussians.rotations[i * 4 + 1]);
    writeFloat(gaussians.rotations[i * 4 + 2]);
    writeFloat(gaussians.rotations[i * 4 + 3]);
  }

  // Write .tqply
  const header: TqplyHeader = {
    version: TQPLY_VERSION,
    shDegree,
    numGaussians: n,
    tqSeed: seed,
    tqDim: dim,
    tqBlobSize: blobSize,
    nonShStride: NON_SH_STRIDE,
  };

  console.log("Writing .tqply...");
  const tqplyBytes = writeTqply(header, nonShBlock, tqShBlock);
  await Bun.write(output, tqplyBytes);

  const compressedSize = tqplyBytes.byteLength;
  const totalTime = performance.now() - startTime;

  console.log("\n=== Compression Results ===");
  console.log(`  Gaussians:       ${n.toLocaleString()}`);
  console.log(`  SH degree:       ${shDegree} (${dim} rest coefficients)`);
  console.log(`  Original .ply:   ${(originalSize / 1024 / 1024).toFixed(2)} MB`);
  console.log(`  Compressed:      ${(compressedSize / 1024 / 1024).toFixed(2)} MB`);
  console.log(`  Ratio:           ${(originalSize / compressedSize).toFixed(2)}x overall`);
  console.log(`  SH compression:  ${dim * 4} → ${blobSize} bytes = ${(dim * 4 / blobSize).toFixed(2)}x`);
  console.log(`  Encode time:     ${(encodeTime / 1000).toFixed(2)}s`);
  console.log(`  Total time:      ${(totalTime / 1000).toFixed(2)}s`);
  console.log(`  Output:          ${output}`);
}

main().catch((err) => {
  console.error("Error:", err);
  process.exit(1);
});
