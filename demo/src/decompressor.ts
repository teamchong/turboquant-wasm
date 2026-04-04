/**
 * Browser-side .tqply decompressor.
 * Fetches a .tqply file, decompresses SH via turboquant-wasm,
 * and reconstructs a valid PLY blob for GaussianSplats3D.
 */

import { TurboQuant } from "turboquant-wasm";
import {
  readTqplyHeader,
  readTqplyBody,
  NON_SH_STRIDE,
  type TqplyHeader,
} from "./tqply-format.js";
import type { ParsedGaussians } from "./ply-parser.js";
import { buildPlyBinary } from "./ply-writer.js";

export interface DecompressResult {
  plyBlob: Blob;
  blobUrl: string;
  stats: {
    compressedFileSize: number;
    numGaussians: number;
    shDegree: number;
    tqDim: number;
    decompressTimeMs: number;
    reconstructedSize: number;
  };
}

export async function decompressTqply(
  source: ArrayBuffer | string,
  onProgress?: (pct: number, msg: string) => void,
): Promise<DecompressResult> {
  // Fetch if URL
  let buffer: ArrayBuffer;
  let compressedFileSize: number;

  if (typeof source === "string") {
    onProgress?.(0, "Fetching .tqply...");
    const resp = await fetch(source);
    if (!resp.ok) throw new Error(`Failed to fetch: ${resp.status} ${resp.statusText}`);
    buffer = await resp.arrayBuffer();
    compressedFileSize = buffer.byteLength;
  } else {
    buffer = source;
    compressedFileSize = buffer.byteLength;
  }

  onProgress?.(10, "Parsing header...");
  const header = readTqplyHeader(buffer);
  const { nonShBlock, tqShBlock } = readTqplyBody(buffer, header);

  const n = header.numGaussians;
  const dim = header.tqDim;
  const blobSize = header.tqBlobSize;

  onProgress?.(15, "Initializing TurboQuant WASM...");
  const tq = await TurboQuant.init({ dim, seed: header.tqSeed });

  // Extract non-SH data
  onProgress?.(20, "Reading vertex data...");
  const positions = new Float32Array(n * 3);
  const shDC = new Float32Array(n * 3);
  const opacity = new Float32Array(n);
  const scales = new Float32Array(n * 3);
  const rotations = new Float32Array(n * 4);

  for (let i = 0; i < n; i++) {
    const base = i * NON_SH_STRIDE;
    let off = 0;

    function readFloat(): number {
      const val = nonShBlock.getFloat32(base + off, true);
      off += 4;
      return val;
    }

    positions[i * 3 + 0] = readFloat();
    positions[i * 3 + 1] = readFloat();
    positions[i * 3 + 2] = readFloat();

    shDC[i * 3 + 0] = readFloat();
    shDC[i * 3 + 1] = readFloat();
    shDC[i * 3 + 2] = readFloat();

    opacity[i] = readFloat();

    scales[i * 3 + 0] = readFloat();
    scales[i * 3 + 1] = readFloat();
    scales[i * 3 + 2] = readFloat();

    rotations[i * 4 + 0] = readFloat();
    rotations[i * 4 + 1] = readFloat();
    rotations[i * 4 + 2] = readFloat();
    rotations[i * 4 + 3] = readFloat();
  }

  // Decompress SH
  onProgress?.(30, "Decompressing SH coefficients...");
  const decompressStart = performance.now();
  const shRest = new Float32Array(n * dim);

  const BATCH = 50000;
  for (let i = 0; i < n; i++) {
    const blob = tqShBlock.slice(i * blobSize, (i + 1) * blobSize);
    const decoded = tq.decode(blob);
    shRest.set(decoded, i * dim);

    if (i % BATCH === 0 && i > 0) {
      const pct = 30 + Math.round((i / n) * 50);
      onProgress?.(pct, `Decoded ${i.toLocaleString()} / ${n.toLocaleString()} Gaussians`);
      // Yield to UI thread
      await new Promise((r) => setTimeout(r, 0));
    }
  }

  const decompressTimeMs = performance.now() - decompressStart;
  tq.destroy();

  // Reconstruct PLY
  onProgress?.(85, "Reconstructing PLY...");
  const gaussians: ParsedGaussians = {
    numGaussians: n,
    positions,
    shDC,
    shRest,
    shRestCount: dim,
    opacity,
    scales,
    rotations,
    originalFileSize: compressedFileSize,
  };

  const plyBytes = buildPlyBinary(gaussians);
  const plyBlob = new Blob([plyBytes.buffer as ArrayBuffer], { type: "application/octet-stream" });
  const blobUrl = URL.createObjectURL(plyBlob);

  onProgress?.(100, "Done!");

  return {
    plyBlob,
    blobUrl,
    stats: {
      compressedFileSize,
      numGaussians: n,
      shDegree: header.shDegree,
      tqDim: dim,
      decompressTimeMs,
      reconstructedSize: plyBytes.byteLength,
    },
  };
}
