/**
 * Load passages, raw embeddings, and TQ-compressed vectors.
 * Supports multiple datasets via DatasetInfo config.
 */

import { TurboQuant } from "turboquant-wasm";

export interface SearchData {
  passages: string[];
  rawVectors: Float32Array | null;
  compressedConcat: Uint8Array;
  tq: TurboQuant;
  dim: number;
  numVectors: number;
  bytesPerVector: number;
  rawSizeBytes: number;
  compressedSizeBytes: number;
}

export interface DatasetInfo {
  name: string;
  label: string;
  basePath: string;
}

export const DATASETS: DatasetInfo[] = [
  { name: "wiki-5k", label: "Wikipedia 5K", basePath: "data" },
  { name: "wiki-50k", label: "Wikipedia 50K", basePath: "data/wiki-50k" },
];

interface TqvHeader {
  numVectors: number;
  dim: number;
  seed: number;
  bytesPerVector: number;
}

function parseTqvHeader(buffer: ArrayBuffer): TqvHeader {
  const view = new DataView(buffer);
  const magic = new Uint8Array(buffer, 0, 4);
  if (magic[0] !== 0x54 || magic[1] !== 0x51 || magic[2] !== 0x56 || magic[3] !== 0x00) {
    throw new Error("Invalid .tqv magic bytes");
  }
  const version = view.getUint8(4);
  if (version !== 1) throw new Error(`Unsupported .tqv version: ${version}`);

  return {
    numVectors: view.getUint32(5, true),
    dim: view.getUint16(9, true),
    seed: view.getUint32(11, true),
    bytesPerVector: view.getUint16(15, true),
  };
}

export async function loadSearchData(
  dataset: DatasetInfo,
  onProgress: (msg: string) => void,
): Promise<SearchData> {
  const base = dataset.basePath;

  onProgress("Loading passages...");
  const passages: string[] = await fetch(`${base}/passages.json`).then((r) =>
    r.json(),
  );

  onProgress("Loading compressed vectors...");
  const tqvBuffer = await fetch(`${base}/compressed.tqv`).then((r) =>
    r.arrayBuffer(),
  );
  const header = parseTqvHeader(tqvBuffer);

  const bodyOffset = 17;
  const compressedConcat = new Uint8Array(
    tqvBuffer,
    bodyOffset,
    header.numVectors * header.bytesPerVector,
  );

  onProgress("Initializing TurboQuant WASM...");
  const tq = await TurboQuant.init({ dim: header.dim, seed: header.seed });

  let rawVectors: Float32Array | null = null;
  onProgress("Loading uncompressed vectors for comparison...");
  try {
    const rawBuffer = await fetch(`${base}/embeddings.bin`).then((r) => {
      if (!r.ok) throw new Error(r.statusText);
      return r.arrayBuffer();
    });
    rawVectors = new Float32Array(rawBuffer);
  } catch {
    // Compressed-only mode — brute-force comparison unavailable
  }

  const rawSizeBytes = header.numVectors * header.dim * 4;
  const compressedSizeBytes = tqvBuffer.byteLength;

  return {
    passages,
    rawVectors,
    compressedConcat,
    tq,
    dim: header.dim,
    numVectors: header.numVectors,
    bytesPerVector: header.bytesPerVector,
    rawSizeBytes,
    compressedSizeBytes,
  };
}
