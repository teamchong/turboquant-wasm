/**
 * Search engine: TQ vector search vs brute-force f32 baseline.
 * Both paths use WebGPU when available — fair apple-to-apple comparison.
 * Brute-force is disabled when raw vectors exceed GPU buffer limits.
 */

import type { SearchData } from "./data-loader.js";
import { BruteGpuIndex } from "turboquant-wasm/gpu";

export interface SearchResult {
  index: number;
  passage: string;
  score: number;
}

export interface SearchComparison {
  tqResults: SearchResult[];
  bruteResults: SearchResult[] | null;
  tqTimeMs: number;
  bruteTimeMs: number | null;
  recallAtK: number | null;
  bruteDisabledReason: string | null;
}

let bruteGpu: BruteGpuIndex | null = null;
let bruteScoresBuffer: Float32Array | null = null;
let usedBuffer: Uint8Array | null = null;
let bruteDisabledReason: string | null = null;

function ensureBuffers(n: number) {
  if (!bruteScoresBuffer || bruteScoresBuffer.length < n) {
    bruteScoresBuffer = new Float32Array(n);
  }
  if (!usedBuffer || usedBuffer.length < n) {
    usedBuffer = new Uint8Array(n);
  }
}

function topKFromScores(scores: Float32Array, k: number, n: number): number[] {
  const topK: number[] = [];
  usedBuffer!.fill(0, 0, n);

  for (let t = 0; t < k; t++) {
    let bestIdx = -1;
    let bestScore = -Infinity;
    for (let i = 0; i < n; i++) {
      if (!usedBuffer![i] && scores[i] > bestScore) {
        bestScore = scores[i];
        bestIdx = i;
      }
    }
    if (bestIdx >= 0) {
      topK.push(bestIdx);
      usedBuffer![bestIdx] = 1;
    }
  }
  return topK;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

/** Reset GPU state. Call when switching datasets. */
export function resetSearch(): void {
  if (bruteGpu) {
    bruteGpu.destroy();
    bruteGpu = null;
  }
  bruteScoresBuffer = null;
  usedBuffer = null;
  bruteDisabledReason = null;
}

export async function search(
  query: Float32Array,
  data: SearchData,
  topK: number = 10,
): Promise<SearchComparison> {
  ensureBuffers(data.numVectors);

  const tqStart = performance.now();
  const tqScores = await data.tq.dotBatch(query, data.compressedConcat, data.bytesPerVector);
  const tqTimeMs = performance.now() - tqStart;

  const tqTopK = topKFromScores(tqScores, topK, data.numVectors);
  const tqResults: SearchResult[] = tqTopK.map((i) => ({
    index: i,
    passage: data.passages[i],
    score: tqScores[i],
  }));

  let bruteResults: SearchResult[] | null = null;
  let bruteTimeMs: number | null = null;
  let recallAtK: number | null = null;

  if (data.rawVectors) {
    // Init GPU brute-force on first call
    if (!bruteGpu && !bruteDisabledReason && typeof navigator !== "undefined" && navigator.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
          const rawBytes = data.rawVectors.byteLength;
          const maxSize = adapter.limits.maxStorageBufferBindingSize;
          if (rawBytes > maxSize) {
            bruteDisabledReason =
              `Raw vectors (${formatBytes(rawBytes)}) exceed GPU buffer limit (${formatBytes(maxSize)}) — TQ compressed: ${formatBytes(data.compressedSizeBytes)}`;
          } else {
            const device = await adapter.requestDevice();
            bruteGpu = await BruteGpuIndex.create(device, data.rawVectors, data.dim);
          }
        }
      } catch {
        /* fall back to JS */
      }
    }

    const bruteStart = performance.now();
    let scores: Float32Array;

    if (bruteGpu) {
      scores = await bruteGpu.dotBatch(query);
    } else if (!bruteDisabledReason) {
      // CPU fallback
      scores = bruteScoresBuffer!;
      for (let i = 0; i < data.numVectors; i++) {
        let dot = 0;
        const base = i * data.dim;
        for (let d = 0; d < data.dim; d++) {
          dot += query[d] * data.rawVectors[base + d];
        }
        scores[i] = dot;
      }
    } else {
      // Brute-force disabled — skip
      return { tqResults, bruteResults: null, tqTimeMs, bruteTimeMs: null, recallAtK: null, bruteDisabledReason };
    }
    bruteTimeMs = performance.now() - bruteStart;

    const bruteTopK = topKFromScores(scores!, topK, data.numVectors);
    bruteResults = bruteTopK.map((i) => ({
      index: i,
      passage: data.passages[i],
      score: scores![i],
    }));

    const bruteSet = new Set(bruteTopK);
    const matches = tqTopK.filter((i) => bruteSet.has(i)).length;
    recallAtK = matches / topK;
  }

  return { tqResults, bruteResults, tqTimeMs, bruteTimeMs, recallAtK, bruteDisabledReason };
}
