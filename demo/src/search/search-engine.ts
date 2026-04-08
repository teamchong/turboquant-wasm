/**
 * Search engine: TQ compressed dot-product search vs brute-force comparison.
 * Pre-allocates reusable buffers to avoid GC pressure between searches.
 * Optionally uses WebGPU for GPU-accelerated dot product scan.
 */

import type { SearchData } from "./data-loader.js";
import { TQGpuIndex } from "turboquant-wasm/gpu";

export interface SearchResult {
  index: number;
  passage: string;
  score: number;
}

export interface SearchComparison {
  tqResults: SearchResult[];
  bruteResults: SearchResult[] | null;
  gpuResults: SearchResult[] | null;
  tqTimeMs: number;
  bruteTimeMs: number | null;
  gpuTimeMs: number | null;
  recallAtK: number | null;
}

let gpuIndex: TQGpuIndex | null = null;

export async function initGpuSearch(data: SearchData): Promise<boolean> {
  gpuIndex = await TQGpuIndex.create(data.tq, data.compressedConcat, data.bytesPerVector);
  return gpuIndex !== null;
}

// Pre-allocated buffers (created once, reused across searches)
let bruteScoresBuffer: Float32Array | null = null;
let usedBuffer: Uint8Array | null = null;

function ensureBuffers(n: number) {
  if (!bruteScoresBuffer || bruteScoresBuffer.length < n) {
    bruteScoresBuffer = new Float32Array(n);
  }
  if (!usedBuffer || usedBuffer.length < n) {
    usedBuffer = new Uint8Array(n);
  }
}

// O(n*k) partial selection — avoids full O(n log n) sort for small k
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

export async function search(
  query: Float32Array,
  data: SearchData,
  topK: number = 10,
): Promise<SearchComparison> {
  ensureBuffers(data.numVectors);

  // CPU TQ search
  const tqStart = performance.now();
  const tqScores = data.tq.dotBatch(query, data.compressedConcat, data.bytesPerVector);
  const tqTimeMs = performance.now() - tqStart;

  const tqTopK = topKFromScores(tqScores, topK, data.numVectors);
  const tqResults: SearchResult[] = tqTopK.map((i) => ({
    index: i,
    passage: data.passages[i],
    score: tqScores[i],
  }));

  // GPU TQ search
  let gpuResults: SearchResult[] | null = null;
  let gpuTimeMs: number | null = null;

  if (gpuIndex) {
    const gpuStart = performance.now();
    const gpuScores = await gpuIndex.dotBatchGpu(query);
    gpuTimeMs = performance.now() - gpuStart;

    const gpuTopK = topKFromScores(gpuScores, topK, data.numVectors);
    gpuResults = gpuTopK.map((i) => ({
      index: i,
      passage: data.passages[i],
      score: gpuScores[i],
    }));
  }

  // Brute-force baseline
  let bruteResults: SearchResult[] | null = null;
  let bruteTimeMs: number | null = null;
  let recallAtK: number | null = null;

  if (data.rawVectors) {
    const scores = bruteScoresBuffer!;
    const bruteStart = performance.now();
    for (let i = 0; i < data.numVectors; i++) {
      let dot = 0;
      const base = i * data.dim;
      for (let d = 0; d < data.dim; d++) {
        dot += query[d] * data.rawVectors[base + d];
      }
      scores[i] = dot;
    }
    bruteTimeMs = performance.now() - bruteStart;

    const bruteTopK = topKFromScores(scores, topK, data.numVectors);
    bruteResults = bruteTopK.map((i) => ({
      index: i,
      passage: data.passages[i],
      score: scores[i],
    }));

    const bruteSet = new Set(bruteTopK);
    const matches = tqTopK.filter((i) => bruteSet.has(i)).length;
    recallAtK = matches / topK;
  }

  return { tqResults, bruteResults, gpuResults, tqTimeMs, bruteTimeMs, gpuTimeMs, recallAtK };
}
