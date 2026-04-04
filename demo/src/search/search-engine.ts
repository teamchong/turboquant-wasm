/**
 * Search engine: TQ compressed dot-product search vs brute-force comparison.
 */

import type { SearchData } from "./data-loader.js";

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
}

export function search(
  query: Float32Array,
  data: SearchData,
  topK: number = 10,
): SearchComparison {
  // TurboQuant compressed search using tq.dot()
  const tqStart = performance.now();
  const tqScores = new Float32Array(data.numVectors);
  for (let i = 0; i < data.numVectors; i++) {
    tqScores[i] = data.tq.dot(query, data.compressedBlobs[i]);
  }
  const tqTimeMs = performance.now() - tqStart;

  const tqTopK = topKIndices(tqScores, topK);
  const tqResults: SearchResult[] = tqTopK.map((i) => ({
    index: i,
    passage: data.passages[i],
    score: tqScores[i],
  }));

  // Debug: verify TQ dot product quality
  if (data.rawVectors) {
    console.log("=== TQ dot product debug ===");
    console.log("Query dim:", query.length, "norm:", Math.sqrt(query.reduce((s, v) => s + v * v, 0)).toFixed(4));
    for (const idx of tqTopK.slice(0, 3)) {
      let trueDot = 0;
      for (let d = 0; d < data.dim; d++) trueDot += query[d] * data.rawVectors[idx * data.dim + d];
      console.log(`  vec[${idx}]: tq.dot=${tqScores[idx].toFixed(4)}, true dot=${trueDot.toFixed(4)}`);
    }
  }

  // Brute-force uncompressed search
  let bruteResults: SearchResult[] | null = null;
  let bruteTimeMs: number | null = null;
  let recallAtK: number | null = null;

  if (data.rawVectors) {
    const bruteStart = performance.now();
    const bruteScores = new Float32Array(data.numVectors);
    for (let i = 0; i < data.numVectors; i++) {
      let dot = 0;
      const base = i * data.dim;
      for (let d = 0; d < data.dim; d++) {
        dot += query[d] * data.rawVectors[base + d];
      }
      bruteScores[i] = dot;
    }
    bruteTimeMs = performance.now() - bruteStart;

    const bruteTopK = topKIndices(bruteScores, topK);
    bruteResults = bruteTopK.map((i) => ({
      index: i,
      passage: data.passages[i],
      score: bruteScores[i],
    }));

    // Recall: how many of TQ top-K appear in brute-force top-K
    const bruteSet = new Set(bruteTopK);
    const matches = tqTopK.filter((i) => bruteSet.has(i)).length;
    recallAtK = matches / topK;
  }

  return { tqResults, bruteResults, tqTimeMs, bruteTimeMs, recallAtK };
}

function topKIndices(scores: Float32Array, k: number): number[] {
  const indices = Array.from({ length: scores.length }, (_, i) => i);
  indices.sort((a, b) => scores[b] - scores[a]);
  return indices.slice(0, k);
}
