/**
 * Client-side query embedding using Transformers.js (all-MiniLM-L6-v2).
 * The ONNX model (~23MB) is downloaded once and cached by the browser.
 */

import { pipeline, type FeatureExtractionPipeline } from "@huggingface/transformers";

let embedder: FeatureExtractionPipeline | null = null;

export async function initEmbedder(
  onProgress: (msg: string) => void,
): Promise<void> {
  onProgress("Loading embedding model (~23MB, cached after first visit)...");
  embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
  onProgress("Embedding model ready");
}

export async function embedQuery(text: string): Promise<Float32Array> {
  if (!embedder) throw new Error("Embedder not initialized — call initEmbedder first");
  const output = await embedder(text, {
    pooling: "mean",
    normalize: true,
  });
  return new Float32Array(output.tolist()[0] as number[]);
}
