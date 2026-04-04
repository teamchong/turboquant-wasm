/**
 * Image similarity demo: click an image to find similar ones via TurboQuant compressed CLIP embeddings.
 */

import { TurboQuant } from "turboquant-wasm";

interface ImageMeta {
  id: string;
  url: string;
  desc: string;
}

function thumbUrl(url: string): string {
  // Unsplash image URLs support dynamic resizing via query params
  return url + "?w=200&h=200&fit=crop&auto=format&q=75";
}

interface ImageData {
  meta: ImageMeta[];
  rawVectors: Float32Array | null;
  compressedBlobs: Uint8Array[];
  tq: TurboQuant;
  dim: number;
  numImages: number;
  rawSizeBytes: number;
  compressedSizeBytes: number;
}

function showLoading(msg: string) {
  const overlay = document.getElementById("loading-overlay")!;
  const text = document.getElementById("loading-text")!;
  overlay.classList.remove("hidden");
  text.textContent = msg;
}

function hideLoading() {
  document.getElementById("loading-overlay")!.classList.add("hidden");
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

async function loadImageData(onProgress: (msg: string) => void): Promise<ImageData> {
  onProgress("Loading image metadata...");
  const meta: ImageMeta[] = await fetch("data/image_meta.json").then((r) => r.json());

  onProgress("Loading compressed CLIP embeddings...");
  const tqvBuffer = await fetch("data/image_compressed.tqv").then((r) => r.arrayBuffer());
  const hView = new DataView(tqvBuffer);
  const numImages = hView.getUint32(5, true);
  const dim = hView.getUint16(9, true);
  const seed = hView.getUint32(11, true);
  const bytesPerVector = hView.getUint16(15, true);

  const compressedBlobs: Uint8Array[] = [];
  for (let i = 0; i < numImages; i++) {
    compressedBlobs.push(new Uint8Array(tqvBuffer, 17 + i * bytesPerVector, bytesPerVector));
  }

  onProgress("Initializing TurboQuant...");
  const tq = await TurboQuant.init({ dim, seed });

  onProgress("Loading uncompressed embeddings for comparison...");
  let rawVectors: Float32Array | null = null;
  try {
    const rawBuf = await fetch("data/image_embeddings.bin").then((r) => r.arrayBuffer());
    rawVectors = new Float32Array(rawBuf);
  } catch { /* compressed-only mode */ }

  return {
    meta,
    rawVectors,
    compressedBlobs,
    tq,
    dim,
    numImages,
    rawSizeBytes: numImages * dim * 4,
    compressedSizeBytes: tqvBuffer.byteLength,
  };
}

function renderStats(data: ImageData) {
  const ratio = data.rawSizeBytes / data.compressedSizeBytes;
  document.getElementById("index-stats")!.innerHTML = `
    <div class="stat-item"><div class="stat-value">${data.numImages.toLocaleString()}</div><div class="stat-label">images (${data.dim}-dim CLIP)</div></div>
    <div class="stat-item"><div class="stat-value">${formatBytes(data.rawSizeBytes)}</div><div class="stat-label">uncompressed</div></div>
    <div class="stat-item"><div class="stat-value">${formatBytes(data.compressedSizeBytes)}</div><div class="stat-label">TurboQuant</div></div>
    <div class="stat-item"><div class="stat-value">${ratio.toFixed(1)}x</div><div class="stat-label">compression</div></div>
  `;
}

function renderGallery(
  data: ImageData,
  order: number[] | null,
  scores: Float32Array | null,
  selectedIdx: number | null,
) {
  const gallery = document.getElementById("gallery")!;
  const indices = order ?? Array.from({ length: data.numImages }, (_, i) => i);
  const displayCount = Math.min(indices.length, 100);

  gallery.innerHTML = indices
    .slice(0, displayCount)
    .map((idx) => {
      const m = data.meta[idx];
      const sel = idx === selectedIdx ? " selected" : "";
      const scoreBadge =
        scores && idx !== selectedIdx
          ? `<div class="score-badge">${scores[idx].toFixed(3)}</div>`
          : "";
      return `<div class="gallery-item${sel}" data-idx="${idx}">
        <img src="${thumbUrl(m.url)}" loading="lazy" alt="${m.desc}" />
        ${scoreBadge}
      </div>`;
    })
    .join("");

  // Click handlers
  gallery.querySelectorAll(".gallery-item").forEach((el) => {
    el.addEventListener("click", () => {
      const idx = parseInt((el as HTMLElement).dataset.idx!);
      findSimilar(data, idx);
    });
  });
}

function findSimilar(data: ImageData, queryIdx: number) {
  const querySection = document.getElementById("query-section")!;
  const queryImage = document.getElementById("query-image") as HTMLImageElement;
  const queryInfo = document.getElementById("query-info")!;

  queryImage.src = thumbUrl(data.meta[queryIdx].url);
  querySection.classList.remove("hidden");

  // Compute similarity using TQ dot product
  const start = performance.now();
  const scores = new Float32Array(data.numImages);
  const queryBlob = data.compressedBlobs[queryIdx];
  const queryVec = data.tq.decode(queryBlob);

  for (let i = 0; i < data.numImages; i++) {
    scores[i] = data.tq.dot(queryVec, data.compressedBlobs[i]);
  }
  const tqMs = performance.now() - start;

  // Sort by score descending (skip self)
  const ranked = Array.from({ length: data.numImages }, (_, i) => i);
  ranked.sort((a, b) => scores[b] - scores[a]);

  // Show info
  queryInfo.innerHTML = `
    Found similar images in <span class="fast">${tqMs.toFixed(1)}ms</span>
    using <span class="highlight">tq.dot()</span> on compressed vectors
  `;

  renderGallery(data, ranked, scores, queryIdx);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

async function main() {
  showLoading("Loading image index...");
  const data = await loadImageData((msg) => showLoading(msg));
  renderStats(data);
  hideLoading();

  // Show random initial grid
  const shuffled = Array.from({ length: data.numImages }, (_, i) => i);
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  renderGallery(data, shuffled, null, null);

  // Clear button
  document.getElementById("btn-clear")!.addEventListener("click", () => {
    document.getElementById("query-section")!.classList.add("hidden");
    renderGallery(data, shuffled, null, null);
  });

  console.log("Image similarity demo ready!", {
    images: data.numImages,
    dim: data.dim,
    compressedMB: (data.compressedSizeBytes / 1e6).toFixed(1),
  });
}

main().catch((err) => {
  console.error("Image demo error:", err);
  const text = document.getElementById("loading-text")!;
  text.textContent = `Error: ${err.message}`;
  text.style.color = "#f44336";
});
