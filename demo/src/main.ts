/**
 * Demo entry point: side-by-side 3DGS viewer comparing original vs TurboQuant-compressed.
 */

import { SceneFormat } from "@mkkellogg/gaussian-splats-3d";
import { createViewerPair, loadScene } from "./viewer.js";
import { decompressTqply } from "./decompressor.js";
import { renderLeftStats, renderRightStats, showRatioBadge } from "./stats.js";

// Git LFS files served via media.githubusercontent.com (supports CORS)
const LFS_BASE = "https://media.githubusercontent.com/media/teamchong/turboquant-wasm/main/demo/public/data";
const DEFAULT_PLY_URL = `${LFS_BASE}/scene.ply`;
const DEFAULT_TQPLY_URL = `${LFS_BASE}/scene.tqply`;

function getSceneUrls(): { plyUrl: string; tqplyUrl: string } {
  const params = new URLSearchParams(window.location.search);
  return {
    plyUrl: params.get("ply") || DEFAULT_PLY_URL,
    tqplyUrl: params.get("tqply") || DEFAULT_TQPLY_URL,
  };
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

async function main() {
  const { plyUrl, tqplyUrl } = getSceneUrls();
  const shDegree = 2;

  const leftEl = document.getElementById("viewer-left")!;
  const rightEl = document.getElementById("viewer-right")!;
  const statsLeft = document.getElementById("stats-left")!;
  const statsRight = document.getElementById("stats-right")!;
  const ratioBadge = document.getElementById("ratio-badge")!;

  showLoading("Creating viewers...");
  const viewers = createViewerPair(leftEl, rightEl, shDegree);

  // Get original file size
  let originalFileSize = 0;
  try {
    const headResp = await fetch(plyUrl, { method: "HEAD" });
    const cl = headResp.headers.get("content-length");
    if (cl) originalFileSize = parseInt(cl, 10);
  } catch {
    /* fallback below */
  }

  // Decompress .tqply and load both scenes in parallel
  let tqResult: Awaited<ReturnType<typeof decompressTqply>> | undefined;

  await Promise.all([
    // Left: load original .ply
    (async () => {
      showLoading("Loading original .ply...");
      await loadScene(viewers.left, plyUrl, shDegree);
      console.log("Original .ply loaded");
    })(),

    // Right: decompress .tqply then load reconstructed PLY
    (async () => {
      tqResult = await decompressTqply(tqplyUrl, (_pct, msg) => {
        showLoading(msg);
      });
      showLoading("Loading TurboQuant scene...");
      await loadScene(viewers.right, tqResult.blobUrl, shDegree, SceneFormat.Ply);
      console.log("TurboQuant scene loaded");
    })(),
  ]);

  // Render stats
  const stats = {
    originalFileSize,
    compressedFileSize: tqResult!.stats.compressedFileSize,
    numGaussians: tqResult!.stats.numGaussians,
    shDegree: tqResult!.stats.shDegree,
    decodeMs: tqResult!.stats.decompressTimeMs,
  };

  renderLeftStats(statsLeft, stats);
  renderRightStats(statsRight, stats);

  if (originalFileSize) {
    showRatioBadge(ratioBadge, originalFileSize / stats.compressedFileSize);
  }

  hideLoading();
  viewers.startSync();
  console.log("Demo ready!", tqResult!.stats);
}

main().catch((err) => {
  console.error("Demo error:", err);
  const text = document.getElementById("loading-text")!;
  text.textContent = `Error: ${err.message}`;
  text.style.color = "#f44336";
});
