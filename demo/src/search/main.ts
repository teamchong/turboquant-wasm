/**
 * Vector search demo entry point.
 * Side-by-side: TurboQuant compressed search vs brute-force uncompressed.
 */

import { loadSearchData, type SearchData } from "./data-loader.js";
import { search, type SearchComparison } from "./search-engine.js";
import { initEmbedder, embedQuery } from "./embedder.js";

const SUGGESTED_QUERIES = [
  "how does the human heart pump blood",
  "history of the roman empire",
  "what causes earthquakes and tsunamis",
  "how do computers store information",
  "largest animals in the ocean",
  "how does electricity work",
  "space exploration and mars missions",
  "evolution of programming languages",
];

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

function renderIndexStats(data: SearchData) {
  const ratio = data.rawSizeBytes / data.compressedSizeBytes;
  document.getElementById("index-stats")!.innerHTML = `
    <div class="stat-item">
      <div class="stat-value">${data.numVectors.toLocaleString()}</div>
      <div class="stat-label">vectors (${data.dim}-dim)</div>
    </div>
    <div class="stat-item">
      <div class="stat-value">${formatBytes(data.rawSizeBytes)}</div>
      <div class="stat-label">uncompressed</div>
    </div>
    <div class="stat-item">
      <div class="stat-value">${formatBytes(data.compressedSizeBytes)}</div>
      <div class="stat-label">TurboQuant compressed</div>
    </div>
    <div class="stat-item">
      <div class="stat-value">${ratio.toFixed(1)}x</div>
      <div class="stat-label">compression ratio</div>
    </div>
  `;
}

function renderSearchInfo(comparison: SearchComparison, embedMs: number) {
  const parts: string[] = [];
  parts.push(`Embed: ${embedMs.toFixed(0)}ms`);
  parts.push(`TQ search: <span class="fast">${comparison.tqTimeMs.toFixed(1)}ms</span>`);
  if (comparison.bruteTimeMs !== null) {
    parts.push(`Brute: ${comparison.bruteTimeMs.toFixed(1)}ms`);
  }
  if (comparison.recallAtK !== null) {
    const pct = (comparison.recallAtK * 100).toFixed(0);
    parts.push(`Recall@10: <span class="fast">${pct}%</span>`);
  }
  document.getElementById("search-info")!.innerHTML = parts.join(" &middot; ");
}

function renderResults(comparison: SearchComparison) {
  const container = document.getElementById("results")!;
  const bruteSet = comparison.bruteResults
    ? new Set(comparison.bruteResults.map((r) => r.index))
    : null;

  let html = '<div class="results-columns">';

  // TurboQuant column
  html += '<div class="results-col">';
  html += '<div class="col-header tq">TurboQuant (compressed)</div>';
  for (let i = 0; i < comparison.tqResults.length; i++) {
    const r = comparison.tqResults[i];
    const inBrute = bruteSet ? bruteSet.has(r.index) : true;
    const matchClass = inBrute ? "match" : "miss";
    html += `
      <div class="result-card ${matchClass}">
        <div class="result-rank">#${i + 1}</div>
        <div class="result-text">${escapeHtml(truncate(r.passage, 200))}</div>
        <div class="result-meta">
          <span class="score">score: ${r.score.toFixed(4)}</span>
          ${inBrute ? '<span class="match-badge">in brute-force top-10</span>' : '<span class="miss-badge">not in brute-force top-10</span>'}
        </div>
      </div>`;
  }
  html += "</div>";

  // Brute-force column
  if (comparison.bruteResults) {
    const tqSet = new Set(comparison.tqResults.map((r) => r.index));
    html += '<div class="results-col">';
    html += '<div class="col-header brute">Brute-force (uncompressed)</div>';
    for (let i = 0; i < comparison.bruteResults.length; i++) {
      const r = comparison.bruteResults[i];
      const inTq = tqSet.has(r.index);
      const matchClass = inTq ? "match" : "miss";
      html += `
        <div class="result-card ${matchClass}">
          <div class="result-rank">#${i + 1}</div>
          <div class="result-text">${escapeHtml(truncate(r.passage, 200))}</div>
          <div class="result-meta">
            <span class="score">score: ${r.score.toFixed(4)}</span>
            ${inTq ? '<span class="match-badge">in TQ top-10</span>' : '<span class="miss-badge">not in TQ top-10</span>'}
          </div>
        </div>`;
    }
    html += "</div>";
  }

  html += "</div>";
  container.innerHTML = html;
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function truncate(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen) + "...";
}

function renderSuggestedQueries(onSelect: (query: string) => void) {
  const container = document.getElementById("suggested-queries")!;
  container.innerHTML = SUGGESTED_QUERIES.map(
    (q) => `<button class="suggestion">${escapeHtml(q)}</button>`,
  ).join("");
  container.querySelectorAll(".suggestion").forEach((btn, i) => {
    btn.addEventListener("click", () => onSelect(SUGGESTED_QUERIES[i]));
  });
}

async function main() {
  showLoading("Loading search index...");
  const data = await loadSearchData((msg) => showLoading(msg));

  showLoading("Loading embedding model...");
  await initEmbedder((msg) => showLoading(msg));

  renderIndexStats(data);
  hideLoading();

  const input = document.getElementById("query-input") as HTMLInputElement;

  async function doSearch(queryText: string) {
    if (!queryText.trim()) return;

    const embedStart = performance.now();
    const queryVec = await embedQuery(queryText);
    const embedMs = performance.now() - embedStart;

    const comparison = search(queryVec, data, 10);
    renderSearchInfo(comparison, embedMs);
    renderResults(comparison);
  }

  // Debounced search on input
  let debounceTimer: ReturnType<typeof setTimeout>;
  input.addEventListener("input", () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => doSearch(input.value), 400);
  });

  // Enter key
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      clearTimeout(debounceTimer);
      doSearch(input.value);
    }
  });

  // Suggested queries
  renderSuggestedQueries((query) => {
    input.value = query;
    doSearch(query);
  });

  // Run initial search with default value
  if (input.value) {
    doSearch(input.value);
  }

  console.log("Vector search demo ready!", {
    vectors: data.numVectors,
    dim: data.dim,
    rawMB: (data.rawSizeBytes / 1e6).toFixed(1),
    compressedMB: (data.compressedSizeBytes / 1e6).toFixed(1),
  });
}

main().catch((err) => {
  console.error("Search demo error:", err);
  const text = document.getElementById("loading-text")!;
  text.textContent = `Error: ${err.message}`;
  text.style.color = "#f44336";
});
