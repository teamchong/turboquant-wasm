export interface ComparisonStats {
  originalFileSize: number;
  compressedFileSize: number;
  numGaussians: number;
  shDegree: number;
  decodeMs: number;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function transferTime(bytes: number, mbps: number): string {
  const seconds = (bytes * 8) / (mbps * 1_000_000);
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
  return `${seconds.toFixed(1)}s`;
}

export function renderLeftStats(
  el: HTMLElement,
  data: ComparisonStats,
): void {
  el.innerHTML = `
    Transfer: <span class="highlight">${formatBytes(data.originalFileSize)}</span><br>
    ${data.numGaussians.toLocaleString()} Gaussians, SH degree ${data.shDegree}<br>
    @ 50 Mbps: ${transferTime(data.originalFileSize, 50)}<br>
    @ 10 Mbps: ${transferTime(data.originalFileSize, 10)}
  `;
}

export function renderRightStats(
  el: HTMLElement,
  data: ComparisonStats,
): void {
  const ratio = data.originalFileSize / data.compressedFileSize;
  const saved = data.originalFileSize - data.compressedFileSize;

  el.innerHTML = `
    Transfer: <span class="highlight">${formatBytes(data.compressedFileSize)}</span>
    <span class="green">(${ratio.toFixed(1)}x smaller)</span><br>
    Decode: ${(data.decodeMs / 1000).toFixed(2)}s, saves ${formatBytes(saved)} bandwidth<br>
    @ 50 Mbps: ${transferTime(data.compressedFileSize, 50)} + ${(data.decodeMs / 1000).toFixed(1)}s decode<br>
    @ 10 Mbps: ${transferTime(data.compressedFileSize, 10)} + ${(data.decodeMs / 1000).toFixed(1)}s decode
  `;
}

export function showRatioBadge(el: HTMLElement, ratio: number): void {
  el.textContent = `${ratio.toFixed(1)}x smaller`;
  el.classList.add("visible");
}
