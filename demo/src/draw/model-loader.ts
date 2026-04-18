/**
 * GGUF Model Loader for WebGPU — OPFS Streaming
 *
 * Downloads the model to Origin Private File System (persistent, no eviction).
 * Reads tensors via random access from disk — peak CPU ~5MB, not 3.1GB.
 *
 * Flow:
 *   1. Stream download → OPFS file (0 JS heap)
 *   2. Parse header: read first 1MB from OPFS → WASM
 *   3. Per-tensor upload: OPFS random read → GPU buffer (~5MB peak)
 */

export const GGMLType = {
  F32: 0, F16: 1, Q4_0: 2, Q4_1: 3,
  Q5_0: 6, Q5_1: 7, Q8_0: 8, Q8_1: 9,
  Q2_K: 10, Q3_K: 11, Q4_K: 12, Q5_K: 13, Q6_K: 14, Q8_K: 15,
  BF16: 30,
} as const;

export interface TensorMeta {
  name: string;
  dims: number[];
  type: number;
  offset: number;
  size: number;
  gpuBuffer?: GPUBuffer;
}

export interface ModelMetadata {
  architecture: string;
  nLayers: number;
  nHeads: number;
  nKvHeads: number;
  hiddenSize: number;
  intermediateSize: number;
  vocabSize: number;
  contextLength: number;
  headDim: number;
  ropeFreqBase: number;
}

export interface LoadedModel {
  metadata: ModelMetadata;
  tensors: Map<string, TensorMeta>;
  file: File;  // OPFS-backed file for random access reads
}

// WASM module instance
let wasmInstance: WebAssembly.Instance | null = null;

async function loadWasm(): Promise<WebAssembly.Instance> {
  if (wasmInstance) return wasmInstance;
  // import.meta.env.BASE_URL is "/" in dev and "/turboquant-wasm/" in
  // production so the same fetch URL works on localhost + GitHub Pages
  // subfolder + any other base path vite is configured for. Was previously
  // hardcoded to "/gguf-parser.wasm" which 404'd on Pages.
  const resp = await fetch(import.meta.env.BASE_URL + "gguf-parser.wasm");
  const bytes = await resp.arrayBuffer();
  const result = await WebAssembly.instantiate(bytes, {});
  wasmInstance = result.instance;
  return wasmInstance;
}

function getExports(instance: WebAssembly.Instance) {
  return instance.exports as Record<string, (...args: any[]) => any> & {
    memory: WebAssembly.Memory;
  };
}

/** Get OPFS root directory. */
async function opfsRoot(): Promise<FileSystemDirectoryHandle> {
  return navigator.storage.getDirectory();
}

/** Filename for a GGUF URL in OPFS. */
function opfsFilename(url: string): string {
  // Use the last path segment as filename
  return url.split("/").pop() || "model.gguf";
}

/**
 * Download GGUF model to OPFS (persistent, 0 JS heap).
 * Returns the OPFS-backed File for random access reads.
 */
export async function fetchGGUF(
  url: string,
  onProgress?: (loaded: number, total: number) => void,
): Promise<File> {
  const root = await opfsRoot();
  const filename = opfsFilename(url);
  // Sidecar: a tiny marker file holding the expected byte size of the GGUF.
  // Only written after the download stream finishes cleanly. On startup we
  // check both files exist AND that the GGUF size matches the sidecar. If
  // either is missing, or sizes disagree, the previous download was
  // interrupted (tab closed mid-stream, network drop) and we redownload.
  // Previously a partial download became a ghost "cache hit" next visit,
  // parseGGUF threw a cryptic header error, and users had to wipe manually.
  const markerName = `${filename}.complete`;

  // Check if already downloaded AND the sidecar marker confirms completion.
  // Two valid cache-hit paths:
  //   A. Sidecar present and size matches → trust and use.
  //   B. Sidecar absent (legacy download from before this commit) → HEAD
  //      the URL to learn the real size. If file.size matches, backfill
  //      the sidecar and use. If it doesn't, delete the partial.
  try {
    const existing = await root.getFileHandle(filename);
    const file = await existing.getFile();
    let expected = 0;
    try {
      const markerHandle = await root.getFileHandle(markerName);
      const marker = await markerHandle.getFile();
      expected = Number((await marker.text()).trim());
    } catch {
      // Legacy path — no sidecar. Use HEAD to learn expected size.
      try {
        const head = await fetch(url, { method: "HEAD" });
        if (head.ok) expected = Number(head.headers.get("content-length") || 0);
      } catch { /* HEAD failed, treat as unknown */ }
      if (file.size > 0 && expected > 0 && file.size === expected) {
        const markerHandle = await root.getFileHandle(markerName, { create: true });
        const w = await markerHandle.createWritable();
        await w.write(String(file.size));
        await w.close();
        console.log(`[model-loader] OPFS hit (backfilled marker):`, filename, (file.size / 1e9).toFixed(2), "GB");
      }
    }
    if (file.size > 0 && expected > 0 && file.size === expected) {
      console.log("[model-loader] OPFS hit:", filename, (file.size / 1e9).toFixed(2), "GB");
      onProgress?.(file.size, file.size);
      return file;
    }
    if (expected > 0) {
      console.warn(`[model-loader] OPFS file present but size mismatch (${file.size} vs expected ${expected}) — re-downloading`);
      await root.removeEntry(filename).catch(() => {});
      await root.removeEntry(markerName).catch(() => {});
    }
  } catch { /* either file or marker missing — fall through to download */ }

  console.log("[model-loader] Downloading:", url);
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to fetch: ${resp.status} ${resp.statusText}`);

  const contentLength = Number(resp.headers.get("content-length") || 0);
  const reader = resp.body!.getReader();

  // Create OPFS file and get writable stream
  const fileHandle = await root.getFileHandle(filename, { create: true });
  const writable = await fileHandle.createWritable();
  let loaded = 0;

  try {
    // Stream directly to OPFS — chunks never accumulate in JS heap
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      await writable.write(value);
      loaded += value.byteLength;
      onProgress?.(loaded, contentLength);
    }
    await writable.close();
  } catch (e) {
    // Mid-stream failure — leave no poisoned partial file behind.
    try { await writable.abort(); } catch { /* best effort */ }
    await root.removeEntry(filename).catch(() => {});
    throw new Error(`Download failed at ${loaded} / ${contentLength || "?"} bytes: ${(e as Error).message}`);
  }

  // Verify total bytes match the server's reported size before writing the
  // completion marker. If reader closed early without an exception (rare —
  // happens with some buggy proxies), `loaded` will be short.
  if (contentLength > 0 && loaded !== contentLength) {
    await root.removeEntry(filename).catch(() => {});
    throw new Error(`Download ended early: got ${loaded} / ${contentLength} bytes. Retry.`);
  }

  const file = await fileHandle.getFile();
  // Write completion marker only on success.
  const markerHandle = await root.getFileHandle(markerName, { create: true });
  const markerWritable = await markerHandle.createWritable();
  await markerWritable.write(String(file.size));
  await markerWritable.close();

  console.log("[model-loader] Saved to OPFS:", (file.size / 1e9).toFixed(2), "GB");
  return file;
}

/**
 * Parse GGUF header from OPFS file.
 * Reads only first 1MB — the rest stays on disk.
 */
export async function parseGGUF(file: File): Promise<LoadedModel> {
  const wasm = await loadWasm();
  const exports = getExports(wasm);
  const mem = exports.memory;

  // Read only the header
  // Gemma 4 header includes tokenizer vocab (262K strings ≈ 5MB). 16MB covers any model.
  const headerSize = Math.min(file.size, 16 * 1024 * 1024);
  const headerSlice = file.slice(0, headerSize);
  const headerBuf = await headerSlice.arrayBuffer();

  const wasmOffset = exports.gguf_alloc(headerSize) as number;
  if (wasmOffset === 0) throw new Error("WASM allocation failed");
  new Uint8Array(mem.buffer, wasmOffset, headerSize).set(new Uint8Array(headerBuf));

  const result = exports.gguf_parse(wasmOffset, headerSize);
  if (result !== 0) throw new Error("Failed to parse GGUF header");

  // Read metadata
  const metaPtr = exports.gguf_get_metadata() as number;
  if (!metaPtr) throw new Error("No metadata");

  const metaView = new DataView(mem.buffer, metaPtr);
  const archBytes = new Uint8Array(mem.buffer, metaPtr, 64);
  const archLen = new DataView(mem.buffer, metaPtr + 64).getUint32(0, true);
  const architecture = new TextDecoder().decode(archBytes.slice(0, archLen));

  const metadata: ModelMetadata = {
    architecture,
    nLayers: metaView.getUint32(68, true),
    nHeads: metaView.getUint32(72, true),
    nKvHeads: metaView.getUint32(76, true),
    hiddenSize: metaView.getUint32(80, true),
    intermediateSize: metaView.getUint32(84, true),
    vocabSize: metaView.getUint32(88, true),
    contextLength: metaView.getUint32(92, true),
    headDim: metaView.getUint32(96, true),
    ropeFreqBase: metaView.getFloat32(100, true),
  };

  // Read tensor info
  const tensorCount = exports.gguf_tensor_count() as number;
  const tensors = new Map<string, TensorMeta>();

  for (let i = 0; i < tensorCount; i++) {
    const namePtr = exports.gguf_get_tensor_name_ptr(i) as number;
    const nameLen = exports.gguf_get_tensor_name_len(i) as number;
    const name = new TextDecoder().decode(new Uint8Array(mem.buffer, namePtr, nameLen));

    const nDims = exports.gguf_get_tensor_ndims(i) as number;
    const dims: number[] = [];
    for (let d = 0; d < nDims; d++) {
      dims.push(Number(exports.gguf_get_tensor_dim(i, d)));
    }

    tensors.set(name, {
      name, dims,
      type: exports.gguf_get_tensor_type(i) as number,
      offset: Number(exports.gguf_get_tensor_offset(i)),
      size: Number(exports.gguf_get_tensor_size(i)),
    });
  }

  exports.gguf_free();
  exports.gguf_dealloc(wasmOffset, headerSize);

  console.log(`[model-loader] Parsed: ${metadata.architecture}, ${metadata.nLayers} layers, ${tensorCount} tensors`);
  return { metadata, tensors, file };
}

/**
 * Upload tensors to GPU one at a time from OPFS file.
 * Random access reads: file.slice(offset, offset+size).arrayBuffer()
 * Peak CPU: size of largest single tensor (~5MB).
 */
/**
 * Tensor upload strategies. We default to `chunked` which turned out fastest
 * on the benchmark — see `benchmarkTensorUpload` below for the rest.
 *
 *   serial      — original: one slice().arrayBuffer() at a time
 *   parallel    — up to CONCURRENCY slice reads in flight at once
 *   chunked     — one big slice per ~256 MB chunk, serve tensors from memory
 *   syncAccess  — createSyncAccessHandle + read() in worker (fastest OPFS API)
 */
export type UploadStrategy = "serial" | "parallel" | "chunked" | "syncAccess";

export async function uploadTensorsToGPU(
  device: GPUDevice,
  model: LoadedModel,
  onProgress?: (uploaded: number, total: number) => void,
  filter?: (name: string) => boolean,
  strategy: UploadStrategy = "syncAccess",
  concurrency = 8,
  chunkBytes = 256 * 1024 * 1024,
): Promise<void> {
  const tensors: TensorMeta[] = [];
  for (const [name, tensor] of model.tensors) {
    if (filter && !filter(name)) continue;
    tensors.push(tensor);
    void name;
  }
  const total = tensors.length;
  const t0 = performance.now();
  let uploadedBytes = 0;
  let uploaded = 0;

  const mkBuf = (size: number): GPUBuffer =>
    device.createBuffer({
      size: Math.ceil(size / 4) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

  const uploadFromView = (tensor: TensorMeta, view: Uint8Array) => {
    const gpuBuf = mkBuf(tensor.size);
    device.queue.writeBuffer(gpuBuf, 0, view as Uint8Array<ArrayBuffer>);
    tensor.gpuBuffer = gpuBuf;
    uploadedBytes += tensor.size;
    uploaded++;
    onProgress?.(uploaded, total);
  };

  if (strategy === "serial") {
    for (const tensor of tensors) {
      const buf = await model.file.slice(tensor.offset, tensor.offset + tensor.size).arrayBuffer();
      uploadFromView(tensor, new Uint8Array(buf));
    }
  } else if (strategy === "parallel") {
    let nextIdx = 0;
    const inflight = new Map<number, Promise<{ idx: number; tensor: TensorMeta; buf: ArrayBuffer }>>();
    const start = (idx: number) => {
      const tensor = tensors[idx];
      const slice = model.file.slice(tensor.offset, tensor.offset + tensor.size);
      inflight.set(idx, slice.arrayBuffer().then(buf => ({ idx, tensor, buf })));
    };
    for (; nextIdx < Math.min(concurrency, total); nextIdx++) start(nextIdx);
    while (inflight.size > 0) {
      const { idx, tensor, buf } = await Promise.race(inflight.values());
      inflight.delete(idx);
      uploadFromView(tensor, new Uint8Array(buf));
      if (nextIdx < total) { start(nextIdx); nextIdx++; }
    }
  } else if (strategy === "chunked") {
    // Sort tensors by byte offset so chunks stay contiguous in the OPFS file.
    const sorted = [...tensors].sort((a, b) => a.offset - b.offset);
    let i = 0;
    while (i < sorted.length) {
      // Greedily extend the chunk until it exceeds chunkBytes OR the next
      // tensor would not fit. A huge single tensor gets its own chunk.
      const chunkStart = sorted[i].offset;
      let chunkEnd = sorted[i].offset + sorted[i].size;
      let j = i + 1;
      while (j < sorted.length && (sorted[j].offset + sorted[j].size - chunkStart) <= chunkBytes) {
        chunkEnd = sorted[j].offset + sorted[j].size;
        j++;
      }
      const chunkBuf = new Uint8Array(await model.file.slice(chunkStart, chunkEnd).arrayBuffer());
      for (let k = i; k < j; k++) {
        const t = sorted[k];
        const offsetInChunk = t.offset - chunkStart;
        uploadFromView(t, chunkBuf.subarray(offsetInChunk, offsetInChunk + t.size));
      }
      i = j;
    }
  } else if (strategy === "syncAccess") {
    // FileSystemSyncAccessHandle — worker-only OPFS API that avoids the per-
    // slice arrayBuffer overhead entirely. Benchmark: ~0.66 GB/s on M1 Chrome,
    // ~35% faster than the parallel-slice path. Falls back to `parallel` if
    // the API is missing (e.g., main-thread call or older browser) OR if it
    // throws — which happens on reload when a previous tab was killed mid-
    // download and left a stuck createWritable handle on the file. Chrome
    // OPFS doesn't release the lock until the origin's storage is wiped OR
    // the other handle is closed (impossible if the owning tab is dead).
    // The parallel path uses `file.slice()` which doesn't take the lock, so
    // it's the only way to recover without asking the user to wipe.
    const root = await navigator.storage.getDirectory();
    const fileName = (model.file as any).name || "model.gguf";
    const handle = await root.getFileHandle(fileName);
    const createSync = (handle as any).createSyncAccessHandle;
    if (typeof createSync !== "function") {
      console.warn("[worker] createSyncAccessHandle not available — falling back to parallel strategy");
      return uploadTensorsToGPU(device, model, onProgress, filter, "parallel", concurrency, chunkBytes);
    }
    let sync: any;
    try {
      sync = await createSync.call(handle);
    } catch (e) {
      console.warn(`[worker] createSyncAccessHandle threw (${(e as Error).message}) — falling back to parallel strategy. If this persists, click Stop + Wipe All Data.`);
      return uploadTensorsToGPU(device, model, onProgress, filter, "parallel", concurrency, chunkBytes);
    }
    try {
      // Sort tensors by offset so reads are linear in file order (best for OPFS).
      const sorted = [...tensors].sort((a, b) => a.offset - b.offset);
      for (const t of sorted) {
        const buf = new Uint8Array(t.size);
        sync.read(buf, { at: t.offset });
        uploadFromView(t, buf);
      }
    } finally {
      sync.close();
    }
  }

  const elapsed = (performance.now() - t0) / 1000;
  const gb = uploadedBytes / 1e9;
  console.log(`[worker] tensor upload: strategy=${strategy} ${uploaded} tensors ${gb.toFixed(2)} GB in ${elapsed.toFixed(1)}s — ${(gb / elapsed).toFixed(2)} GB/s`);
}

/**
 * Bench all strategies against the current model. Each run uploads and then
 * destroys the GPU buffers before the next run, so memory stays bounded.
 * Logs a table of (strategy, seconds, GB/s) so we can pick the winner.
 */
export async function benchmarkTensorUpload(device: GPUDevice, model: LoadedModel): Promise<void> {
  const strategies: Array<{ name: string; run: () => Promise<void> }> = [
    { name: "serial",             run: () => uploadTensorsToGPU(device, model, undefined, undefined, "serial") },
    { name: "parallel-8",         run: () => uploadTensorsToGPU(device, model, undefined, undefined, "parallel", 8) },
    { name: "parallel-32",        run: () => uploadTensorsToGPU(device, model, undefined, undefined, "parallel", 32) },
    { name: "chunked-64MB",       run: () => uploadTensorsToGPU(device, model, undefined, undefined, "chunked", 8, 64 * 1024 * 1024) },
    { name: "chunked-256MB",      run: () => uploadTensorsToGPU(device, model, undefined, undefined, "chunked", 8, 256 * 1024 * 1024) },
    { name: "chunked-1GB",        run: () => uploadTensorsToGPU(device, model, undefined, undefined, "chunked", 8, 1024 * 1024 * 1024) },
    { name: "syncAccess",         run: () => uploadTensorsToGPU(device, model, undefined, undefined, "syncAccess") },
  ];
  const results: Array<{ name: string; ms: number; gbps: number; error?: string }> = [];
  for (const { name, run } of strategies) {
    try {
      // Destroy previously created buffers so each run starts from a clean slate.
      for (const [, t] of model.tensors) { t.gpuBuffer?.destroy(); t.gpuBuffer = undefined; }
      const t0 = performance.now();
      await run();
      const ms = performance.now() - t0;
      // Total bytes = sum of tensor sizes
      let totalBytes = 0;
      for (const [, t] of model.tensors) totalBytes += t.size;
      const gbps = (totalBytes / 1e9) / (ms / 1000);
      results.push({ name, ms, gbps });
    } catch (e) {
      results.push({ name, ms: -1, gbps: 0, error: (e as Error).message });
    }
  }
  console.log("[worker] upload bench:");
  for (const r of results) {
    if (r.error) console.log(`  ${r.name.padEnd(16)} FAIL ${r.error}`);
    else console.log(`  ${r.name.padEnd(16)} ${(r.ms / 1000).toFixed(2).padStart(6)}s  ${r.gbps.toFixed(2)} GB/s`);
  }
}

/**
 * Wipe OPFS files. Recursive so sub-directories get cleared too —
 * default removeEntry throws on non-empty dirs.
 */
export async function wipeModels(): Promise<void> {
  const root = await opfsRoot();
  for await (const [name] of (root as any).entries()) {
    try {
      await root.removeEntry(name, { recursive: true });
    } catch (e) {
      console.warn(`[model-loader] failed to remove ${name}:`, (e as Error).message);
    }
  }
  console.log("[model-loader] OPFS wiped");
}
