/**
 * Burner AI — Gemma 4 E2B in browser with TQ-compressed KV cache.
 * LiteRT CC API + XNNPack + WebGPU. Compiled with Zig, zero Emscripten.
 *
 * Flow: page load → WASM init → fetch model → loadModel(ptr, size) →
 *       compileModel → ready for inference.
 */

import {
  initLiteRt, loadModel, createEnvironment, compileModel,
  type LiteRtExports,
} from "./litert-glue.js";

// MobileNet V1 quantized (4.1MB) to verify pipeline, then switch to Gemma 4 E2B
const MODEL_URL = "/test-model.tflite";

const $ = (s: string) => document.querySelector(s)!;
const statusEl = $("#status") as HTMLElement;
const messagesEl = $("#messages") as HTMLElement;
const inputEl = $("#input") as HTMLInputElement;
const sendBtn = $("#send") as HTMLButtonElement;
const statSpeed = $("#stat-speed") as HTMLElement;
const statCtx = $("#stat-ctx") as HTMLElement;

let lm: LiteRtExports | null = null;
let compiledModel = 0;

function addMsg(role: "user" | "assistant" | "system", text: string): HTMLElement {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

function formatBytes(b: number): string {
  if (b < 1024) return `${b} B`;
  if (b < 1e6) return `${(b / 1024).toFixed(1)} KB`;
  if (b < 1e9) return `${(b / 1e6).toFixed(1)} MB`;
  return `${(b / 1e9).toFixed(2)} GB`;
}

/**
 * Download model to OPFS cache, then stream directly into WASM heap.
 * Never holds the full model in JS — reads in chunks from OPFS File,
 * writes each chunk into WASM linear memory via HEAPU8.set.
 * Returns the WASM pointer and size.
 */
async function fetchAndLoadModel(): Promise<{ ptr: number; size: number }> {
  if (!lm) throw new Error("WASM not initialized");
  const root = await navigator.storage.getDirectory();
  const fileName = "test-model.tflite";

  // Check OPFS cache
  let fileSize = 0;
  try {
    const existing = await root.getFileHandle(fileName);
    const file = await existing.getFile();
    if (file.size > 100_000_000) {
      fileSize = file.size;
      statusEl.textContent = `Model cached (${formatBytes(fileSize)})`;
    }
  } catch { /* not cached */ }

  // Download if not cached
  if (!fileSize) {
    const response = await fetch(MODEL_URL);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const total = Number(response.headers.get("content-length") || 0);
    const fileHandle = await root.getFileHandle(fileName, { create: true });
    const writable = await fileHandle.createWritable();
    const reader = response.body!.getReader();
    let loaded = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      await writable.write(value);
      loaded += value.length;
      if (total > 0) {
        statusEl.textContent = `Downloading Gemma 4 E2B: ${((loaded / total) * 100).toFixed(0)}% (${formatBytes(loaded)})`;
      }
    }
    await writable.close();
    fileSize = loaded;
  }

  // Allocate WASM memory for the model
  statusEl.textContent = `Loading model into WASM (${formatBytes(fileSize)})...`;
  const wasmPtr = lm.wasm_malloc(fileSize);
  if (!wasmPtr) throw new Error("wasm_malloc failed for model");

  // Stream from OPFS into WASM memory in chunks (never hold full model in JS)
  const file = await (await root.getFileHandle(fileName)).getFile();
  const CHUNK_SIZE = 16 * 1024 * 1024; // 16MB chunks
  let offset = 0;
  for (let start = 0; start < fileSize; start += CHUNK_SIZE) {
    const end = Math.min(start + CHUNK_SIZE, fileSize);
    const blob = file.slice(start, end);
    const chunk = new Uint8Array(await blob.arrayBuffer());
    new Uint8Array(lm.memory.buffer, wasmPtr + offset, chunk.length).set(chunk);
    offset += chunk.length;
    statusEl.textContent = `Loading model: ${((offset / fileSize) * 100).toFixed(0)}%`;
  }

  return { ptr: wasmPtr, size: fileSize };
}

async function main() {
  // Step 1: Init WASM
  statusEl.textContent = "Loading WASM runtime (0.8 MB)...";
  try {
    lm = await initLiteRt();
  } catch (e) {
    statusEl.textContent = `WASM error: ${(e as Error).message}`;
    statusEl.classList.add("error");
    console.error(e);
    return;
  }

  // Step 2: Fetch model directly into WASM heap
  let modelPtr: number;
  let modelSize: number;
  try {
    statusEl.textContent = "Fetching model...";
    const response = await fetch(MODEL_URL);
    const modelData = new Uint8Array(await response.arrayBuffer());
    modelSize = modelData.byteLength;
    console.log("[model] fetched", modelSize, "bytes");
    modelPtr = lm.wasm_malloc(modelSize);
    new Uint8Array(lm.memory.buffer, modelPtr, modelSize).set(modelData);
  } catch (e) {
    statusEl.textContent = `Model load failed: ${(e as Error).message}`;
    statusEl.classList.add("error");
    console.error(e);
    return;
  }

  // Step 3: Parse + compile model
  statusEl.textContent = "Parsing model...";
  try {
    const env = createEnvironment(lm);

    console.log("[model] ptr =", modelPtr, "size =", modelSize);
    console.log("[model] calling LiteRtCreateModelFromBuffer...");
    const modelOut = lm.wasm_malloc(4);
    let status: number;
    let model: number;
    try {
      status = lm.LiteRtCreateModelFromBuffer(modelPtr, modelSize, modelOut);
      model = new DataView(lm.memory.buffer).getUint32(modelOut, true);
      console.log("[model] status =", status, "model =", model);
    } catch (e) {
      console.error("[model] LiteRtCreateModelFromBuffer threw:", e);
      lm.wasm_free(modelOut);
      throw e;
    }
    lm.wasm_free(modelOut);
    lm.wasm_free(modelPtr);
    if (status !== 0 || !model) throw new Error(`LiteRtCreateModelFromBuffer failed: status=${status}`);

    statusEl.textContent = "Compiling model...";
    compiledModel = compileModel(lm, env, model);

    statusEl.textContent = "Gemma 4 E2B ready";
    statusEl.classList.add("ready");
    inputEl.focus();
  } catch (e) {
    statusEl.textContent = `Model error: ${(e as Error).message}`;
    statusEl.classList.add("error");
    console.error(e);
  }
}

sendBtn.addEventListener("click", () => {
  const text = inputEl.value.trim();
  if (!text || !lm || !compiledModel) return;
  inputEl.value = "";
  addMsg("user", text);
  addMsg("assistant", "[Inference pipeline in progress — need to wire up tokenizer + run loop]");
});

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendBtn.click();
});

main();
