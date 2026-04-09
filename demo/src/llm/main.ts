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

// Gemma 4 E2B web model (.task format = tflite + metadata)
const MODEL_URL = "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it-web.task";

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
 * Download model to OPFS cache, return as Uint8Array.
 * Streams to disk during download, reads back after.
 */
async function fetchModel(): Promise<Uint8Array> {
  const root = await navigator.storage.getDirectory();
  const fileName = "gemma-4-E2B-it-web.task";

  // Check OPFS cache
  try {
    const existing = await root.getFileHandle(fileName);
    const file = await existing.getFile();
    if (file.size > 100_000_000) {
      statusEl.textContent = `Loading cached model (${formatBytes(file.size)})...`;
      return new Uint8Array(await file.arrayBuffer());
    }
  } catch { /* not cached */ }

  // Stream download to OPFS
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

  // Read back
  statusEl.textContent = "Loading model into memory...";
  const file = await (await root.getFileHandle(fileName)).getFile();
  return new Uint8Array(await file.arrayBuffer());
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

  // Step 2: Fetch model
  let modelData: Uint8Array;
  try {
    modelData = await fetchModel();
  } catch (e) {
    statusEl.textContent = `Download failed: ${(e as Error).message}`;
    statusEl.classList.add("error");
    console.error(e);
    return;
  }

  // Step 3: Load + compile model (matches @litertjs/core flow)
  statusEl.textContent = "Loading model into WASM...";
  try {
    const env = createEnvironment(lm);
    const model = loadModel(lm, modelData);
    modelData = null!; // Release JS reference — model copied into WASM heap

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
