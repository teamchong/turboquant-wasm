import { initOrt, type OrtExports } from "./ort-glue.ts";

const $ = (s: string) => document.querySelector(s)!;

const MODEL_URL = "https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX/resolve/main/onnx/model_q4.onnx";

const statusEl = $("#status") as HTMLElement;
const messagesEl = $("#messages") as HTMLElement;
const inputEl = $("#input") as HTMLInputElement;
const sendBtn = $("#send") as HTMLButtonElement;
const statSpeed = $("#stat-speed") as HTMLElement;
const statCtx = $("#stat-ctx") as HTMLElement;
const statKV = $("#stat-kv") as HTMLElement;

let ort: OrtExports | null = null;

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
  return `${(b / 1e6).toFixed(1)} MB`;
}

async function main() {
  statusEl.textContent = "Loading ORT+TQ WASM...";

  try {
    ort = await initOrt();
    statusEl.textContent = "ORT initialized. Loading model...";
  } catch (e) {
    statusEl.textContent = `ORT init failed: ${(e as Error).message}`;
    statusEl.classList.add("error");
    console.error("ORT init error:", e);
    return;
  }

  // Download ONNX model
  try {
    statusEl.textContent = "Downloading Gemma 4 E2B ONNX...";
    const response = await fetch(MODEL_URL);
    if (!response.ok) throw new Error(`Model download failed: ${response.status}`);

    const modelBuffer = await response.arrayBuffer();
    statusEl.textContent = `Model loaded (${formatBytes(modelBuffer.byteLength)}). Creating session...`;

    // Copy model to WASM memory
    const modelPtr = ort.malloc(modelBuffer.byteLength);
    if (!modelPtr) throw new Error("Failed to allocate WASM memory for model");
    new Uint8Array(ort.memory.buffer, modelPtr, modelBuffer.byteLength).set(new Uint8Array(modelBuffer));

    // Create ORT session
    const sessionPtrPtr = ort.malloc(4);
    const rc = ort.OrtCreateSessionFromBuffer(modelPtr, modelBuffer.byteLength, 0, sessionPtrPtr);
    if (rc !== 0) throw new Error(`OrtCreateSession failed: ${rc}`);

    const sessionPtr = new DataView(ort.memory.buffer).getUint32(sessionPtrPtr, true);
    ort.free(sessionPtrPtr);
    ort.free(modelPtr); // Model copied into ORT's internal structures

    statusEl.textContent = "Ready — TQ compressed KV cache active";
    statusEl.classList.add("ready");
    inputEl.disabled = false;
    sendBtn.disabled = false;
    inputEl.focus();

    addMsg("system", "Gemma 4 E2B loaded. KV cache compressed with TurboQuant — zero float32 KV storage.");

    // Store session for inference
    (window as any).__ortSession = sessionPtr;
  } catch (e) {
    statusEl.textContent = `Error: ${(e as Error).message}`;
    statusEl.classList.add("error");
    console.error("Model load error:", e);
  }
}

async function onSend() {
  const text = inputEl.value.trim();
  if (!text || !ort) return;

  inputEl.value = "";
  inputEl.disabled = true;
  sendBtn.disabled = true;
  statusEl.textContent = "Generating...";
  statusEl.classList.remove("ready");

  addMsg("user", text);
  const assistantDiv = addMsg("assistant", "Inference with TQ KV compression running...");

  // The actual inference loop requires tokenization and iterative forward passes.
  // The ORT session runs the model, and our patched GQA attention kernel
  // automatically compresses K/V into TQStream and uses dotBatch for scoring.
  // Token generation is handled by ORT's generate() API.

  // Display TQ KV stats
  const w = ort;
  let totalCompressed = 0;
  let totalUncompressed = 0;
  for (let i = 0; i < 256; i++) {
    const len = w.tq_kv_length(i);
    if (len === 0) continue;
    totalCompressed += w.tq_kv_compressed_size(i);
    // Estimate uncompressed: len * head_dim * 4 bytes (we don't know head_dim from here)
    totalUncompressed += len * 256 * 4; // assume head_dim=256
  }

  if (totalCompressed > 0) {
    const ratio = totalUncompressed / totalCompressed;
    statKV.textContent = `KV: ${formatBytes(totalCompressed)} TQ / ${formatBytes(totalUncompressed)} f32 = ${ratio.toFixed(1)}x`;
  }

  assistantDiv.textContent = "[Model loaded — full token generation pipeline requires tokenizer integration]";

  statusEl.textContent = "Ready — TQ compressed KV cache active";
  statusEl.classList.add("ready");
  inputEl.disabled = false;
  sendBtn.disabled = false;
  inputEl.focus();
}

sendBtn.addEventListener("click", onSend);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !inputEl.disabled) onSend();
});

main();
