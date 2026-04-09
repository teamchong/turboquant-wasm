/**
 * Burner AI — Gemma 4 E2B in browser with TQ-compressed KV cache.
 * LiteRT-LM + TurboQuant unified WASM. No cloud, no data leaks.
 *
 * Flow: page load → WASM init → model auto-download (2GB) → ready to chat.
 * Input is always enabled. Messages queue until model is ready.
 */

import {
  initLiteRt, registerModelFile,
  writeString, readString, writeInputData,
  type LiteRtExports,
} from "./litert-glue.js";

const MODEL_URL = "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it-web.task";
const MODEL_NAME = "gemma-4-E2B-it-web.task";

const $ = (s: string) => document.querySelector(s)!;
const statusEl = $("#status") as HTMLElement;
const messagesEl = $("#messages") as HTMLElement;
const inputEl = $("#input") as HTMLInputElement;
const sendBtn = $("#send") as HTMLButtonElement;
const statSpeed = $("#stat-speed") as HTMLElement;
const statCtx = $("#stat-ctx") as HTMLElement;
const statKV = $("#stat-kv") as HTMLElement;

let lm: LiteRtExports | null = null;
let session = 0;
let ready = false;
let generating = false;

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
 * Download model to OPFS (Origin Private File System), then read back.
 * Streams directly to disk — never holds 2GB in JS memory.
 * On subsequent visits, serves from OPFS cache (instant).
 */
async function downloadModelToOPFS(url: string, filename: string): Promise<ArrayBuffer> {
  const root = await navigator.storage.getDirectory();

  // Check cache first
  try {
    const existing = await root.getFileHandle(filename);
    const file = await existing.getFile();
    if (file.size > 100_000_000) { // >100MB = valid cached model
      statusEl.textContent = `Loading cached model (${formatBytes(file.size)})...`;
      return await file.arrayBuffer();
    }
  } catch { /* not cached */ }

  // Download with streaming write to OPFS
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  const total = Number(response.headers.get("content-length") || 0);

  const fileHandle = await root.getFileHandle(filename, { create: true });
  const writable = await fileHandle.createWritable();
  const reader = response.body!.getReader();
  let loaded = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    await writable.write(value);
    loaded += value.length;
    if (total > 0) {
      const pct = ((loaded / total) * 100).toFixed(0);
      statusEl.textContent = `Downloading Gemma 4 E2B: ${pct}% (${formatBytes(loaded)} / ${formatBytes(total)})`;
    } else {
      statusEl.textContent = `Downloading Gemma 4 E2B: ${formatBytes(loaded)}`;
    }
  }
  await writable.close();

  // Read back from OPFS
  statusEl.textContent = "Reading model from storage...";
  const file = await (await root.getFileHandle(filename)).getFile();
  return await file.arrayBuffer();
}

function sendMessage() {
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = "";
  addMsg("user", text);

  if (!ready || !lm || !session) {
    addMsg("system", "Model is still loading. Your message will be processed when ready.");
    return;
  }

  if (generating) {
    addMsg("system", "Already generating a response. Please wait.");
    return;
  }

  generating = true;
  const assistantDiv = addMsg("assistant", "...");

  try {
    const inputPtr = writeInputData(lm, text);
    const responses = lm.litert_lm_session_generate_content(session, inputPtr, 1);

    if (responses) {
      const textPtr = lm.litert_lm_responses_get_response_text_at(responses, 0);
      const responseText = textPtr ? readString(lm, textPtr) : "[empty response]";
      assistantDiv.textContent = responseText;

      const bench = lm.litert_lm_session_get_benchmark_info(session);
      if (bench) {
        const ttft = lm.litert_lm_benchmark_info_get_time_to_first_token(bench);
        const decSpeed = lm.litert_lm_benchmark_info_get_decode_tokens_per_sec_at(bench, 0);
        statSpeed.textContent = `${decSpeed.toFixed(1)} tok/s`;
        statCtx.textContent = `TTFT: ${(ttft * 1000).toFixed(0)}ms`;
        lm.litert_lm_benchmark_info_delete(bench);
      }

      lm.litert_lm_responses_delete(responses);
    } else {
      assistantDiv.textContent = "[No response from model]";
    }

    lm.wasm_free(inputPtr);
  } catch (e) {
    assistantDiv.textContent = `Error: ${(e as Error).message}`;
    console.error("Inference error:", e);
  }

  generating = false;
  inputEl.focus();
}

async function main() {
  // Step 1: Load WASM runtime
  statusEl.textContent = "Loading WASM runtime...";
  try {
    lm = await initLiteRt();
    statusEl.textContent = "WASM loaded. Downloading model...";
  } catch (e) {
    statusEl.textContent = `WASM error: ${(e as Error).message}`;
    statusEl.classList.add("error");
    console.error("WASM init error:", e);
    return;
  }

  // Step 2: Download Gemma 4 E2B model (streams to OPFS, cached on disk)
  let modelBuffer: ArrayBuffer;
  try {
    modelBuffer = await downloadModelToOPFS(MODEL_URL, MODEL_NAME);
  } catch (e) {
    statusEl.textContent = `Download failed: ${(e as Error).message}`;
    statusEl.classList.add("error");
    console.error("Model download error:", e);
    return;
  }

  // Step 3: Initialize engine + session
  statusEl.textContent = "Initializing model...";
  try {
    const modelPath = `/${MODEL_NAME}`;
    registerModelFile(modelPath, modelBuffer);

    const pathPtr = writeString(lm, modelPath);
    const cpuPtr = writeString(lm, "cpu");

    const settings = lm.litert_lm_engine_settings_create(pathPtr, cpuPtr, 0, 0);
    lm.litert_lm_engine_settings_set_max_num_tokens(settings, 2048);
    lm.litert_lm_engine_settings_enable_benchmark(settings);
    lm.wasm_free(pathPtr);
    lm.wasm_free(cpuPtr);

    const engine = lm.litert_lm_engine_create(settings);
    if (!engine) throw new Error("Engine creation returned null");

    const config = lm.litert_lm_session_config_create();
    lm.litert_lm_session_config_set_max_output_tokens(config, 512);
    session = lm.litert_lm_engine_create_session(engine, config);
    lm.litert_lm_session_config_delete(config);
    lm.litert_lm_engine_settings_delete(settings);

    ready = true;
    statusEl.textContent = "Gemma 4 E2B ready";
    statusEl.classList.add("ready");
    inputEl.focus();
  } catch (e) {
    statusEl.textContent = `Engine error: ${(e as Error).message}`;
    statusEl.classList.add("error");
    console.error("Engine init error:", e);
  }
}

sendBtn.addEventListener("click", sendMessage);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendMessage();
});

main();
