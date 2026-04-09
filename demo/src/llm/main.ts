/**
 * Burner AI — Gemma in browser with TQ-compressed KV cache.
 * LiteRT-LM + TurboQuant unified WASM. No cloud, no data leaks.
 */

import {
  initLiteRt, registerModelFile,
  writeString, readString, writeInputData,
  type LiteRtExports,
} from "./litert-glue.js";

const $ = (s: string) => document.querySelector(s)!;
const statusEl = $("#status") as HTMLElement;
const messagesEl = $("#messages") as HTMLElement;
const inputEl = $("#input") as HTMLInputElement;
const sendBtn = $("#send") as HTMLButtonElement;
const statSpeed = $("#stat-speed") as HTMLElement;
const statCtx = $("#stat-ctx") as HTMLElement;
const statKV = $("#stat-kv") as HTMLElement;

let lm: LiteRtExports | null = null;
let engine = 0;
let session = 0;

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

function setEnabled(on: boolean) {
  inputEl.disabled = !on;
  sendBtn.disabled = !on;
}

async function loadModel(buffer: ArrayBuffer, name: string) {
  if (!lm) return;

  const modelPath = `/${name}`;
  registerModelFile(modelPath, buffer);

  statusEl.textContent = "Creating engine...";
  const pathPtr = writeString(lm, modelPath);
  const cpuPtr = writeString(lm, "cpu");
  const nullPtr = 0;

  const settings = lm.litert_lm_engine_settings_create(pathPtr, cpuPtr, nullPtr, nullPtr);
  lm.litert_lm_engine_settings_set_max_num_tokens(settings, 2048);
  lm.litert_lm_engine_settings_enable_benchmark(settings);

  lm.wasm_free(pathPtr);
  lm.wasm_free(cpuPtr);

  statusEl.textContent = "Initializing model (this may take a moment)...";

  try {
    engine = lm.litert_lm_engine_create(settings);
    if (!engine) throw new Error("Engine creation returned null");

    const config = lm.litert_lm_session_config_create();
    lm.litert_lm_session_config_set_max_output_tokens(config, 512);
    session = lm.litert_lm_engine_create_session(engine, config);
    lm.litert_lm_session_config_delete(config);

    statusEl.textContent = `Model loaded: ${name}`;
    statusEl.classList.add("ready");
    addMsg("system", `Model "${name}" loaded. Type a message to start chatting.`);
    setEnabled(true);
    inputEl.focus();
  } catch (e) {
    statusEl.textContent = `Model load error: ${(e as Error).message}`;
    statusEl.classList.add("error");
    console.error("Model load error:", e);
  }

  lm.litert_lm_engine_settings_delete(settings);
}

function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || !lm || !session) return;
  inputEl.value = "";
  addMsg("user", text);

  const assistantDiv = addMsg("assistant", "Thinking...");

  try {
    const inputPtr = writeInputData(lm, text);
    const responses = lm.litert_lm_session_generate_content(session, inputPtr, 1);

    if (responses) {
      const textPtr = lm.litert_lm_responses_get_response_text_at(responses, 0);
      const responseText = textPtr ? readString(lm, textPtr) : "[empty response]";
      assistantDiv.textContent = responseText;

      // Show benchmark stats
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
}

async function main() {
  setEnabled(false);

  // 1. Load WASM runtime
  statusEl.textContent = "Loading LiteRT-LM + TQ WASM runtime (2.8 MB gzip)...";
  try {
    lm = await initLiteRt();
    statusEl.textContent = "WASM loaded. Drop a .litertlm model file to begin.";
    addMsg("system", "LiteRT-LM + TQ WASM loaded (9.5 MB, Zig-compiled). Drop a model file to start.");
  } catch (e) {
    statusEl.textContent = `WASM init error: ${(e as Error).message}`;
    statusEl.classList.add("error");
    console.error("WASM init error:", e);
    return;
  }

  // 2. Drag & drop model loading
  document.body.addEventListener("dragover", (e) => { e.preventDefault(); });
  document.body.addEventListener("drop", async (e) => {
    e.preventDefault();
    const file = e.dataTransfer?.files[0];
    if (!file) return;
    statusEl.textContent = `Loading model: ${file.name} (${formatBytes(file.size)})...`;
    const buffer = await file.arrayBuffer();
    await loadModel(buffer, file.name);
  });
}

sendBtn.addEventListener("click", sendMessage);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendMessage();
});

main();
