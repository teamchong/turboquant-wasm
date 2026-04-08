/**
 * Gemma 4 E2B browser demo — ORT + TurboQuant unified WASM.
 * Single binary: ORT runs inference, TQ compresses KV cache in-place.
 */

import { initOrt, type OrtExports } from "./ort-glue.js";

const $ = (s: string) => document.querySelector(s)!;
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

function setEnabled(on: boolean) {
  inputEl.disabled = !on;
  sendBtn.disabled = !on;
}

async function main() {
  statusEl.textContent = "Loading ORT+TQ WASM (17 MB)...";

  try {
    ort = await initOrt();
    statusEl.textContent = "ORT initialized. Ready to load model.";
    statusEl.classList.add("ready");
    setEnabled(true);
    inputEl.focus();
    addMsg("system", "ORT+TQ WASM loaded (17 MB, Zig-compiled, zero Emscripten). Ready for model.");
  } catch (e) {
    statusEl.textContent = `ORT init error: ${(e as Error).message}`;
    statusEl.classList.add("error");
    console.error("ORT init error:", e);
  }
}

sendBtn.addEventListener("click", () => {
  const text = inputEl.value.trim();
  if (!text || !ort) return;
  inputEl.value = "";
  addMsg("user", text);
  addMsg("assistant", "[Model loading + inference pipeline in progress]");
});

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendBtn.click();
});

main();
