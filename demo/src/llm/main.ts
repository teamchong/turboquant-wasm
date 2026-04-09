/**
 * Burner AI — Gemma 4 E2B in browser with TQ-compressed KV cache.
 * LiteRT-LM + TurboQuant unified WASM. No cloud, no data leaks.
 *
 * Architecture:
 *   Main thread: UI + model download to OPFS
 *   Worker thread: WASM + inference with sync OPFS file access (zero-copy)
 */

const MODEL_URL = "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it-web.task";
const MODEL_NAME = "gemma-4-E2B-it-web.task";

const $ = (s: string) => document.querySelector(s)!;
const statusEl = $("#status") as HTMLElement;
const messagesEl = $("#messages") as HTMLElement;
const inputEl = $("#input") as HTMLInputElement;
const sendBtn = $("#send") as HTMLButtonElement;
const statSpeed = $("#stat-speed") as HTMLElement;
const statCtx = $("#stat-ctx") as HTMLElement;

let worker: Worker | null = null;
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
 * Download model to OPFS. Streams to disk — never holds full model in JS heap.
 * Returns true if model is ready in OPFS.
 */
async function ensureModelInOPFS(): Promise<boolean> {
  const root = await navigator.storage.getDirectory();

  // Check cache
  try {
    const existing = await root.getFileHandle(MODEL_NAME);
    const file = await existing.getFile();
    if (file.size > 100_000_000) {
      statusEl.textContent = `Model cached (${formatBytes(file.size)})`;
      return true;
    }
  } catch { /* not cached */ }

  // Stream download to OPFS
  const response = await fetch(MODEL_URL);
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  const total = Number(response.headers.get("content-length") || 0);

  const fileHandle = await root.getFileHandle(MODEL_NAME, { create: true });
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
      statusEl.textContent = `Downloading: ${formatBytes(loaded)}`;
    }
  }
  await writable.close();
  return true;
}

let pendingResolve: ((msg: any) => void) | null = null;

function handleWorkerMessage(e: MessageEvent) {
  const msg = e.data;
  switch (msg.type) {
    case "status":
      statusEl.textContent = msg.text;
      if (msg.text.includes("ready")) {
        ready = true;
        statusEl.classList.add("ready");
        inputEl.focus();
      }
      break;
    case "result":
      if (pendingResolve) {
        pendingResolve(msg);
        pendingResolve = null;
      }
      generating = false;
      inputEl.focus();
      break;
    case "error":
      statusEl.textContent = `Error: ${msg.text}`;
      statusEl.classList.add("error");
      console.error("Worker error:", msg.text);
      if (pendingResolve) {
        pendingResolve(msg);
        pendingResolve = null;
      }
      generating = false;
      break;
  }
}

function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || !worker) return;
  inputEl.value = "";
  addMsg("user", text);

  if (!ready) {
    addMsg("system", "Model is still loading.");
    return;
  }
  if (generating) {
    addMsg("system", "Generating...");
    return;
  }

  generating = true;
  const assistantDiv = addMsg("assistant", "...");

  const resultPromise = new Promise<any>((resolve) => { pendingResolve = resolve; });
  worker.postMessage({ type: "generate", text });

  resultPromise.then((msg) => {
    if (msg.type === "result") {
      assistantDiv.textContent = msg.text || "[empty]";
      if (msg.tokPerSec > 0) {
        statSpeed.textContent = `${msg.tokPerSec.toFixed(1)} tok/s`;
      }
      if (msg.ttft > 0) {
        statCtx.textContent = `TTFT: ${(msg.ttft * 1000).toFixed(0)}ms`;
      }
    } else {
      assistantDiv.textContent = `Error: ${msg.text}`;
    }
  });
}

async function main() {
  // Step 1: Download model to OPFS (streams to disk, no JS heap copy)
  statusEl.textContent = "Checking model cache...";
  try {
    await ensureModelInOPFS();
  } catch (e) {
    statusEl.textContent = `Download failed: ${(e as Error).message}`;
    statusEl.classList.add("error");
    console.error("Download error:", e);
    return;
  }

  // Step 2: Start worker — WASM + inference with sync OPFS access
  statusEl.textContent = "Starting inference engine...";
  worker = new Worker(
    new URL("./litert-worker.ts", import.meta.url),
    { type: "module" }
  );
  worker.onmessage = handleWorkerMessage;
  worker.onerror = (e) => {
    statusEl.textContent = `Worker error: ${e.message}`;
    statusEl.classList.add("error");
  };
  worker.postMessage({ type: "init", modelFileName: MODEL_NAME });
}

sendBtn.addEventListener("click", sendMessage);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendMessage();
});

main();
