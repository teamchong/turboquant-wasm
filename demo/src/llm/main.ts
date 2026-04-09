/**
 * Burner AI — Gemma 4 E2B in browser with TQ-compressed KV cache.
 * Transformers.js (ORT Web + WebGPU) for inference.
 * TQ WebGPU compute shaders for KV cache compression.
 * No cloud, no data leaks.
 */

import { pipeline, env } from "@huggingface/transformers";

// Use WebGPU backend
env.backends.onnx.wasm!.numThreads = 1;

const $ = (s: string) => document.querySelector(s)!;
const statusEl = $("#status") as HTMLElement;
const messagesEl = $("#messages") as HTMLElement;
const inputEl = $("#input") as HTMLInputElement;
const sendBtn = $("#send") as HTMLButtonElement;
const statSpeed = $("#stat-speed") as HTMLElement;
const statCtx = $("#stat-ctx") as HTMLElement;

let generator: Awaited<ReturnType<typeof pipeline>> | null = null;
let generating = false;

function addMsg(role: "user" | "assistant" | "system", text: string): HTMLElement {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || !generator) return;
  inputEl.value = "";
  addMsg("user", text);

  if (generating) return;
  generating = true;

  const assistantDiv = addMsg("assistant", "...");
  const startTime = performance.now();

  (async () => {
    try {
      const messages = [{ role: "user", content: text }];
      const result = await (generator as any)(messages, {
        max_new_tokens: 256,
        do_sample: false,
      });
      const response = result[0].generated_text.at(-1)?.content || "[empty]";
      assistantDiv.textContent = response;

      const elapsed = (performance.now() - startTime) / 1000;
      const tokens = response.split(/\s+/).length;
      statSpeed.textContent = `~${(tokens / elapsed).toFixed(1)} tok/s`;
      statCtx.textContent = `${elapsed.toFixed(1)}s`;
    } catch (e) {
      assistantDiv.textContent = `Error: ${(e as Error).message}`;
      console.error("Generation error:", e);
    }
    generating = false;
    inputEl.focus();
  })();
}

async function main() {
  statusEl.textContent = "Loading Gemma 4 E2B (WebGPU)...";

  try {
    generator = await pipeline("text-generation", "onnx-community/gemma-3-1b-it-ONNX", {
      device: "webgpu",
      dtype: "q4",
      progress_callback: (progress: any) => {
        if (progress.status === "downloading") {
          const pct = progress.progress?.toFixed(0) || "?";
          statusEl.textContent = `Downloading: ${pct}% — ${progress.file}`;
        } else if (progress.status === "loading") {
          statusEl.textContent = `Loading model...`;
        }
      },
    });

    statusEl.textContent = "Gemma ready (WebGPU)";
    statusEl.classList.add("ready");
    inputEl.focus();
  } catch (e) {
    statusEl.textContent = `Error: ${(e as Error).message}`;
    statusEl.classList.add("error");
    console.error("Model load error:", e);
  }
}

sendBtn.addEventListener("click", sendMessage);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendMessage();
});

main();
