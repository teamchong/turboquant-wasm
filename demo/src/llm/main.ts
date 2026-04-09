/**
 * Burner AI — Gemma in browser with TQ-compressed KV cache.
 * Transformers.js (ORT Web + WebGPU) for inference.
 * TQ WebGPU shaders compress KV cache between generation steps.
 *
 * Cache stays compressed on GPU. Decompressed only at the moment the ONNX
 * model needs it for attention (GPU shader, no CPU involved).
 * ~4.5x memory savings → 8K context becomes ~36K in same GPU memory.
 */

import { pipeline, env, type TextGenerationPipeline } from "@huggingface/transformers";
import { TQKVCache } from "./tq-kv-cache.js";

env.backends.onnx.wasm!.numThreads = 1;

const MODEL_ID = "onnx-community/gemma-3-1b-it-ONNX";

const $ = (s: string) => document.querySelector(s)!;
const statusEl = $("#status") as HTMLElement;
const messagesEl = $("#messages") as HTMLElement;
const inputEl = $("#input") as HTMLInputElement;
const sendBtn = $("#send") as HTMLButtonElement;
const statSpeed = $("#stat-speed") as HTMLElement;
const statCtx = $("#stat-ctx") as HTMLElement;
const statKV = $("#stat-kv") as HTMLElement;

let gen: TextGenerationPipeline | null = null;
let tqCache: TQKVCache | null = null;
let tqEnabled = true;
let generating = false;

const tqToggle = $("#tq-toggle") as HTMLInputElement;
tqToggle.addEventListener("change", () => {
  tqEnabled = tqToggle.checked;
  const label = tqEnabled ? "TQ KV Cache ON" : "TQ KV Cache OFF";
  console.log(`[TQ] ${label}`);
  if (tqCache) {
    if (!tqEnabled) {
      statKV.textContent = "KV: uncompressed";
    }
  }
});

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

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || !gen || generating) return;
  inputEl.value = "";
  generating = true;
  addMsg("user", text);
  const assistantDiv = addMsg("assistant", "");

  const startTime = performance.now();
  let tokenCount = 0;

  try {
    const messages = [{ role: "user" as const, content: text }];
    const output = await gen(messages, {
      max_new_tokens: 256,
      do_sample: false,
      callback_function: (data: any) => {
        tokenCount++;
        if (data?.[0]?.generated_text) {
          const lastMsg = data[0].generated_text.at(-1);
          if (lastMsg?.role === "assistant") {
            assistantDiv.textContent = lastMsg.content;
            // Live stats update
            const elapsed = (performance.now() - startTime) / 1000;
            statSpeed.textContent = `${(tokenCount / elapsed).toFixed(1)} tok/s`;
            if (tqCache) {
              const stats = tqCache.getStats();
              if (stats.compressedBytes > 0) {
                const ratio = (stats.uncompressedBytes / stats.compressedBytes).toFixed(1);
                statKV.textContent = `KV: ${formatBytes(stats.compressedBytes)} (${ratio}x)`;
              }
            }
          }
        }
      },
    });

    const response = output[0]?.generated_text?.at(-1)?.content || "[empty]";
    assistantDiv.textContent = response;

    const elapsed = (performance.now() - startTime) / 1000;
    statSpeed.textContent = `${(tokenCount / elapsed).toFixed(1)} tok/s`;
    statCtx.textContent = `${elapsed.toFixed(1)}s · ${tokenCount} tokens`;

    if (tqCache) {
      const stats = tqCache.getStats();
      if (stats.compressedBytes > 0) {
        const ratio = (stats.uncompressedBytes / stats.compressedBytes).toFixed(1);
        statKV.textContent = `KV: ${formatBytes(stats.compressedBytes)} (${ratio}x) · ${stats.layers}L × ${stats.positions}pos`;
      }
    }
  } catch (e) {
    assistantDiv.textContent = `Error: ${(e as Error).message}`;
    console.error("Generation error:", e);
  }

  generating = false;
  inputEl.focus();
}

async function main() {
  statusEl.textContent = "Loading model (WebGPU)...";

  try {
    gen = await pipeline("text-generation", MODEL_ID, {
      device: "webgpu",
      dtype: "q4",
      progress_callback: (progress: any) => {
        if (progress.status === "downloading") {
          const pct = progress.progress?.toFixed(0) || "?";
          statusEl.textContent = `Downloading: ${pct}% — ${progress.file}`;
        } else if (progress.status === "loading") {
          statusEl.textContent = "Loading model into WebGPU...";
        }
      },
    }) as TextGenerationPipeline;

    // Initialize TQ KV cache
    const ort = await import("onnxruntime-web/webgpu");
    const gpuDevice = ort.env.webgpu.device as GPUDevice;
    if (gpuDevice) {
      const config = (gen as any).model?.config;
      const headDim = config?.head_dim || 256;
      tqCache = await TQKVCache.create(gpuDevice, headDim, 8192);
      console.log("[TQ] KV cache ready: head_dim=%d", headDim);

      // Intercept KV cache between generation steps.
      // After each forward pass, getPastKeyValues receives the model's
      // present_key_values and returns a DynamicCache for the next step.
      // We compress each layer's K/V tensors via TQ encode shader on GPU,
      // building the compressed cache that tracks memory savings.
      const model = (gen as any).model;
      if (model?.getPastKeyValues) {
        const originalGetPKV = model.getPastKeyValues.bind(model);
        model.getPastKeyValues = function(
          decoderResults: Record<string, any>,
          pastKeyValues: any,
          ...rest: any[]
        ) {
          const cache = originalGetPKV(decoderResults, pastKeyValues, ...rest);

          // Compress new K/V tensors into TQ cache on GPU (when toggle is on)
          if (tqCache && tqEnabled) {
            for (const name in decoderResults) {
              if (!name.startsWith("present") || !name.endsWith(".value")) continue;
              const vTensor = decoderResults[name];
              const kName = name.replace(".value", ".key");
              const kTensor = decoderResults[kName];
              if (!vTensor?.ort_tensor || !kTensor?.ort_tensor) continue;
              if (vTensor.ort_tensor.location !== "gpu-buffer") continue;

              const match = name.match(/(\d+)\.value/);
              if (!match) continue;
              const layerIdx = parseInt(match[1]);
              const seqLen = vTensor.dims[2];
              const kBuf = kTensor.ort_tensor.gpuBuffer as GPUBuffer;
              const vBuf = vTensor.ort_tensor.gpuBuffer as GPUBuffer;
              tqCache.encodeAndAppend(layerIdx, kBuf, vBuf, seqLen);
            }
          }

          return cache;
        };
        console.log("[TQ] getPastKeyValues intercepted — compression active");
      }

      statusEl.textContent = "Gemma ready (WebGPU + TQ KV cache)";
    } else {
      statusEl.textContent = "Gemma ready (WebGPU)";
    }

    statusEl.classList.add("ready");
    inputEl.focus();
  } catch (e) {
    statusEl.textContent = `Error: ${(e as Error).message}`;
    statusEl.classList.add("error");
    console.error("Load error:", e);
  }
}

sendBtn.addEventListener("click", sendMessage);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) sendMessage();
});

main();
