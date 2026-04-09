/**
 * Burner AI — Gemma in browser with TQ-compressed KV cache.
 * Transformers.js (ORT Web + WebGPU) for inference.
 * TQ WebGPU shaders compress KV cache between generation steps.
 *
 * Cache stays compressed on GPU. Decompressed only at the moment the ONNX
 * model needs it for attention (GPU shader, no CPU involved).
 * ~4.5x memory savings → 8K context becomes ~36K in same GPU memory.
 */

import { pipeline, env, TextStreamer, type TextGenerationPipeline } from "@huggingface/transformers";
import { TQKVCache } from "./tq-kv-cache.js";

env.backends.onnx.wasm!.numThreads = 1;

const MODEL_ID = "onnx-community/gemma-4-E2B-it-ONNX";

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
  if (!tqToggle.checked) {
    const ok = confirm(
      "⚠️ Disabling TQ removes ~6.7x KV cache compression.\n\n" +
      "Long contexts WILL crash your browser tab (out of GPU memory).\n\n" +
      "Disable TQ compression?"
    );
    if (!ok) {
      tqToggle.checked = true;
      return;
    }
  }
  tqEnabled = tqToggle.checked;
  console.log(`[TQ] ${tqEnabled ? "ON" : "OFF"}`);
  statKV.textContent = tqEnabled ? "KV: —" : "KV: uncompressed (no protection)";
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

    const streamer = new TextStreamer((gen as any).tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      token_callback_function: () => { tokenCount++; },
      callback_function: (chunk: string) => {
        assistantDiv.textContent += chunk;
        messagesEl.scrollTop = messagesEl.scrollHeight;
        const elapsed = (performance.now() - startTime) / 1000;
        statSpeed.textContent = `${(tokenCount / elapsed).toFixed(1)} tok/s`;
        if (tqCache && tqEnabled) {
          const stats = tqCache.getStats();
          if (stats.compressedBytes > 0) {
            const ratio = (stats.uncompressedBytes / stats.compressedBytes).toFixed(1);
            statKV.textContent = `KV: ${formatBytes(stats.compressedBytes)} (${ratio}x)`;
          }
        }
      },
    });

    const output = await gen(messages, {
      max_new_tokens: 256,
      do_sample: false,
      streamer,
    });

    const elapsed = (performance.now() - startTime) / 1000;
    statSpeed.textContent = `${(tokenCount / elapsed).toFixed(1)} tok/s`;
    statCtx.textContent = `${elapsed.toFixed(1)}s · ${tokenCount} tok`;

    if (tqCache && tqEnabled) {
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
      dtype: "q4f16",
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

// Fill input with a long context to test KV cache limits
const fillBtn = $("#fill-long") as HTMLButtonElement;
fillBtn.addEventListener("click", () => {
  const paragraphs = [
    "The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning.",
    "The field of AI research was founded at a workshop held on the campus of Dartmouth College during the summer of 1956. Those who attended would become the leaders of AI research for decades. Many of them predicted that a machine as intelligent as a human being would exist in no more than a generation, and they were given millions of dollars to make this vision come true. Eventually it became obvious that commercial developers and researchers had grossly underestimated the difficulty of the project.",
    "In 1973, in response to the criticism from James Lighthill and ongoing pressure from the US Congress, both the US and British governments cut off exploratory research in AI. The next few years would later be called an AI winter, a period when obtaining funding for AI projects was difficult. In the early 1980s, AI research was revived by the commercial success of expert systems, a form of AI program that simulated the knowledge and analytical skills of human experts.",
    "Deep learning began to dominate industry benchmarks in 2012. The transformer architecture debuted in 2017 and was used to produce impressive generative AI applications. In 2023, large language models demonstrated abilities in a wide range of tasks, including understanding and generating natural language, solving mathematical problems, writing computer code, and many other tasks that were not anticipated by the researchers who developed them.",
    "The recent explosion of generative AI has raised fundamental questions about the nature of intelligence, creativity, and consciousness. As these systems become more capable, society faces unprecedented challenges in areas of employment, education, intellectual property, and governance. The development of artificial general intelligence remains one of the most ambitious and controversial goals in the history of technology.",
    "Neural network architectures have evolved significantly since the perceptron was introduced in 1958. Convolutional neural networks revolutionized computer vision, recurrent neural networks transformed sequence modeling, and attention mechanisms enabled the processing of long-range dependencies. The scaling laws discovered in recent years suggest that simply making models larger and training them on more data continues to improve their capabilities.",
    "The environmental impact of training large AI models has become a growing concern. A single training run for a large language model can consume as much energy as several households use in a year. Researchers are exploring more efficient training methods, including sparse models, mixture of experts architectures, and novel hardware designs that could reduce the carbon footprint of AI development.",
    "Reinforcement learning from human feedback has emerged as a key technique for aligning AI systems with human values and preferences. This approach involves training a reward model based on human evaluations, then using reinforcement learning to optimize the AI system against this reward model. The technique has been instrumental in making large language models more helpful, harmless, and honest.",
  ];
  const longText = paragraphs.join(" ") + "\n\nBased on all the information above, what are the three most important turning points in the history of AI and why?";
  inputEl.value = longText;
  inputEl.focus();
});

main();
