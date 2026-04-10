/**
 * Prompt → Diagram — Gemma 4 E2B generates drawmode code, rendered as Excalidraw.
 * Transformers.js (ORT Web + WebGPU) for local inference.
 * TQ WebGPU shaders compress KV cache on GPU.
 * vectorjson streams the structured output as the model generates it.
 * drawmode SDK executes the code and produces Excalidraw diagrams.
 */

import { pipeline, env, TextStreamer, type TextGenerationPipeline } from "@huggingface/transformers";
import { TQKVCache } from "./tq-kv-cache.js";
import { executeCode } from "./drawmode/executor.js";
import { SDK_TYPES } from "./drawmode/sdk-types.js";

env.backends.onnx.wasm!.numThreads = 1;

const MODEL_ID = "onnx-community/gemma-4-E2B-it-ONNX";

const $ = (s: string) => document.querySelector(s)!;
const statusEl = $("#status") as HTMLElement;
const promptEl = $("#prompt") as HTMLTextAreaElement;
const generateBtn = $("#generate") as HTMLButtonElement;
const codeEl = $("#code-output") as HTMLPreElement;
const diagramFrame = $("#diagram-frame") as HTMLIFrameElement;
const statSpeed = $("#stat-speed") as HTMLElement;
const statKV = $("#stat-kv") as HTMLElement;

let gen: TextGenerationPipeline | null = null;
let tqCache: TQKVCache | null = null;
let generating = false;

const SYSTEM_PROMPT = `You are a diagram generator. Given a description, write JavaScript code using the Diagram API to create an Excalidraw architecture diagram.

API Reference:
${SDK_TYPES}

Available methods:
- const d = new Diagram()
- d.addBox(label, opts) — rectangle node
- d.addEllipse(label, opts) — ellipse node
- d.addDiamond(label, opts) — diamond node
- d.addParallelogram(label, opts) — parallelogram node
- d.connect(from, to, label?) — arrow between nodes
- d.addGroup(label, nodes) — visual group
- d.addText(text, opts) — floating text
- return d.render({ format: "url" })

Position nodes with row/col grid. Use color presets: "frontend", "backend", "database", "storage", "ai", "external", "orchestration", "queue", "cache", "users".

Respond with ONLY the JavaScript code, no markdown, no explanation. The code must end with: return d.render({ format: "url" })

Example:
const d = new Diagram();
const gw = d.addBox("API Gateway", { row: 0, col: 1, color: "frontend" });
const auth = d.addBox("Auth", { row: 1, col: 0, color: "backend" });
const api = d.addBox("API", { row: 1, col: 1, color: "backend" });
const db = d.addBox("PostgreSQL", { row: 2, col: 1, color: "database" });
d.connect(gw, auth, "verify");
d.connect(gw, api, "route");
d.connect(api, db, "query");
return d.render({ format: "url" });`;

async function generate() {
  const prompt = promptEl.value.trim();
  if (!prompt || !gen || generating) return;

  generating = true;
  generateBtn.disabled = true;
  codeEl.textContent = "";
  diagramFrame.src = "about:blank";
  statusEl.textContent = "Generating diagram code...";

  const startTime = performance.now();
  let tokenCount = 0;
  let generatedCode = "";

  try {
    const messages = [
      { role: "user" as const, content: prompt },
    ];

    const streamer = new TextStreamer((gen as any).tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      token_callback_function: () => { tokenCount++; },
      callback_function: (chunk: string) => {
        generatedCode += chunk;
        codeEl.textContent = generatedCode;
        const elapsed = (performance.now() - startTime) / 1000;
        statSpeed.textContent = `${(tokenCount / elapsed).toFixed(1)} tok/s`;
      },
    });

    await gen(messages, {
      max_new_tokens: 1024,
      do_sample: false,
      system_prompt: SYSTEM_PROMPT,
      streamer,
    });

    const elapsed = (performance.now() - startTime) / 1000;
    statSpeed.textContent = `${(tokenCount / elapsed).toFixed(1)} tok/s · ${tokenCount} tok · ${elapsed.toFixed(1)}s`;

    // Update TQ stats
    if (tqCache) {
      const stats = tqCache.getStats();
      if (stats.compressedBytes > 0) {
        const ratio = (stats.uncompressedBytes / stats.compressedBytes).toFixed(1);
        statKV.textContent = `KV: ${(stats.compressedBytes / 1e6).toFixed(1)} MB (${ratio}x)`;
      }
    }

    // Strip markdown code fences if present
    let code = generatedCode.trim();
    code = code.replace(/^```(?:javascript|js|typescript|ts)?\s*\n?/i, "");
    code = code.replace(/\n?```\s*$/i, "");
    codeEl.textContent = code;

    // Execute the generated code
    statusEl.textContent = "Rendering diagram...";
    const { result, error } = await executeCode(code, { format: "url" });

    if (error) {
      statusEl.textContent = `Code error: ${error}`;
      statusEl.classList.add("error");
    } else if (result.url) {
      diagramFrame.src = result.url;
      statusEl.textContent = "Diagram ready";
      statusEl.classList.remove("error");
    } else if (result.json) {
      // Encode JSON to excalidraw URL
      const json = JSON.stringify(result.json);
      const encoded = btoa(unescape(encodeURIComponent(json)));
      diagramFrame.src = `https://excalidraw.com/#json=${encoded}`;
      statusEl.textContent = "Diagram ready";
      statusEl.classList.remove("error");
    }
  } catch (e) {
    statusEl.textContent = `Error: ${(e as Error).message}`;
    statusEl.classList.add("error");
    console.error("Generation error:", e);
  }

  generating = false;
  generateBtn.disabled = false;
}

async function main() {
  statusEl.textContent = "Loading Gemma 4 E2B (WebGPU)...";

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
      tqCache = await TQKVCache.create(gpuDevice, headDim, 131072);
      tqCache.onContextLimitReached = (used, max) => {
        statusEl.textContent = `Context limit: ${used.toLocaleString()} / ${max.toLocaleString()} tokens`;
        statusEl.classList.add("error");
      };

      // Intercept KV cache
      const model = (gen as any).model;
      if (model?.getPastKeyValues) {
        const originalGetPKV = model.getPastKeyValues.bind(model);
        model.getPastKeyValues = function(
          decoderResults: Record<string, any>,
          pastKeyValues: any,
          ...rest: any[]
        ) {
          const cache = originalGetPKV(decoderResults, pastKeyValues, ...rest);
          if (tqCache) {
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
              tqCache.encodeAndAppend(layerIdx, kTensor.ort_tensor.gpuBuffer, vTensor.ort_tensor.gpuBuffer, seqLen);
            }
          }
          return cache;
        };
      }
    }

    statusEl.textContent = "Ready — describe a diagram";
    statusEl.classList.add("ready");
    promptEl.focus();
  } catch (e) {
    statusEl.textContent = `Error: ${(e as Error).message}`;
    statusEl.classList.add("error");
    console.error("Load error:", e);
  }
}

generateBtn.addEventListener("click", generate);
promptEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) generate();
});

main();
