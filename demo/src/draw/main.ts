/**
 * Prompt → Diagram — Gemma 4 E2B generates drawmode code, rendered as Excalidraw.
 * Transformers.js (ORT Web + WebGPU) for local inference.
 * TQ WebGPU shaders compress KV cache on GPU.
 * vectorjson streams the structured output as the model generates it.
 * drawmode SDK executes the code and produces Excalidraw diagrams.
 */

import { pipeline, env, TextStreamer, InterruptableStoppingCriteria, type TextGenerationPipeline } from "@huggingface/transformers";
import { TurboQuant } from "turboquant-wasm";
import { executeCode } from "./drawmode/executor.js";
import { SDK_TYPES } from "./drawmode/sdk-types.js";
import { loadWasm } from "./drawmode/layout.js";
import drawmodeWasm from "./drawmode/drawmode-wasm.js";
import { mountExcalidraw, updateDiagram } from "./excalidraw-viewer.js";
import { createEditor, setCode, appendCode, getCode } from "./code-editor.js";

env.backends.onnx.wasm!.numThreads = 1;

const MODEL_ID = "onnx-community/gemma-4-E2B-it-ONNX";

const $ = (s: string) => document.querySelector(s)!;
const statusEl = $("#status") as HTMLElement;
const promptEl = $("#prompt") as HTMLTextAreaElement;
const generateBtn = $("#generate") as HTMLButtonElement;
const codeArea = $("#code-area") as HTMLElement;
const diagramContainer = $("#diagram-container") as HTMLElement;
const statSpeed = $("#stat-speed") as HTMLElement;
const statKV = $("#stat-kv") as HTMLElement;

let gen: TextGenerationPipeline | null = null;
let tq: TurboQuant | null = null;
let busy = false;
let currentCode = "";
const stopCriteria = new InterruptableStoppingCriteria();

// Compressed KV cache — stores TQ-encoded vectors per layer
const kvCache: Map<number, { keys: Uint8Array[]; values: Uint8Array[]; }> = new Map();
let kvUncompressedBytes = 0;
let kvCompressedBytes = 0;

const SYSTEM_PROMPT = `You generate Excalidraw diagrams by writing JavaScript code using the Diagram API. Respond with ONLY code, no markdown fences, no explanation.

${SDK_TYPES}

The code must end with: return d.render()

Use row/col for grid layout. Use color presets for visual clarity. Use icons, groups, diamonds, ellipses, dashed connections — whatever fits the diagram. Labels support \\n for line breaks.

Themes: "default", "sketch", "blueprint", "minimal"
Directions: "TB" (top-bottom), "LR" (left-right)
Diagram types: "architecture" (default), "sequence"

Example — architecture:
const d = new Diagram({ direction: "TB" });
const user = d.addEllipse("Users", { row: 0, col: 1, color: "users", icon: "users" });
const cdn = d.addBox("CDN", { row: 1, col: 0, color: "external", icon: "globe" });
const gw = d.addBox("API Gateway", { row: 1, col: 1, color: "frontend", icon: "api" });
const auth = d.addBox("Auth Service", { row: 2, col: 0, color: "backend", icon: "lock" });
const api = d.addBox("Core API", { row: 2, col: 1, color: "backend", icon: "server" });
const cache = d.addBox("Redis", { row: 2, col: 2, color: "cache", icon: "cache" });
const db = d.addBox("PostgreSQL", { row: 3, col: 1, color: "database", icon: "database" });
const queue = d.addBox("Message Queue", { row: 3, col: 2, color: "queue", icon: "queue" });
const worker = d.addBox("Worker", { row: 4, col: 2, color: "backend", icon: "server" });
d.connect(user, cdn, "static assets");
d.connect(user, gw, "API calls");
d.connect(gw, auth, "verify");
d.connect(gw, api, "route");
d.connect(api, cache, "read/write");
d.connect(api, db, "query");
d.connect(api, queue, "publish");
d.connect(queue, worker, "consume");
d.addGroup("Backend", [auth, api, cache]);
d.addGroup("Data", [db, queue, worker]);
return d.render();

Example — sequence:
const d = new Diagram({ type: "sequence" });
const client = d.addActor("Client");
const server = d.addActor("Server");
const db = d.addActor("Database");
d.message(client, server, "POST /login");
d.message(server, db, "SELECT user");
d.message(db, server, "user row");
d.message(server, client, "JWT token");
return d.render();

Example — flowchart with decisions:
const d = new Diagram();
const start = d.addEllipse("Start", { row: 0, col: 1, color: "frontend" });
const check = d.addDiamond("Valid?", { row: 1, col: 1, color: "orchestration" });
const yes = d.addBox("Process", { row: 2, col: 0, color: "backend" });
const no = d.addBox("Reject", { row: 2, col: 2, color: "external" });
const end = d.addEllipse("End", { row: 3, col: 1, color: "frontend" });
d.connect(start, check);
d.connect(check, yes, "yes");
d.connect(check, no, "no");
d.connect(yes, end);
d.connect(no, end);
return d.render();`;

async function generate() {
  const prompt = promptEl.value.trim();
  if (!prompt) return;

  // Cancel previous generation if running
  if (busy) {
    stopCriteria.interrupt();
    return;
  }

  busy = true;
  stopCriteria.reset();

  if (!gen) {
    generateBtn.innerHTML = '<span class="spinner"></span> Waiting for model...';
    while (!gen) {
      await new Promise(r => setTimeout(r, 500));
    }
  }
  generateBtn.innerHTML = '<span class="spinner"></span> Generating...';
  setCode("");
  statusEl.textContent = "Generating diagram code...";

  const startTime = performance.now();
  let tokenCount = 0;
  let generatedCode = "";

  try {
    // Always read latest from editor (user may have edited manually)
    currentCode = getCode();
    let userContent = prompt;
    if (currentCode) {
      userContent = `Current diagram code:\n\`\`\`\n${currentCode}\n\`\`\`\n\nModify it: ${prompt}`;
    }
    const messages = [
      { role: "system" as const, content: SYSTEM_PROMPT },
      { role: "user" as const, content: userContent },
    ];

    const streamer = new TextStreamer((gen as any).tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      token_callback_function: () => { tokenCount++; },
      callback_function: (chunk: string) => {
        generatedCode += chunk;
        appendCode(chunk);
        const elapsed = (performance.now() - startTime) / 1000;
        statSpeed.textContent = `${(tokenCount / elapsed).toFixed(1)} tok/s`;
      },
    });

    await gen(messages, {
      max_new_tokens: 1024,
      do_sample: false,
      streamer,
      stopping_criteria: stopCriteria,
    });

    const elapsed = (performance.now() - startTime) / 1000;
    statSpeed.textContent = `${(tokenCount / elapsed).toFixed(1)} tok/s · ${tokenCount} tok · ${elapsed.toFixed(1)}s`;

    // Update TQ stats
    if (kvCompressedBytes > 0) {
      const compressed = (kvCompressedBytes / 1e6).toFixed(1);
      const f16Size = (kvUncompressedBytes / 1e6).toFixed(1);
      const f32Size = (kvUncompressedBytes * 2 / 1e6).toFixed(1);
      const ratioF16 = (kvUncompressedBytes / kvCompressedBytes).toFixed(1);
      const ratioF32 = (kvUncompressedBytes * 2 / kvCompressedBytes).toFixed(1);
      statKV.textContent = `KV: ${compressed} MB (${ratioF16}x vs f16, ${ratioF32}x vs f32)`;
    } else {
      statKV.textContent = `${tokenCount} tokens`;
    }

    // Strip markdown code fences if present
    let code = generatedCode.trim();
    code = code.replace(/^```(?:javascript|js|typescript|ts)?\s*\n?/i, "");
    code = code.replace(/\n?```\s*$/i, "");
    setCode(code);

    // Execute the generated code — get JSON directly, no upload API
    statusEl.textContent = "Rendering diagram...";
    const { result, error } = await executeCode(code);

    if (error) {
      statusEl.textContent = `Code error: ${error}`;
      statusEl.classList.add("error");
    } else if (result.json) {
      currentCode = code;
      updateDiagram(result.json.elements || []);
      statusEl.textContent = "Diagram ready";
      statusEl.classList.remove("error");
    }
  } catch (e) {
    statusEl.textContent = `Error: ${(e as Error).message}`;
    statusEl.classList.add("error");
    console.error("Generation error:", e);
  }

  busy = false;
  generateBtn.textContent = "Generate Diagram";
}

async function main() {
  // Initialize code editor
  createEditor(codeArea, (code) => { currentCode = code; });

  // Load layout engine WASM (Graphviz, inlined as base64)
  await loadWasm(drawmodeWasm);

  // Mount Excalidraw viewer
  mountExcalidraw(diagramContainer);

  statusEl.textContent = "Loading Gemma 4 E2B (WebGPU)...";

  try {
    gen = await pipeline("text-generation", MODEL_ID, {
      device: "webgpu",
      dtype: "q4f16",
      progress_callback: (progress: any) => {
        if (progress.status === "progress_total" || progress.status === "progress") {
          const pct = progress.progress?.toFixed(0) || "0";
          const loaded = (progress.loaded / 1e9).toFixed(1);
          const total = (progress.total / 1e9).toFixed(1);
          statusEl.innerHTML = `<span class="spinner"></span> Downloading Gemma 4 E2B: ${pct}% (${loaded} / ${total} GB)`;
        } else if (progress.status === "loading" || progress.status === "ready") {
          statusEl.innerHTML = `<span class="spinner"></span> Loading model into WebGPU...`;
        }
      },
    }) as TextGenerationPipeline;

    // Initialize TQ for KV cache compression
    const config = (gen as any).model?.config;
    const headDim = config?.head_dim || 256;
    tq = await TurboQuant.init({ dim: headDim });
    console.log("[TQ] initialized: head_dim=%d", headDim);

    // Float16 → Float32 conversion (KV tensors are float16 on CPU)
    function f16ToF32(f16: Uint16Array): Float32Array {
      const f32 = new Float32Array(f16.length);
      const buf = new ArrayBuffer(4);
      const f32v = new Float32Array(buf);
      const u32v = new Uint32Array(buf);
      for (let i = 0; i < f16.length; i++) {
        const h = f16[i];
        const s = (h >> 15) & 1;
        const e = (h >> 10) & 0x1f;
        const m = h & 0x3ff;
        if (e === 0) {
          u32v[0] = (s << 31) | (m << 13);
          f32[i] = f32v[0];
        } else if (e === 31) {
          f32[i] = m ? NaN : (s ? -Infinity : Infinity);
        } else {
          u32v[0] = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
          f32[i] = f32v[0];
        }
      }
      return f32;
    }

    // Intercept KV cache — compress with TQ, replace originals, decompress on demand
    const { Tensor } = await import("@huggingface/transformers");
    const model = (gen as any).model;

    // Float32 → Float16 conversion
    function f32ToF16(f32: Float32Array): Uint16Array {
      const f16 = new Uint16Array(f32.length);
      for (let i = 0; i < f32.length; i++) {
        const v = f32[i];
        // Fast f32 to f16 via DataView
        const buf = new ArrayBuffer(4);
        new Float32Array(buf)[0] = v;
        const bits = new Uint32Array(buf)[0];
        const s = (bits >> 16) & 0x8000;
        const e = ((bits >> 23) & 0xff) - 127 + 15;
        const m = (bits >> 13) & 0x3ff;
        if (e <= 0) f16[i] = s;
        else if (e >= 31) f16[i] = s | 0x7c00;
        else f16[i] = s | (e << 10) | m;
      }
      return f16;
    }

    if (model?.getPastKeyValues) {
      const originalGetPKV = model.getPastKeyValues.bind(model);
      model.getPastKeyValues = function(
        decoderResults: Record<string, any>,
        pastKeyValues: any,
        ...rest: any[]
      ) {
        const cache = originalGetPKV(decoderResults, pastKeyValues, ...rest);

        if (!tq) return cache;

        // Compress each layer's new K/V vectors
        for (const name in decoderResults) {
          if (!name.startsWith("present") || !name.endsWith(".value")) continue;
          const vTensor = decoderResults[name];
          const kName = name.replace(".value", ".key");
          const kTensor = decoderResults[kName];
          if (!vTensor?.ort_tensor?.cpuData || !kTensor?.ort_tensor?.cpuData) continue;

          const match = name.match(/(\d+)\.value/);
          if (!match) continue;
          const layerIdx = parseInt(match[1]);
          if (!kvCache.has(layerIdx)) kvCache.set(layerIdx, { keys: [], values: [] });
          const layer = kvCache.get(layerIdx)!;

          const kData = kTensor.ort_tensor.cpuData;
          const vData = vTensor.ort_tensor.cpuData;
          const seqLen = vTensor.dims[2];
          const numHeads = vTensor.dims[1];

          // Only compress the NEW positions (seqLen - existing compressed count)
          const existingCount = layer.keys.length;
          for (let pos = existingCount; pos < seqLen; pos++) {
            for (let h = 0; h < numHeads; h++) {
              const offset = (h * seqLen + pos) * headDim;
              const kSlice = f16ToF32(new Uint16Array(kData.buffer, kData.byteOffset + offset * 2, headDim));
              const vSlice = f16ToF32(new Uint16Array(vData.buffer, vData.byteOffset + offset * 2, headDim));
              layer.keys.push(tq.encode(kSlice));
              layer.values.push(tq.encode(vSlice));
              kvCompressedBytes += layer.keys[layer.keys.length - 1].byteLength + layer.values[layer.values.length - 1].byteLength;
              kvUncompressedBytes += headDim * 2 * 2; // K+V float16
            }
          }
        }

        // Replace cache tensors with decompressed from TQ compressed data.
        // This frees the original ORT tensors and reconstructs from compressed.
        const pkvNames = Object.keys(cache).filter(n => n.startsWith("past_key_values."));
        for (const name of pkvNames) {
          const match = name.match(/(\d+)\.(key|value)/);
          if (!match) continue;
          const layerIdx = parseInt(match[1]);
          const isKey = match[2] === "key";
          const layer = kvCache.get(layerIdx);
          if (!layer) continue;

          const origTensor = (cache as any)[name];
          const [batch, numHeads, seqLen, dim] = origTensor.dims;
          const stored = isKey ? layer.keys : layer.values;
          if (stored.length === 0) continue;

          // Reconstruct float16 tensor from compressed data
          const f16Data = new Uint16Array(batch * numHeads * seqLen * dim);
          const positionsPerHead = seqLen;
          for (let h = 0; h < numHeads; h++) {
            for (let pos = 0; pos < positionsPerHead; pos++) {
              const idx = pos * numHeads + h;
              if (idx >= stored.length) break;
              const decoded = tq.decode(stored[idx]);
              const f16 = f32ToF16(decoded);
              const outOffset = (h * positionsPerHead + pos) * dim;
              f16Data.set(f16, outOffset);
            }
          }
          (cache as any)[name] = new Tensor("float16", f16Data, [batch, numHeads, seqLen, dim]);
        }

        return cache;
      };
      console.log("[TQ] KV cache compress/decompress active");
    }

    statusEl.innerHTML = "Ready";
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
