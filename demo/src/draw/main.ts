/** Prompt to Diagram: Gemma 4 E2B generates drawmode code, rendered as Excalidraw. */

import { resetTqCaches, getTqStats } from "./tq-apply-attention.js";
import { pipeline, env, TextStreamer, InterruptableStoppingCriteria, type TextGenerationPipeline } from "@huggingface/transformers";
import { executeCode } from "./drawmode/executor.js";
import { SDK_TYPES } from "./drawmode/sdk-types.js";
import { loadWasm } from "./drawmode/layout.js";
import drawmodeWasm from "./drawmode/drawmode-wasm.js";
import { mountExcalidraw, updateDiagram, resetDiagram, fitToScreen } from "./excalidraw-viewer.js";
import { createEditor, setCode, appendCode, getCode } from "./code-editor.js";

env.backends.onnx.wasm!.numThreads = 1;

const MODEL_ID = "onnx-community/gemma-4-E2B-it-ONNX";

const $ = (s: string) => document.querySelector(s)!;
const statusEl = $("#status") as HTMLElement;
const promptEl = $("#prompt") as HTMLTextAreaElement;
const generateBtn = $("#generate") as HTMLButtonElement;
const codeArea = $("#code-area") as HTMLElement;
const diagramContainer = $("#diagram-container") as HTMLElement;
const renderBtn = $("#render") as HTMLButtonElement;
const wipeBtn = $("#wipe") as HTMLButtonElement;
const statSpeed = $("#stat-speed") as HTMLElement;
const statKV = $("#stat-kv") as HTMLElement;

let gen: TextGenerationPipeline | null = null;
let busy = false;
let currentCode = "";
let lastStmtCount = 0;
const stopCriteria = new InterruptableStoppingCriteria();

const SYSTEM_PROMPT = `Output a complete Diagram script. No markdown, no comments, no helpers. One statement per line. Assign every node to a const, use those consts in connect/addGroup.

Format:
const d = new Diagram({ direction: "TB" });
const api = d.addBox("API Server\\n(Node.js)", { row: 1, col: 2, color: "backend", icon: "server" });
d.connect(api, db, "TCP :5432");
d.addGroup("Backend", [api, auth, cache]);
return d.render();

Rules:
- Think about what each component ACTUALLY DOES before generating. Do not create separate nodes for the same thing (e.g. "Cloudflare Worker" IS the API — don't add a separate "API Endpoint" node). Each node must represent a distinct real component.
- Use d.addBox for ALL nodes. NEVER use d.addText — it causes overlaps.
- EVERY node needs row, col, color, icon. Nothing else — no width, height, fillStyle, strokeColor.
- Spread nodes across multiple columns (col 0-4), not a single column.
- color: "frontend"|"backend"|"database"|"cache"|"queue"|"external"|"orchestration"|"users"|"storage"|"ai"
- icon: "server"|"database"|"lock"|"globe"|"users"|"api"|"cache"|"queue"|"cloud"|"code"|"shield"|"search"
- d.connect(from, to, "label") — string label only, no options object, no hex colors.
- Detailed labels with \\n. 15+ nodes. 4+ groups.

${SDK_TYPES}`;

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
  resetDiagram();
  statusEl.textContent = "Generating diagram code...";

  const startTime = performance.now();
  let tokenCount = 0;
  let generatedCode = "";

  lastStmtCount = 0;

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
        const tq = getTqStats();
        if (tq.contextLength > 0) {
          const compKB = (tq.compressedBytes / 1024).toFixed(0);
          const rawKB = (tq.uncompressedBytes / 1024).toFixed(0);
          statKV.textContent = `KV: ${compKB}KB / ${rawKB}KB (${tq.ratio.toFixed(1)}x) · ${tq.contextLength} pos · ${tq.layers} layers`;
        }
        // Streaming render on each new complete statement
        const stmts = (generatedCode.match(/;\s*\n/g) || []).length;
        if (stmts > lastStmtCount) {
          lastStmtCount = stmts;
          let partial = generatedCode.trim();
          partial = partial.replace(/^```(?:javascript|js|typescript|ts)?\s*\n?/i, "");
          const lastSemi = partial.lastIndexOf(";");
          if (lastSemi >= 0) {
            partial = partial.substring(0, lastSemi + 1);
            if (!partial.includes("new Diagram")) {
              partial = `const d = new Diagram({ direction: "TB" });\n${partial}`;
            }
            if (!partial.includes("d.render()")) {
              partial = `${partial}\nreturn d.render();`;
            }
            executeCode(partial).then(({ result, error }) => {
              if (!error && result.json) updateDiagram(result.json.elements || []);
            });
          }
        }
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

    const tqFinal = getTqStats();
    const compKB = (tqFinal.compressedBytes / 1024).toFixed(0);
    const rawKB = (tqFinal.uncompressedBytes / 1024).toFixed(0);
    statKV.textContent = `KV: ${compKB}KB / ${rawKB}KB (${tqFinal.ratio.toFixed(1)}x) · ${tqFinal.contextLength} pos · ${tqFinal.layers} layers`;

    let code = generatedCode.trim();
    code = code.replace(/^```(?:javascript|js|typescript|ts)?\s*\n?/i, "");
    code = code.replace(/\n?```\s*$/i, "");
    setCode(code);

    // Execute — if it fails, feed the error back to the model and retry
    const MAX_RETRIES = 3;
    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      statusEl.textContent = attempt === 0 ? "Rendering diagram..." : `Retrying (${attempt}/${MAX_RETRIES})...`;
      const { result, error } = await executeCode(code);

      if (!error && result.json) {
        currentCode = code;
        updateDiagram(result.json.elements || []);
        fitToScreen();
        statusEl.textContent = attempt === 0 ? "Diagram ready" : `Diagram ready (retry ${attempt})`;
        statusEl.classList.remove("error");
        break;
      }

      if (attempt === MAX_RETRIES) {
        statusEl.textContent = `Code error after ${MAX_RETRIES} retries: ${error}`;
        statusEl.classList.add("error");
        break;
      }

      // Feed the error back to the model to get corrected code
      generatedCode = "";
      tokenCount = 0;
      setCode("");
      const retryMessages = [
        { role: "system" as const, content: SYSTEM_PROMPT },
        { role: "user" as const, content: prompt },
        { role: "assistant" as const, content: code },
        { role: "user" as const, content: `That code threw an error: ${error}\nRespond with ONLY the corrected code.` },
      ];
      const retryStreamer = new TextStreamer((gen as any).tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        token_callback_function: () => { tokenCount++; },
        callback_function: (chunk: string) => {
          generatedCode += chunk;
          appendCode(chunk);
        },
      });
      await gen(retryMessages, {
        max_new_tokens: 1024,
        do_sample: false,
        streamer: retryStreamer,
        stopping_criteria: stopCriteria,
      });
      code = generatedCode.trim();
      code = code.replace(/^```(?:javascript|js|typescript|ts)?\s*\n?/i, "");
      code = code.replace(/\n?```\s*$/i, "");
      setCode(code);
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
  createEditor(codeArea, (code) => { currentCode = code; });
  await loadWasm(drawmodeWasm);
  mountExcalidraw(diagramContainer);

  // TQ attention is patched into ORT's GQA kernel by the Vite plugin (vite-plugin-tq-gqa).
  // No manual hook needed — the plugin replaces applyAttention with tqApplyAttention
  // so all KV data is stored and operated on in TQ compressed format automatically.

  statusEl.innerHTML = '<span class="spinner"></span> Loading Gemma 4 E2B...';

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

    console.log("[TQ] model loaded — TQ attention active via GQA kernel patch");

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

document.querySelectorAll(".suggestion").forEach(btn => {
  btn.addEventListener("click", () => {
    promptEl.value = (btn as HTMLElement).dataset.prompt!;
    promptEl.focus();
  });
});

renderBtn.addEventListener("click", async () => {
  const code = getCode();
  if (!code) return;
  statusEl.textContent = "Rendering...";
  const { result, error } = await executeCode(code);
  if (error) {
    statusEl.textContent = `Code error: ${error}`;
    statusEl.classList.add("error");
  } else if (result.json) {
    updateDiagram(result.json.elements || []);
    fitToScreen();
    statusEl.textContent = "Diagram ready";
    statusEl.classList.remove("error");
  }
});

wipeBtn.addEventListener("click", () => {
  if (busy) stopCriteria.interrupt();
  window.stop(); // abort all in-flight fetches (model downloads)
  statusEl.textContent = "Wiping all data...";
  wipeBtn.disabled = true;

  // Delete cached model weights from Cache API and IndexedDB
  const wipeCache = "caches" in window
    ? caches.keys().then(names => Promise.all(names.map(n => caches.delete(n))))
    : Promise.resolve();
  const wipeDb = indexedDB.databases
    ? indexedDB.databases().then(list => list.forEach(db => { if (db.name) indexedDB.deleteDatabase(db.name); }))
    : Promise.resolve();

  Promise.all([wipeCache, wipeDb]).then(() => {
    resetTqCaches();
    gen = null;
    setCode("");
    currentCode = "";
    updateDiagram([]);
    statSpeed.textContent = "--";
    statKV.textContent = "KV: --";
    statusEl.textContent = "All data wiped.";
    wipeBtn.style.display = "none";
    generateBtn.disabled = true;
  });
});

// Show wipe button immediately
wipeBtn.style.display = "inline-block";

main();
