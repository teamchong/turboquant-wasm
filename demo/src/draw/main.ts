/** Prompt to Diagram: Gemma 4 E2B generates drawmode code, rendered as Excalidraw. */

import { resetTqCaches, getTqStats } from "./tq-apply-attention.js";
import { env, TextStreamer, InterruptableStoppingCriteria, RawImage, AutoTokenizer, AutoProcessor, AutoModelForImageTextToText } from "@huggingface/transformers";
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
const attachBtn = $("#attach-btn") as HTMLButtonElement;
const fileInput = $("#file-input") as HTMLInputElement;
const thumbnailsEl = $("#thumbnails") as HTMLElement;
const promptArea = $("#prompt-area") as HTMLElement;

let tokenizer: any = null;
let processor: any = null;
let model: any = null;
let modelReady = false;
let busy = false;
let currentCode = "";
let lastStmtCount = 0;
const stopCriteria = new InterruptableStoppingCriteria();

// -- Attachments --
interface Attachment { blob: Blob; objectUrl: string; rawImage: RawImage; isPdf: boolean }
const attachments: Attachment[] = [];
const MAX_ATTACHMENTS = 6;

async function pdfPageToBlob(file: File): Promise<Blob> {
  const pdfjsLib: any = await import(/* @vite-ignore */ "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.9.155/pdf.min.mjs");
  pdfjsLib.GlobalWorkerOptions.workerSrc = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.9.155/pdf.worker.min.mjs";
  const pdf = await pdfjsLib.getDocument({ data: await file.arrayBuffer() }).promise;
  const page = await pdf.getPage(1);
  const vp = page.getViewport({ scale: 1.5 });
  const canvas = new OffscreenCanvas(vp.width, vp.height);
  await page.render({ canvasContext: canvas.getContext("2d")!, viewport: vp }).promise;
  return canvas.convertToBlob({ type: "image/png" });
}

async function addFiles(files: FileList | File[]) {
  for (const file of Array.from(files)) {
    if (attachments.length >= MAX_ATTACHMENTS) break;
    const isPdf = file.type === "application/pdf";
    if (!file.type.startsWith("image/") && !isPdf) continue;
    const blob = isPdf ? await pdfPageToBlob(file) : file;
    const objectUrl = URL.createObjectURL(blob);
    const rawImage = await RawImage.fromBlob(blob);
    attachments.push({ blob, objectUrl, rawImage, isPdf });
  }
  renderThumbnails();
}

function renderThumbnails() {
  thumbnailsEl.innerHTML = "";
  attachments.forEach((att, i) => {
    const wrap = document.createElement("div");
    wrap.className = "thumb";
    const img = document.createElement("img");
    img.src = att.objectUrl;
    wrap.appendChild(img);
    if (att.isPdf) {
      const badge = document.createElement("span");
      badge.className = "pdf-badge";
      badge.textContent = "PDF";
      wrap.appendChild(badge);
    }
    const x = document.createElement("button");
    x.className = "thumb-x";
    x.textContent = "\u00d7";
    x.addEventListener("click", () => { URL.revokeObjectURL(attachments[i].objectUrl); attachments.splice(i, 1); renderThumbnails(); });
    wrap.appendChild(x);
    thumbnailsEl.appendChild(wrap);
  });
}

/** Unified generation: always goes through processor → model.generate */
async function callModel(
  messages: Array<{ role: string; content: any }>,
  images: RawImage[] | null,
  genOptions: Record<string, any>,
) {
  const text: string = tokenizer.apply_chat_template(messages, { tokenize: false, add_generation_prompt: true });
  const processed = await processor(text, images?.length ? images : null);
  await model.generate({ ...processed, ...genOptions });
}

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

  if (!modelReady) {
    generateBtn.innerHTML = '<span class="spinner"></span> Waiting for model...';
    while (!modelReady) {
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

    // Build text part of user content
    let textPart = prompt;
    if (currentCode) {
      textPart = `Current diagram code:\n\`\`\`\n${currentCode}\n\`\`\`\n\nModify it: ${prompt}`;
    }

    // Build user content — multimodal array when images attached, plain string otherwise
    const images = attachments.map(a => a.rawImage);
    let userContent: any = textPart;
    if (images.length > 0) {
      const blocks: any[] = images.map(img => ({ type: "image", image: img }));
      blocks.push({ type: "text", text: textPart });
      userContent = blocks;
    }

    const messages = [
      { role: "system" as const, content: SYSTEM_PROMPT },
      { role: "user" as const, content: userContent },
    ];

    const streamer = new TextStreamer(tokenizer, {
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

    await callModel(messages, images, {
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
      const retryStreamer = new TextStreamer(tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        token_callback_function: () => { tokenCount++; },
        callback_function: (chunk: string) => {
          generatedCode += chunk;
          appendCode(chunk);
        },
      });
      await callModel(retryMessages, null, {
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

  const dlFiles: Record<string, { loaded: number; total: number }> = {};
  const progress_callback = (progress: any) => {
    if (progress.status === "progress" && progress.file) {
      dlFiles[progress.file] = { loaded: progress.loaded || 0, total: progress.total || 0 };
      let totalLoaded = 0, totalSize = 0;
      for (const f of Object.values(dlFiles)) { totalLoaded += f.loaded; totalSize += f.total; }
      const pct = totalSize > 0 ? ((totalLoaded / totalSize) * 100).toFixed(0) : "0";
      statusEl.innerHTML = `<span class="spinner"></span> Downloading Gemma 4 E2B: ${pct}% (${(totalLoaded / 1e9).toFixed(1)} / ${(totalSize / 1e9).toFixed(1)} GB)`;
    } else if (progress.status === "loading" || progress.status === "ready") {
      statusEl.innerHTML = `<span class="spinner"></span> Loading model into WebGPU...`;
    }
  };
  const modelOpts = { device: "webgpu" as const, dtype: "q4f16" as const, progress_callback };

  try {
    [tokenizer, processor, model] = await Promise.all([
      AutoTokenizer.from_pretrained(MODEL_ID, modelOpts),
      AutoProcessor.from_pretrained(MODEL_ID, modelOpts),
      AutoModelForImageTextToText.from_pretrained(MODEL_ID, modelOpts),
    ]);
    modelReady = true;

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

// -- Attachment event listeners --
attachBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => { if (fileInput.files) addFiles(fileInput.files); fileInput.value = ""; });
promptArea.addEventListener("dragover", (e) => { e.preventDefault(); promptArea.classList.add("drag-over"); });
promptArea.addEventListener("dragleave", () => promptArea.classList.remove("drag-over"));
promptArea.addEventListener("drop", (e) => { e.preventDefault(); promptArea.classList.remove("drag-over"); if (e.dataTransfer?.files.length) addFiles(e.dataTransfer.files); });
promptEl.addEventListener("paste", (e) => {
  const items = e.clipboardData?.items;
  if (!items) return;
  const imageFiles: File[] = [];
  for (const item of items) {
    if (item.type.startsWith("image/")) { const f = item.getAsFile(); if (f) imageFiles.push(f); }
  }
  if (imageFiles.length > 0) { e.preventDefault(); addFiles(imageFiles); }
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
    tokenizer = null; processor = null; model = null; modelReady = false;
    setCode("");
    currentCode = "";
    attachments.forEach(a => URL.revokeObjectURL(a.objectUrl));
    attachments.length = 0;
    renderThumbnails();
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
