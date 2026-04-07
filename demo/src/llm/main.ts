import { pipeline, TextStreamer, type TextGenerationPipeline } from "@huggingface/transformers";

const $ = (s: string) => document.querySelector(s)!;

const MODEL_ID = "onnx-community/gemma-4-E2B-it-ONNX";
const statusEl = $("#status") as HTMLElement;
const messagesEl = $("#messages") as HTMLElement;
const inputEl = $("#input") as HTMLInputElement;
const sendBtn = $("#send") as HTMLButtonElement;
const statSpeed = $("#stat-speed") as HTMLElement;
const statCtx = $("#stat-ctx") as HTMLElement;

let generator: TextGenerationPipeline | null = null;
let contextTokens = 0;
let generating = false;

type ChatMessage = { role: "user" | "assistant"; content: string };
const history: ChatMessage[] = [];

function addMsg(role: "user" | "assistant" | "system", text: string): HTMLElement {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

async function initModel() {
  if (!navigator.gpu) {
    statusEl.textContent = "WebGPU not available";
    statusEl.classList.add("error");
    addMsg("system", "WebGPU required. Use Chrome 113+ or Edge 113+.");
    return;
  }

  statusEl.textContent = "Loading Gemma 4 E2B (~1.3 GB)...";

  try {
    generator = await pipeline("text-generation", MODEL_ID, {
      dtype: "q4",
      device: "webgpu",
      progress_callback: (progress: { status: string; progress?: number; file?: string }) => {
        if (progress.status === "progress" && progress.progress !== undefined) {
          statusEl.textContent = `Downloading ${progress.file?.split("/").pop() ?? ""} ${Math.round(progress.progress)}%`;
        }
      },
    }) as TextGenerationPipeline;

    statusEl.textContent = "Ready";
    statusEl.classList.add("ready");
    inputEl.disabled = false;
    sendBtn.disabled = false;
    inputEl.focus();
  } catch (e) {
    statusEl.textContent = `Error: ${(e as Error).message}`;
    statusEl.classList.add("error");
    console.error("Model init failed:", e);
  }
}

async function onSend() {
  if (!generator || generating) return;
  const text = inputEl.value.trim();
  if (!text) return;

  inputEl.value = "";
  inputEl.disabled = true;
  sendBtn.disabled = true;
  generating = true;
  statusEl.textContent = "Generating...";
  statusEl.classList.remove("ready");

  addMsg("user", text);
  history.push({ role: "user", content: text });

  const assistantDiv = addMsg("assistant", "");
  let response = "";
  let tokenCount = 0;
  const startTime = performance.now();

  const streamer = new TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    callback_function: (chunk: string) => {
      response += chunk;
      tokenCount++;
      assistantDiv.textContent = response;
      messagesEl.scrollTop = messagesEl.scrollHeight;

      const elapsed = (performance.now() - startTime) / 1000;
      if (elapsed > 0) {
        statSpeed.textContent = `${(tokenCount / elapsed).toFixed(1)} tok/s`;
      }
    },
  });

  try {
    const messages = history.map((m) => ({
      role: m.role as string,
      content: m.content,
    }));

    await generator(messages, {
      max_new_tokens: 512,
      temperature: 0.7,
      do_sample: true,
      streamer,
    });

    history.push({ role: "assistant", content: response });
    contextTokens += tokenCount;
    statCtx.textContent = `~${contextTokens} tokens`;
  } catch (e) {
    assistantDiv.textContent = `Error: ${(e as Error).message}`;
    console.error("Generation error:", e);
  }

  generating = false;
  statusEl.textContent = "Ready";
  statusEl.classList.add("ready");
  inputEl.disabled = false;
  sendBtn.disabled = false;
  inputEl.focus();
}

sendBtn.addEventListener("click", onSend);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !inputEl.disabled) onSend();
});

initModel();
