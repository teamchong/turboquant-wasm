/**
 * WebLLM baseline: run @mlc-ai/web-llm in the SAME browser environment
 * our TQ engine runs in, so we have a ground-truth tok/s number on THIS
 * machine (not the paper's M3 Max) for comparison.
 *
 * WebLLM doesn't support gemma-3n / Gemma 4 E2B (their newest Gemma is
 * Gemma-2), so a direct same-model comparison is impossible. The closest
 * apples-to-apples is:
 *   - gemma-2-2b-it-q4f16_1-MLC: same parameter class (~2B) as Gemma 4
 *     E2B's effective size
 *   - Phi-3.5-mini-instruct-q4f16_1-MLC: reference point from the WebLLM
 *     paper (their reported number: 71.1 tok/s on M3 Max)
 *
 * The first run downloads the model (~1.5-2.3 GB) to IndexedDB. The
 * persistent Playwright profile keeps it cached across runs.
 *
 * Run with:
 *   cd demo && npx playwright test tests/webllm-baseline.spec.ts
 */

import { test, chromium, type BrowserContext, type Page } from "@playwright/test";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const HERE = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = resolve(HERE, "..", ".playwright-data");

const PROMPT = "What happens when you type a URL into your browser? Include every network layer.";
const MAX_NEW_TOKENS = 256;

// The default is Gemma-2-2b for closest parameter-class parity with
// Gemma 4 E2B. Set WEBLLM_MODEL env var to override.
const MODEL_ID = process.env.WEBLLM_MODEL ?? "gemma-2-2b-it-q4f16_1-MLC";

test.describe.serial("WebLLM baseline", () => {
  test.setTimeout(1_800_000);
  let context: BrowserContext;
  let page: Page;

  test.beforeAll(async () => {
    test.setTimeout(1_800_000);
    context = await chromium.launchPersistentContext(DATA_DIR, { channel: "chrome", args: [] });
    page = context.pages()[0] || await context.newPage();
    page.on("console", msg => {
      const t = msg.text();
      // Echo useful lines from the harness + WebLLM itself.
      if (t.includes("WebLLM") || t.includes("webllm") || t.startsWith("[")) {
        console.log(t);
      }
    });
    page.on("pageerror", err => console.log(`[pageerror] ${err.message}`));
    await page.goto("http://localhost:5173/webllm-bench.html");
    // Wait for the module to finish importing and set __ready.
    await page.waitForFunction(() => (window as any).__ready === true, {}, { timeout: 60_000 });
    console.log("[webllm] harness loaded");
  });

  test.afterAll(async () => {
    test.setTimeout(120_000);
    await context?.close();
  });

  test(`${MODEL_ID} — tok/s on URL prompt`, async () => {
    // First run: downloads ~1.5-2.3 GB to IndexedDB + compiles the
    // WebGPU library (a few seconds). Subsequent runs: just init +
    // decode, ~3-5 seconds to warm.
    const result = await page.evaluate(async ({ modelId, prompt, maxTokens }) => {
      return await (window as any).__runBench(modelId, prompt, maxTokens);
    }, { modelId: MODEL_ID, prompt: PROMPT, maxTokens: MAX_NEW_TOKENS });

    console.log("");
    console.log("==========================================================");
    console.log(`[webllm] MODEL:    ${result.modelId}`);
    console.log(`[webllm] TOKENS:   ${result.tokenCount}`);
    console.log(`[webllm] PREFILL:  ${(result.prefillMs / 1000).toFixed(2)}s`);
    console.log(`[webllm] DECODE:   ${(result.decodeMs / 1000).toFixed(2)}s`);
    console.log(`[webllm] SPEED:    ${result.tokPerSec.toFixed(1)} tok/s`);
    console.log("[webllm] WebLLM runtime stats:");
    console.log(result.stats);
    console.log("==========================================================");
    console.log("");
    console.log(`[webllm] our engine on Gemma 4 E2B: 10.4 tok/s (for comparison)`);
  });
});
