/**
 * Perf baseline test — runs the URL prompt against the real demo flow
 * (system prompt cached, then user query → generate full response) and
 * records tok/s + per-op GPU profile. Use this as the reference number
 * for all future perf work.
 *
 * Reference baseline captured 2026-04-13 on commit 2afc616:
 *   prompt:  "What happens when you type a URL into your browser? ..."
 *   result:  643 tokens of valid Diagram code in 107.4s
 *   speed:   6.0 tok/s steady state
 *   KV:      23 MB compressed / 42 MB raw, 2407 cached positions
 *   per-token GPU profile (head-of-context):
 *     matmul       55.9 ms  (140 passes/token)
 *     tq_wsum      32.0 ms  (35 passes)
 *     tq_softmax   20.3 ms  (35 passes)
 *     tq_scores    15.1 ms  (35 passes)
 *     tq_encode     6.0 ms  (15 passes)
 *     tq_rotate     4.1 ms  (35 passes)
 *     norm          3.5 ms  (141 passes)
 *     logits        3.0 ms  (3 passes)
 *     tq_invrot     1.8 ms  (35 passes)
 *     [other small]
 *
 * The test does not gate on absolute numbers — it logs them so a regression
 * is visible at a glance. Compare against the reference manually.
 */

import { test, chromium, type BrowserContext, type Page } from "@playwright/test";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const DATA_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..", ".playwright-data");

const URL_PROMPT = "What happens when you type a URL into your browser? Include every network layer";

test.describe.serial("Perf baseline", () => {
  test.setTimeout(1_800_000);
  let context: BrowserContext;
  let page: Page;

  test.beforeAll(async () => {
    test.setTimeout(1_800_000);
    context = await chromium.launchPersistentContext(DATA_DIR, { channel: "chrome", args: [] });
    page = context.pages()[0] || await context.newPage();
    page.on("console", msg => {
      const t = msg.text();
      if (t.startsWith("[draw]") || t.startsWith("[engine]") || t.startsWith("[worker]")) console.log(t);
    });
    page.on("pageerror", err => console.log(`[pageerror] ${err.message}`));
    await page.goto("http://localhost:5173/draw.html");
    await page.waitForFunction(
      () => document.querySelector("#status")?.textContent === "Ready",
      {}, { timeout: 1_500_000 },
    );
  });

  test.afterAll(async () => {
    test.setTimeout(120_000);
    await context?.close();
  });

  test("URL prompt — measure tok/s and GPU profile", async () => {
    const result = await page.evaluate(async (prompt: string) => {
      const engine = (window as any).__engine;
      const tokenizer = (window as any).__tokenizer;
      engine.restoreCache();

      const userMessages = [{ role: "user", content: prompt }];
      const userText = tokenizer.apply_chat_template(userMessages, { tokenize: false, add_generation_prompt: true });
      const userEncoded = tokenizer.encode(userText);
      const userTokenIds: number[] = Array.from(userEncoded);
      // System prompt cache has BOS at position 0; strip duplicate from user tokens.
      if (userTokenIds[0] === 2) userTokenIds.shift();

      const tPrefill0 = performance.now();
      let t = await engine.prefill(userTokenIds);
      const prefillMs = performance.now() - tPrefill0;

      // Stream the decode using the pipelined path (GPU-resident tokenId chain,
      // K=4 encoders in flight, worker-side loop). Stops on <eos>=1 or <turn|>=106.
      // streamTokens skips profiler.resolve() to keep the pipeline hot, so the
      // per-op breakdown is captured via a short generateToken() sample below.
      const tDecode0 = performance.now();
      const tokens: number[] = [t];
      const MAX = 800;
      let lastToken = t;
      await engine.streamTokens(t, MAX - 1, [1, 106], (msg: any) => {
        tokens.push(msg.id);
        lastToken = msg.id;
      });
      const decodeMs = performance.now() - tDecode0;

      // Profile sample: call generateToken() a few times at the long-context
      // steady state we just reached. generateToken() waits for profiler
      // mapAsync per call, so it's ~2× slower than streamTokens — but it
      // populates engine.lastProfile with per-op ms/count data we can sum.
      // 8 samples is enough to average out warmup jitter without adding more
      // than ~1s to the test.
      const PROFILE_SAMPLES = 8;
      const profileSums: Record<string, { ms: number; count: number }> = {};
      let profileTokens = 0;
      // Discard the first sample — the pipeline drain at the end of
      // streamTokens leaves the GPU in a cold state that skews the first
      // generateToken call by ~20ms.
      const WARMUP = 1;
      for (let i = 0; i < PROFILE_SAMPLES + WARMUP; i++) {
        if (tokens.length >= MAX) break;
        lastToken = await engine.generateToken(lastToken);
        tokens.push(lastToken);
        const p = engine.lastProfile as Record<string, { ms: number; count: number }> | null;
        if (i < WARMUP || !p) continue;
        profileTokens++;
        for (const [name, v] of Object.entries(p)) {
          const acc = profileSums[name] ?? { ms: 0, count: 0 };
          acc.ms += v.ms;
          acc.count += v.count;
          profileSums[name] = acc;
        }
      }

      const stats = engine.getStats ? await engine.getStats() : null;

      return {
        prefillMs,
        decodeMs,
        userTokenCount: userTokenIds.length,
        // Throughput numbers exclude the post-stream profile samples so the
        // tps number reflects the fast streaming path, not the slower
        // generateToken-for-profile path.
        generatedTokens: tokens.length - PROFILE_SAMPLES - WARMUP,
        outputText: tokenizer.decode(tokens, { skip_special_tokens: false }),
        profileSums,
        profileTokens,
        stats,
      };
    }, URL_PROMPT);

    const tps = result.generatedTokens / (result.decodeMs / 1000);
    console.log("");
    console.log("=========================================");
    console.log("[perf] PERF BASELINE — URL prompt");
    console.log("=========================================");
    console.log(`[perf] user tokens:      ${result.userTokenCount}`);
    console.log(`[perf] prefill:          ${(result.prefillMs / 1000).toFixed(1)}s`);
    console.log(`[perf] generated tokens: ${result.generatedTokens}`);
    console.log(`[perf] decode:           ${(result.decodeMs / 1000).toFixed(1)}s`);
    console.log(`[perf] speed:            ${tps.toFixed(1)} tok/s`);
    if (result.stats) {
      console.log(`[perf] KV:               ${result.stats.compressedMB?.toFixed(1)}MB / ${result.stats.uncompressedMB?.toFixed(1)}MB (${result.stats.ratio?.toFixed(1)}x) · ${result.stats.positions} pos`);
    }
    console.log("");
    console.log(`[perf] per-token GPU profile (avg over ${result.profileTokens} tokens):`);
    const sorted = Object.entries(result.profileSums)
      .map(([k, v]) => ({ name: k, ms: v.ms / Math.max(1, result.profileTokens), count: Math.round(v.count / Math.max(1, result.profileTokens)) }))
      .sort((a, b) => b.ms - a.ms);
    for (const { name, ms, count } of sorted) {
      console.log(`[perf]   ${name.padEnd(16)} ${ms.toFixed(2).padStart(7)} ms  (${count} passes)`);
    }
    console.log("=========================================");
    console.log("");
    console.log(`[perf] output preview: ${result.outputText.slice(0, 200)}...`);
  });
});
