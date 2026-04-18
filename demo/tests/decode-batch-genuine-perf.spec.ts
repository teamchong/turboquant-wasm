/**
 * Perf test: decodeBatchGenuine(N) vs N × generateToken.
 *
 * The batched path uses matmulBatched (Q/K/V/attn_out/gate/up/down) so
 * the matmul category (50% of per-token time) should amortize across N
 * activation vectors. Attention is still sequential per slot because
 * each slot has a different causal range.
 *
 * Speedup is reported per TOKEN — i.e. (N × seq_per_token) / batched_total.
 * >1 means batched wins; if the result is >=2× we get a clear speculative-
 * decoding win (2.27 tokens/pass projected → ≥2× net tok/s).
 */

import { test, chromium, type BrowserContext, type Page } from "@playwright/test";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const HERE = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = resolve(HERE, "..", ".playwright-data");
const BATCH_SIZES = [2, 4, 8];
const REPS = 10;

test.describe.serial("decodeBatchGenuine perf", () => {
  test.setTimeout(1_200_000);
  let context: BrowserContext;
  let page: Page;

  test.beforeAll(async () => {
    test.setTimeout(1_800_000);
    context = await chromium.launchPersistentContext(DATA_DIR, { channel: "chrome", args: [] });
    page = context.pages()[0] || await context.newPage();
    page.on("console", msg => {
      const t = msg.text();
      if (t.startsWith("[draw]") || t.startsWith("[engine]") || t.startsWith("[perf]")) console.log(t);
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

  test("batched vs sequential wall-clock per token", async () => {
    const results = await page.evaluate(async (args) => {
      const { batchSizes, reps } = args;
      const engine = (window as any).__engine;
      const out: Array<any> = [];

      for (const B of batchSizes) {
        // Warm-up run (not timed) so the first-run pipeline-compile overhead
        // doesn't skew the measurement.
        engine.restoreCache();
        const warmTokens = Array.from({ length: B }, (_, i) => 2 + i);
        await engine.decodeBatchGenuine(warmTokens);

        // Sequential: run `reps` iterations of `B` sequential generateToken
        // calls from the restored cache snapshot.
        engine.restoreCache();
        const seqStart = performance.now();
        for (let r = 0; r < reps; r++) {
          engine.restoreCache();
          let tok = 2;
          for (let i = 0; i < B; i++) tok = await engine.generateToken(tok);
        }
        const seqMs = performance.now() - seqStart;

        // Batched: run `reps` iterations of `decodeBatchGenuine` with B draft
        // tokens, each starting from the restored cache.
        engine.restoreCache();
        const batchStart = performance.now();
        for (let r = 0; r < reps; r++) {
          engine.restoreCache();
          // Build B tokens. Use the first sequential result as the seed so
          // acceptance is meaningful if a real drafter is wired in later.
          const tokens = Array.from({ length: B }, (_, i) => 2 + i);
          await engine.decodeBatchGenuine(tokens);
        }
        const batchMs = performance.now() - batchStart;

        out.push({
          batchSize: B,
          reps,
          seqMs,
          batchMs,
          seqMsPerToken: seqMs / (reps * B),
          batchMsPerToken: batchMs / (reps * B),
          speedupPerToken: seqMs / batchMs,
        });
      }
      return out;
    }, { batchSizes: BATCH_SIZES, reps: REPS });

    console.log("");
    console.log("===============================================================");
    console.log(`[perf] decodeBatchGenuine vs sequential generateToken (reps=${REPS})`);
    console.log("[perf]");
    console.log("[perf]   N  seq(ms/tok)  bat(ms/tok)  speedup");
    for (const r of results) {
      const seqPt = r.seqMsPerToken.toFixed(2).padStart(8);
      const batPt = r.batchMsPerToken.toFixed(2).padStart(8);
      const speedup = r.speedupPerToken.toFixed(2);
      const marker = r.speedupPerToken >= 1.1 ? "WIN " : (r.speedupPerToken <= 0.9 ? "LOSS" : "wash");
      console.log(`[perf]   ${r.batchSize}  ${seqPt}    ${batPt}    ${speedup}× ${marker}`);
    }
    console.log("===============================================================");
  });
});
