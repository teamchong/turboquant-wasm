/**
 * Benchmark batched vs sequential matmul.
 *
 * Measures per-iteration wall time for:
 *   - 1× matmulBatched(N=K) dispatch
 *   - K× matmul(N=1) dispatches
 * Both produce the same total output (K row vectors). If the batched
 * kernel amortizes weight-load and dequant cost across K activations
 * the batched path should be meaningfully faster per output vector.
 *
 * Speedup = sequentialMs / batchedMs. A value > 1 means batching wins.
 * From the prior session's claim, Q4K batched was 4-7× SLOWER (speedup
 * 0.14-0.28×) on M1 because the cache was already hiding redundant
 * weight loads. Q6K won 1.3-2.8× because its dequant is expensive.
 */

import { test, chromium, type BrowserContext, type Page } from "@playwright/test";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const HERE = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = resolve(HERE, "..", ".playwright-data");

const WEIGHTS_TO_TEST = [
  "blk.0.attn_q.weight",
  "blk.0.attn_k.weight",
  "blk.0.attn_v.weight",
  "blk.0.ffn_gate.weight",
  "blk.0.ffn_down.weight",
];
const BATCH_SIZES = [2, 4, 8];
const ITERS = 200;

test.describe.serial("Batched matmul benchmark", () => {
  test.setTimeout(600_000);
  let context: BrowserContext;
  let page: Page;

  test.beforeAll(async () => {
    test.setTimeout(1_800_000);
    context = await chromium.launchPersistentContext(DATA_DIR, { channel: "chrome", args: [] });
    page = context.pages()[0] || await context.newPage();
    page.on("console", msg => {
      const t = msg.text();
      if (t.startsWith("[draw]") || t.startsWith("[engine]") || t.startsWith("[bench]")) console.log(t);
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

  test("batched vs sequential wall-clock", async () => {
    const results = await page.evaluate(async (args) => {
      const { weights, batchSizes, iters } = args;
      const engine = (window as any).__engine;
      const out: Array<any> = [];
      for (const w of weights) {
        for (const b of batchSizes) {
          try {
            const r = await engine.benchmarkBatchedMatmul(w, b, iters);
            out.push(r);
          } catch (e) {
            out.push({ weightName: w, batchSize: b, error: (e as Error).message });
          }
        }
      }
      return out;
    }, { weights: WEIGHTS_TO_TEST, batchSizes: BATCH_SIZES, iters: ITERS });

    console.log("");
    console.log("===============================================================");
    console.log(`[bench] Batched matmul benchmark (iters=${ITERS})`);
    console.log("[bench]");
    console.log("[bench]   weight                   kind N  dims         seq_ms   bat_ms   speedup");
    for (const r of results) {
      if (r.error) {
        console.log(`[bench]   ${r.weightName.padEnd(24)} ERR ${r.error}`);
        continue;
      }
      const kind = r.isQ6K ? "Q6K" : "Q4K";
      const dims = `${r.nRows}×${r.nCols}`.padEnd(12);
      const seq = r.sequentialMs.toFixed(1).padStart(7);
      const bat = r.batchedMs.toFixed(1).padStart(7);
      const speedup = r.speedup.toFixed(2);
      const marker = r.speedup >= 1.1 ? "WIN " : (r.speedup <= 0.9 ? "LOSS" : "wash");
      console.log(`[bench]   ${r.weightName.padEnd(24)} ${kind}  ${r.batchSize}  ${dims} ${seq}  ${bat}   ${speedup}× ${marker}`);
    }
    console.log("===============================================================");
  });
});
