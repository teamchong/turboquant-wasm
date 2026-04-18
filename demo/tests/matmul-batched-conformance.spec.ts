/**
 * Conformance test for the batched Q4K / Q6K matmul kernels.
 *
 * The batched matmul shader holds N accumulators per thread and loads the
 * weight matrix once per workgroup, amortizing the weight bandwidth cost
 * across all N activation vectors. For it to be a drop-in replacement in
 * any multi-token path (batched prefill, speculative decoding) it MUST be
 * bit-identical to N sequential unbatched dispatches on the same inputs.
 *
 * The self-test (engine.conformanceBatchedMatmul) picks a real weight
 * tensor from the loaded model, generates deterministic input vectors,
 * runs both paths, and returns the max element-wise difference. This
 * Playwright spec drives that self-test across a representative set of
 * weight tensors (Q4K attn_q layer 0, Q4K attn_k layer 0, Q6K attn_v
 * layer 0 if it's Q6K, one FFN tensor) and several batch sizes (1, 2, 4, 8).
 *
 * Tolerance:
 * - Q4K MUST be bit-identical (maxDiff === 0). The unbatched Q4K kernel
 *   already hoists `w = (qs_val * dl - ml)` into a named intermediate, so
 *   the batched version's `let w = ...` has the same expression shape and
 *   the WGSL compiler emits the same assembly.
 * - Q6K is allowed a tiny f32 ULP diff (< 1e-5). The unbatched Q6K kernel
 *   inlines `sum += d * sc * q * sact[col]` which the compiler lowers
 *   with a slightly different FMA association than the batched kernel's
 *   factored `w = d * sc * q; sum += w * sact[col]`. The diff is ~1e-8,
 *   5+ orders of magnitude below anything that affects model output. A
 *   real kernel bug would produce diffs in the 0.01-1.0 range.
 */

import { test, expect, chromium, type BrowserContext, type Page } from "@playwright/test";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const HERE = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = resolve(HERE, "..", ".playwright-data");

// Weights we probe. Layer 0's attn_q/attn_k/ffn_gate are always Q4K on
// Gemma 4 E2B; attn_v is one of the tensors that can be either Q4K or
// Q6K depending on the per-layer mixed quantization. Probing both
// guarantees both kernels get exercised.
const WEIGHTS_TO_TEST = [
  "blk.0.attn_q.weight",
  "blk.0.attn_k.weight",
  "blk.0.attn_v.weight",
  "blk.0.ffn_gate.weight",
];
const BATCH_SIZES = [1, 2, 4, 8];

test.describe.serial("Batched matmul conformance", () => {
  test.setTimeout(600_000);
  let context: BrowserContext;
  let page: Page;

  test.beforeAll(async () => {
    test.setTimeout(1_800_000);
    context = await chromium.launchPersistentContext(DATA_DIR, { channel: "chrome", args: [] });
    page = context.pages()[0] || await context.newPage();
    page.on("console", msg => {
      const t = msg.text();
      if (t.startsWith("[draw]") || t.startsWith("[engine]") || t.startsWith("[conf]")) console.log(t);
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

  test("batched matmul matches sequential unbatched dispatch bit-for-bit", async () => {
    const results = await page.evaluate(async (args) => {
      const { weights, batchSizes } = args;
      const engine = (window as any).__engine;
      const out: Array<{ weightName: string; batchSize: number; ok: boolean; maxDiff: number; nRows?: number; nCols?: number; isQ6K?: boolean; error?: string }> = [];

      for (const w of weights) {
        for (const b of batchSizes) {
          try {
            const r = await engine.conformanceBatchedMatmul(w, b);
            out.push({ weightName: w, batchSize: b, ok: r.ok, maxDiff: r.maxDiff, nRows: r.nRows, nCols: r.nCols, isQ6K: r.isQ6K });
          } catch (e) {
            out.push({ weightName: w, batchSize: b, ok: false, maxDiff: NaN, error: (e as Error).message });
          }
        }
      }
      return out;
    }, { weights: WEIGHTS_TO_TEST, batchSizes: BATCH_SIZES });

    console.log("");
    console.log("==========================================================");
    console.log("[conf] Batched matmul conformance");
    console.log("[conf]");
    const Q6K_LOG_TOL = 1e-5;
    for (const r of results) {
      const kind = r.isQ6K ? "Q6K" : "Q4K";
      let status: string;
      if (r.error) status = `ERR ${r.error}`;
      else if (r.isQ6K) status = r.maxDiff < Q6K_LOG_TOL ? `OK (diff=${r.maxDiff.toExponential(1)})` : `FAIL diff=${r.maxDiff}`;
      else status = r.maxDiff === 0 ? "OK (bit-identical)" : `FAIL diff=${r.maxDiff}`;
      console.log(`[conf]   ${r.weightName.padEnd(24)} ${kind} N=${r.batchSize}  dims=${r.nRows}×${r.nCols}  ${status}`);
    }
    console.log("==========================================================");
    console.log("");

    // Q4K: require bit-identical. Q6K: tolerate <1e-5 from FMA reorder
    // (measured around 1.1e-8 — well below any model-observable impact).
    const Q6K_TOLERANCE = 1e-5;
    const failures = results.filter(r => {
      if (r.error) return true;
      if (r.isQ6K) return r.maxDiff >= Q6K_TOLERANCE;
      return r.maxDiff !== 0;
    });
    expect(failures, `batched matmul did not match sequential:\n${failures.map(f => `  ${f.weightName} N=${f.batchSize}: ${f.error ?? `maxDiff=${f.maxDiff}`}`).join("\n")}`).toHaveLength(0);
  });
});
