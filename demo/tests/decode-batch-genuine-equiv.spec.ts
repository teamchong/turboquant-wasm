/**
 * Conformance test for decodeBatchGenuine.
 *
 * The genuinely-batched decode path must produce the same argmax tokens as
 * running N sequential generateToken calls from the same starting state.
 * If it doesn't, the batched matmul / per-slot attention / offset-bound PLE
 * plumbing is wrong and the speedup is chasing wrong numerics.
 *
 * Method:
 *   1. Snapshot cache + position.
 *   2. Run sequential generateToken for N drafts, collect argmaxes.
 *   3. Rollback KV to the snapshot position.
 *   4. Run decodeBatchGenuine for the same N drafts, collect argmaxes.
 *   5. Assert equality.
 *
 * Starts from the cached system prompt (position ≈ 4945), so the test picks
 * up at realistic KV cache sizes where the attention range actually matters.
 */

import { test, expect, chromium, type BrowserContext, type Page } from "@playwright/test";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const HERE = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = resolve(HERE, "..", ".playwright-data");

const BATCH_SIZES = [1, 2, 4];

test.describe.serial("decodeBatchGenuine equivalence", () => {
  test.setTimeout(600_000);
  let context: BrowserContext;
  let page: Page;

  test.beforeAll(async () => {
    test.setTimeout(1_800_000);
    context = await chromium.launchPersistentContext(DATA_DIR, { channel: "chrome", args: [] });
    page = context.pages()[0] || await context.newPage();
    page.on("console", msg => {
      const t = msg.text();
      if (t.startsWith("[draw]") || t.startsWith("[engine]") || t.startsWith("[batch-test]")) console.log(t);
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

  for (const B of BATCH_SIZES) {
    test(`decodeBatchGenuine(N=${B}) matches sequential generateToken`, async () => {
      const result = await page.evaluate(async (batchSize: number) => {
        const engine = (window as any).__engine;
        engine.restoreCache();

        // Seed sequential path with a first token to get the draft stream going.
        const firstToken = 2; // BOS
        const seqTokens = [firstToken];
        const seqArgmaxes: number[] = [];
        // Run `batchSize` sequential generateTokens starting from BOS at the
        // cached position.
        let tok = firstToken;
        for (let i = 0; i < batchSize; i++) {
          const next = await (engine as any).generateToken(tok);
          seqArgmaxes.push(next);
          tok = next;
        }
        const seqPos = engine.lastStats?.positions ?? -1;

        // Rollback and run the batched path on the same input sequence.
        engine.restoreCache();
        // Build the batched input: first draft is BOS, rest are the sequential
        // predictions so decodeBatchGenuine predicts one next-token per slot
        // conditional on the prefix.
        const batchInputs = [firstToken, ...seqArgmaxes.slice(0, batchSize - 1)];
        const batchArgmaxes = await (engine as any).decodeBatchGenuine(batchInputs);

        return { seqArgmaxes, batchArgmaxes, seqPos, batchInputs };
      }, B);

      console.log(`[batch-test] N=${B}:`);
      console.log(`[batch-test]   batch inputs:  ${JSON.stringify(result.batchInputs)}`);
      console.log(`[batch-test]   seq argmax:    ${JSON.stringify(result.seqArgmaxes)}`);
      console.log(`[batch-test]   batch argmax:  ${JSON.stringify(result.batchArgmaxes)}`);
      expect(result.batchArgmaxes).toEqual(result.seqArgmaxes);
    });
  }
});
