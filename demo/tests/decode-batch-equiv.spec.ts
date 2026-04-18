/**
 * Phase 2 correctness test for speculative decoding.
 *
 * The speculative decode loop depends on decodeBatch producing the same
 * outputs AND the same KV cache side-effects as a sequence of single-
 * token generateToken calls on the same inputs. If this invariant
 * doesn't hold, speculative decoding will silently produce outputs that
 * diverge from the sequential path — a quality regression.
 *
 * This test runs two paths in succession:
 *   A. generateToken(t0) → argmax_A0, generateToken(t1) → argmax_A1
 *   B. rollback, decodeBatch([t0, t1]) → [argmax_B0, argmax_B1]
 *
 * Asserts A == B token-for-token, AND that the post-state KV length is
 * the same in both paths (so further decode steps land on identical state).
 */

import { test, expect, chromium, type BrowserContext, type Page } from "@playwright/test";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const HERE = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = resolve(HERE, "..", ".playwright-data");

test.describe.serial("decodeBatch equivalence", () => {
  test.setTimeout(1_800_000);
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

  test("sequential vs batched argmax match for 2, 4, and 8 tokens", async () => {
    const result = await page.evaluate(async () => {
      const engine = (window as any).__engine;
      engine.restoreCache();

      // Use tokens that would plausibly appear after the system prompt
      // + a short user query. Any valid vocab ids work — we're checking
      // that the model's argmax path is consistent regardless of the
      // specific token sequence.
      const testTokens = [106, 1645, 108, 147791, 107, 108, 106, 1645];

      async function runSequential(tokens: number[]): Promise<{ argmaxes: number[]; positionAfter: number }> {
        engine.restoreCache();
        const stats0 = await engine.getStats();
        const startPos = stats0.positions;
        const argmaxes: number[] = [];
        for (const t of tokens) {
          const next = await engine.generateToken(t);
          argmaxes.push(next);
        }
        const stats1 = await engine.getStats();
        return { argmaxes, positionAfter: stats1.positions };
      }

      async function runBatched(tokens: number[]): Promise<{ argmaxes: number[]; positionAfter: number }> {
        engine.restoreCache();
        const argmaxes = await engine.decodeBatch(tokens);
        const stats1 = await engine.getStats();
        return { argmaxes, positionAfter: stats1.positions };
      }

      const results: Array<{
        batchSize: number;
        sequential: { argmaxes: number[]; positionAfter: number };
        batched:    { argmaxes: number[]; positionAfter: number };
        match: boolean;
      }> = [];

      for (const batchSize of [2, 4, 8]) {
        const subset = testTokens.slice(0, batchSize);
        const sequential = await runSequential(subset);
        const batched = await runBatched(subset);
        const match = sequential.argmaxes.length === batched.argmaxes.length
          && sequential.argmaxes.every((v, i) => v === batched.argmaxes[i])
          && sequential.positionAfter === batched.positionAfter;
        results.push({ batchSize, sequential, batched, match });
      }

      return results;
    });

    for (const r of result) {
      console.log(`[batch-test] B=${r.batchSize}:`);
      console.log(`[batch-test]   sequential argmax: [${r.sequential.argmaxes.join(", ")}] pos=${r.sequential.positionAfter}`);
      console.log(`[batch-test]   batched    argmax: [${r.batched.argmaxes.join(", ")}] pos=${r.batched.positionAfter}`);
      console.log(`[batch-test]   match: ${r.match}`);
      expect(r.batched.argmaxes, `B=${r.batchSize} argmaxes must match`).toEqual(r.sequential.argmaxes);
      expect(r.batched.positionAfter, `B=${r.batchSize} position after must match`).toBe(r.sequential.positionAfter);
    }
  });

  test("rollbackKV resets position and allows re-decode to same result", async () => {
    const result = await page.evaluate(async () => {
      const engine = (window as any).__engine;
      engine.restoreCache();

      const tokens = [106, 1645, 108, 147791];
      const statsStart = await engine.getStats();
      const startPos = statsStart.positions;

      // Batched decode of all 4 tokens.
      const full = await engine.decodeBatch(tokens);
      const statsAfterFull = await engine.getStats();

      // Roll back to position + 2, re-decode tokens[2..3], argmaxes should match full[2..3].
      await engine.rollbackKV(startPos + 2);
      const statsAfterRollback = await engine.getStats();
      const partial = await engine.decodeBatch(tokens.slice(2));
      const statsAfterPartial = await engine.getStats();

      return {
        startPos,
        full,
        positionAfterFull: statsAfterFull.positions,
        positionAfterRollback: statsAfterRollback.positions,
        partial,
        positionAfterPartial: statsAfterPartial.positions,
      };
    });

    console.log("[batch-test] rollback test:");
    console.log(`[batch-test]   start position:     ${result.startPos}`);
    console.log(`[batch-test]   full decode:        [${result.full.join(", ")}] pos=${result.positionAfterFull}`);
    console.log(`[batch-test]   after rollback(+2): pos=${result.positionAfterRollback}`);
    console.log(`[batch-test]   partial re-decode:  [${result.partial.join(", ")}] pos=${result.positionAfterPartial}`);

    expect(result.positionAfterFull).toBe(result.startPos + 4);
    expect(result.positionAfterRollback).toBe(result.startPos + 2);
    expect(result.positionAfterPartial).toBe(result.startPos + 4);
    // The last two argmaxes of the full batch should match the partial re-decode.
    // This proves rollback correctly restored the KV state so tokens[2..3] see
    // the same context in both runs.
    expect(result.partial).toEqual(result.full.slice(2));
  });
});
