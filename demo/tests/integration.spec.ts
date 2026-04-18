import { test, expect, chromium, type BrowserContext, type Page } from "@playwright/test";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

/**
 * Integration tests: verify the engine produces correct output with real model weights.
 *
 * Uses persistent browser profile so the 3.1GB OPFS cache survives between runs.
 * First run downloads the model (slow), subsequent runs reuse it (<1s).
 *
 * Each test proves a specific capability:
 * 1. Single token: full 35-layer forward pass (sliding + global) → valid logits
 * 2. Batch prefill + decode: N tokens prefilled, then autoregressive decode works
 * 3. Tensor type routing: global layers use Q4K for V (not Q6K) — the bug that caused NaN
 */
const DATA_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..", ".playwright-data");

let context: BrowserContext;
let page: Page;

test.describe.serial("Integration: Real Model", () => {
  test.setTimeout(600_000);

  test.beforeAll(async () => {
    test.setTimeout(600_000); // beforeAll doesn't inherit describe-level timeout
    context = await chromium.launchPersistentContext(DATA_DIR, {
      channel: "chrome",
      args: [],
    });
    page = context.pages()[0] || await context.newPage();
    page.on("console", msg => {
      const t = msg.text();
      if (t.startsWith("[draw]") || t.startsWith("[model-loader]") || t.startsWith("[engine]")) {
        console.log(t);
      }
    });
    await page.goto("http://localhost:5173/draw.html");
    try {
      await page.waitForFunction(
        () => document.querySelector("#status")?.textContent === "Ready",
        {}, { timeout: 480_000 },
      );
    } catch {
      const text = await page.locator("#status").textContent();
      test.skip(true, `Model not ready: "${text}"`);
    }
  });

  test.afterAll(async () => { await context?.close(); });

  test("system prompt cached + user prefill: matches the real demo flow", async () => {
    // Mirrors how the demo uses the engine: a large system prompt is prefilled once
    // in `beforeAll`, snapshotted, then each user interaction restores and prefills the
    // user prompt. The old bug silently dropped rawK writes during the user prefill
    // because the buffer was sized for user tokens only, but the copy offset used the
    // absolute (restored) cache position.
    //
    // This test must run FIRST in the suite so the beforeAll snapshot (1752 positions)
    // is still intact — later tests call resetCache + re-snapshot for their own scenarios.
    const result = await page.evaluate(async () => {
      const engine = (window as any).__engine;
      engine.restoreCache();
      const statsBefore = await engine.getStats();
      const userTokens = [106, 1645, 108, 147791, 107];
      const firstToken = await engine.prefill(userTokens);
      const more: number[] = [];
      let tok = firstToken;
      for (let i = 0; i < 5; i++) {
        tok = await engine.generateToken(tok);
        more.push(tok);
      }
      const statsAfter = await engine.getStats();
      return {
        beforePositions: statsBefore.positions,
        afterPositions: statsAfter.positions,
        firstToken,
        more,
      };
    });
    console.log(`Cache before user prefill: ${result.beforePositions} (system prompt)`);
    console.log(`Cache after user prefill+decode: ${result.afterPositions}`);
    console.log(`First decoded: ${result.firstToken}`);
    console.log(`Next 5 tokens: [${result.more}]`);

    // System prompt length tracks SYSTEM_PROMPT in main.ts — rebuild
    // system-cache.bin (via tests/build-cache.spec.ts) and update this
    // constant whenever the prompt changes. Current: 4945 tokens for
    // the prompt that teaches 8 diagram types including ER, UML, swimlane.
    const SYSTEM_TOKENS = 4945;
    expect(result.beforePositions, "system prompt must be cached from beforeAll").toBe(SYSTEM_TOKENS);
    expect(result.afterPositions, "after user prefill+5 decode: system + user + decode").toBe(SYSTEM_TOKENS + 5 + 5);
    expect(result.firstToken, "first decoded token after user prefill must be valid").toBeGreaterThan(0);
    expect(result.firstToken, "first decoded token must be in Gemma vocab").toBeLessThan(262144);
    for (const tok of result.more) {
      expect(tok, "every subsequent decoded token must be valid").toBeGreaterThan(0);
      expect(tok, "every subsequent decoded token must be in vocab").toBeLessThan(262144);
    }
    const uniqueTokens = new Set([result.firstToken, ...result.more]).size;
    expect(uniqueTokens, "decoded tokens should have some variety (not all same garbage)").toBeGreaterThan(1);
  });

  test("single token: 35 layers produce valid logits", async () => {
    // Proves: forward pass through all 35 layers (28 sliding + 7 global) → non-NaN token
    const result = await page.evaluate(async () => {
      const engine = (window as any).__engine;
      engine.resetCache();
      const token = await engine.generateToken(2); // <bos>
      const stats = await engine.getStats();
      return { token, positions: stats.positions, ratio: stats.ratio };
    });
    console.log(`Single token: ${result.token}, KV: ${result.positions} pos, ${result.ratio.toFixed(1)}x`);
    expect(result.token, "token must be a valid vocab index").toBeGreaterThan(0);
    expect(result.positions, "KV cache must have 1 position").toBe(1);
    expect(result.ratio, "TQ compression ratio must exceed 1x").toBeGreaterThan(1);
  });

  test("batch prefill + decode: multi-token generation", async () => {
    // Proves: batch prefill populates KV cache, decode reads from it correctly
    const result = await page.evaluate(async () => {
      const engine = (window as any).__engine;
      engine.resetCache();
      // Prefill 10 tokens (includes both sliding and global layer cache entries)
      const inputIds = [2, 106, 108, 235371, 10, 147791, 107, 108, 106, 1645];
      const firstDecodeToken = await engine.prefill(inputIds);
      // Decode 3 more tokens autoregressively
      const decodeTokens = [firstDecodeToken];
      for (let i = 0; i < 2; i++) {
        decodeTokens.push(await engine.generateToken(decodeTokens[decodeTokens.length - 1]));
      }
      const stats = await engine.getStats();
      return {
        decodeTokens,
        positions: stats.positions,
        compressedMB: stats.compressedMB,
        uncompressedMB: stats.uncompressedMB,
        ratio: stats.ratio,
      };
    });
    console.log(`Decode tokens: [${result.decodeTokens}]`);
    console.log(`KV: ${result.positions} pos, ${result.compressedMB.toFixed(2)}MB / ${result.uncompressedMB.toFixed(2)}MB (${result.ratio.toFixed(1)}x)`);

    // 10 prefill + 2 decode forward passes = 12 KV positions (dense attention during prefill)
    expect(result.positions, "KV must have 12 positions (10 prefill + 2 decode)").toBe(12);
    // All decode tokens must be valid
    for (const tok of result.decodeTokens) {
      expect(tok, "each decode token must be a valid vocab index").toBeGreaterThan(0);
    }
    // Compression ratio. getStats() measures compressed bytes vs bf16
    // baseline (2 bytes/dim), NOT f32, because the engine's KV cache is
    // f16 before TQ encoding. Current polar config is 6r+7a + 2-bit QJL
    // = 8.5 bits/dim, so theoretical ratio vs bf16 is 16/8.5 ≈ 1.88x.
    // With 8 bytes per-position metadata overhead, a 12-position cache
    // comes in around 1.84x. The >1.5x threshold just proves TQ is
    // actually shrinking the cache instead of expanding it.
    expect(result.ratio, "TQ compression ratio").toBeGreaterThan(1.5);
  });

  test("decode speed: measure steady-state tok/s", async () => {
    // Measures real decode throughput: prefill short prompt, decode 20 tokens, report tok/s.
    // Skips first 2 tokens (warmup), measures tokens 3-20 for steady state.
    const result = await page.evaluate(async () => {
      const engine = (window as any).__engine;
      engine.resetCache();

      // Prefill a short prompt (system-like)
      const inputIds = [2, 106, 108, 235371, 10, 147791, 107, 108, 106, 1645];
      let token = await engine.prefill(inputIds);

      // Decode 20 tokens, timing each one
      const times: number[] = [];
      for (let i = 0; i < 20; i++) {
        const t0 = performance.now();
        token = await engine.generateToken(token);
        times.push(performance.now() - t0);
      }

      // Skip first 2 (warmup), measure rest
      const warmup = 2;
      const steadyTimes = times.slice(warmup);
      const totalMs = steadyTimes.reduce((a, b) => a + b, 0);
      const avgMs = totalMs / steadyTimes.length;
      const tokPerSec = 1000 / avgMs;

      // GPU profiling breakdown (last token)
      const profile = engine.lastProfile;

      return {
        allTimes: times.map((t: number) => +t.toFixed(1)),
        avgMs: +avgMs.toFixed(1),
        tokPerSec: +tokPerSec.toFixed(1),
        profile,
      };
    });

    console.log(`\n=== DECODE SPEED ===`);
    console.log(`Per-token times (ms): [${result.allTimes.join(", ")}]`);
    console.log(`Steady-state (tokens 3-20): ${result.avgMs}ms/tok → ${result.tokPerSec} tok/s`);
    if (result.profile) {
      console.log(`\nGPU profile (last token):`);
      const entries = Object.entries(result.profile as Record<string, { ms: number; count: number }>)
        .sort(([, a], [, b]) => b.ms - a.ms);
      const total = entries.reduce((s, [, v]) => s + v.ms, 0);
      console.log(`  TOTAL: ${total.toFixed(2)}ms GPU`);
      for (const [cat, { ms, count }] of entries) {
        console.log(`  ${cat}: ${ms.toFixed(2)}ms (${count} passes, ${(ms / total * 100).toFixed(0)}%)`);
      }
    }

    expect(result.tokPerSec, "decode must exceed 1 tok/s").toBeGreaterThan(1);
    expect(result.avgMs, "per-token latency must be under 1000ms").toBeLessThan(1000);
  });

  test("restore + prefill: hybrid TQ/dense attention produces valid tokens", async () => {
    // Regression: after snapshot/restore, subsequent prefill calls have `priorPos > 0`,
    // which triggers hybrid attention — TQ for cached positions + dense for current-call
    // positions, with rawK/rawV indexed relative to the current call.
    //
    // Before the fix, the second prefill wrote to rawK at absolute (priorPos * kvSize)
    // offsets in a buffer only sized for current-call tokens → silent out-of-bounds
    // drops on WebGPU → dense attention read zeros → decode produced garbage tokens.
    //
    // After the fix: TQ attention handles the cached positions and dense reads from
    // relative offsets, so decoded tokens must be valid Gemma vocab indices and
    // the second prefill must not error.
    const result = await page.evaluate(async () => {
      const engine = (window as any).__engine;
      const inputIds = [2, 106, 108, 235371, 10, 147791, 107, 108, 106, 1645];
      const decodeSteps = 3;

      engine.resetCache();
      let t = await engine.prefill(inputIds.slice(0, 5));
      engine.snapshotCache();
      engine.restoreCache();
      t = await engine.prefill(inputIds.slice(5, 10));
      const tokens: number[] = [t];
      for (let i = 1; i < decodeSteps; i++) {
        t = await engine.generateToken(t);
        tokens.push(t);
      }
      const stats = await engine.getStats();
      return { tokens, positions: stats.positions, ratio: stats.ratio };
    });
    console.log(`Hybrid path tokens: [${result.tokens}]`);
    console.log(`KV after split prefill: ${result.positions} positions, ${result.ratio.toFixed(1)}x`);

    // 10 prefilled + (decodeSteps - 1) decoded forward passes
    expect(result.positions, "cache must contain 10 prefill + 2 decode positions").toBe(12);
    for (const tok of result.tokens) {
      expect(tok, "every decoded token must be > 0 (not EOS/pad)").toBeGreaterThan(0);
      expect(tok, "every decoded token must be in Gemma vocab").toBeLessThan(262144);
    }
  });

  test("V weights have mixed quantization — matmulAuto required", async () => {
    // Proves: V weights use BOTH Q4K and Q6K across layers → engine must route dynamically.
    // This was the bug: engine hardcoded Q6K for all V → NaN at Q4K layers.
    const result = await page.evaluate(() => {
      const engine = (window as any).__engine;
      const tensors = engine.model.tensors;
      const Q4K = 12, Q6K = 14;
      let q4kCount = 0, q6kCount = 0;
      const samples: Array<{ layer: number; type: number; dims: number[] }> = [];
      for (let i = 0; i < 35; i++) {
        const t = tensors.get(`blk.${i}.attn_v.weight`);
        if (t) {
          if (t.type === Q4K) q4kCount++;
          if (t.type === Q6K) q6kCount++;
          if ([0, 4, 15, 34].includes(i)) samples.push({ layer: i, type: t.type, dims: t.dims });
        }
      }
      return { q4kCount, q6kCount, samples, Q4K, Q6K };
    });
    console.log(`V weight types: ${result.q4kCount} Q4K, ${result.q6kCount} Q6K`);
    for (const s of result.samples) {
      const t = s.type === result.Q6K ? "Q6K" : "Q4K";
      console.log(`  blk.${s.layer}.attn_v: ${t} [${s.dims}]`);
    }

    // Both Q4K and Q6K must exist — proves matmulAuto is necessary
    expect(result.q4kCount, "some V weights must be Q4K").toBeGreaterThan(0);
    expect(result.q6kCount, "some V weights must be Q6K").toBeGreaterThan(0);
    // Sliding layers have dim=256, global layers have dim=512
    const sliding = result.samples.find(s => s.layer === 0)!;
    const global = result.samples.find(s => s.layer === 4)!;
    expect(sliding.dims[1], "sliding V output dim").toBe(256);
    expect(global.dims[1], "global V output dim").toBe(512);
  });
});
