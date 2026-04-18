/**
 * Phase 1 of the speculative-decoding experiment: measure how many
 * tokens an n-gram prompt-lookup drafter would accept if we ran it
 * against the actual output of our engine on a representative diagram
 * prompt.
 *
 * This test runs zero engine-level changes. It uses the existing
 * streamTokens path to capture a full diagram generation, then in
 * pure JS simulates the "look at last K tokens, find them earlier in
 * the context, read M tokens that followed" drafting strategy and
 * counts how many tokens the drafter would have predicted correctly.
 *
 * Decision gate:
 *   - If best acceptance rate is > 40%, proceed to Phase 2 (batched
 *     decode + KV rollback) in the engine.
 *   - If < 40%, abandon — the structural win from speculative decoding
 *     on this workload won't beat the implementation cost.
 */

import { test, expect, chromium, type BrowserContext, type Page } from "@playwright/test";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const HERE = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = resolve(HERE, "..", ".playwright-data");

// Same prompt the perf-baseline uses so acceptance-rate numbers map 1:1
// to the tok/s numbers we'd see at runtime.
const URL_PROMPT = "What happens when you type a URL into your browser? Include every network layer.";
const MAX_NEW_TOKENS = 400;

test.describe.serial("Speculative n-gram lookup — offline acceptance rate", () => {
  test.setTimeout(1_800_000);
  let context: BrowserContext;
  let page: Page;

  test.beforeAll(async () => {
    test.setTimeout(1_800_000);
    context = await chromium.launchPersistentContext(DATA_DIR, { channel: "chrome", args: [] });
    page = context.pages()[0] || await context.newPage();
    page.on("console", msg => {
      const t = msg.text();
      if (t.startsWith("[draw]") || t.startsWith("[engine]") || t.startsWith("[spec]")) console.log(t);
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

  test("acceptance rate sweep over (ngram, lookahead) combos", async () => {
    const result = await page.evaluate(async (args) => {
      const { prompt, maxTokens } = args;
      const engine = (window as any).__engine;
      const tokenizer = (window as any).__tokenizer;
      engine.restoreCache();

      // Tokenize the user prompt with the chat template so the decode
      // starts from a realistic cached-system + user-prefill state.
      const userMessages = [{ role: "user", content: prompt }];
      const userText = tokenizer.apply_chat_template(userMessages, { tokenize: false, add_generation_prompt: true });
      const userTokenIds: number[] = Array.from(tokenizer.encode(userText));
      if (userTokenIds[0] === 2) userTokenIds.shift();

      // Prefill the user prompt through the engine and stream-decode a
      // full diagram. We record every token id produced.
      const firstToken = await engine.prefill(userTokenIds);
      const produced: number[] = [firstToken];
      await engine.streamTokens(firstToken, maxTokens - 1, [1, 106], (msg: any) => {
        produced.push(msg.id);
      });

      // Simulate speculative decoding offline. The "context" the drafter
      // sees is the prefill prompt + everything produced so far. At each
      // step we look at the trailing n-gram and try to find it earlier in
      // that context. If we find it, we read the next `lookahead` tokens
      // from that earlier position and compare against the actual next
      // `lookahead` tokens from the recorded sequence.
      function simulate(ngram: number, lookahead: number) {
        const fullCtx = [...userTokenIds, ...produced];
        const prefillLen = userTokenIds.length;
        // Build a hash table: n-gram -> list of indices where it occurs.
        // Keyed as "t1,t2,...,tK" for a K-gram.
        const ngramIndex = new Map<string, number[]>();
        for (let i = 0; i + ngram <= fullCtx.length; i++) {
          const key = fullCtx.slice(i, i + ngram).join(",");
          let list = ngramIndex.get(key);
          if (!list) { list = []; ngramIndex.set(key, list); }
          list.push(i);
        }

        // Walk the produced stream. At each position p (relative to
        // the start of the produced array, not full context), simulate:
        //   1. Look at the n-gram ending at p-1 (inclusive). If p < ngram,
        //      skip — we don't have enough history yet.
        //   2. Look it up in ngramIndex. Filter matches that occurred
        //      STRICTLY BEFORE the n-gram position we're querying from.
        //   3. If a match exists, the drafter predicts the next
        //      `lookahead` tokens from that earlier position.
        //   4. Compare against produced[p..p+lookahead]. Count accepted
        //      (longest matching prefix).
        //   5. Advance p by accepted + 1 (the +1 is the "current token"
        //      which is always accepted since the real model produced it).
        let accepted = 0;
        let forwardPasses = 0;
        let p = 0;
        const actualSeq = [...userTokenIds, ...produced];
        // We simulate starting from the first produced token. The user
        // prefill tokens are "given" (produced by prefill, not decode).
        while (p < produced.length) {
          forwardPasses++;
          const globalIdx = prefillLen + p;
          if (globalIdx < ngram) { p++; continue; }
          // The n-gram ending just before the token we're generating at p.
          const keySlice = actualSeq.slice(globalIdx - ngram, globalIdx);
          const key = keySlice.join(",");
          const matches = ngramIndex.get(key);
          if (!matches) { p++; continue; }
          // Find the LATEST prior match (most recent repetition — usually
          // gives the most relevant continuation for template-heavy output).
          let bestMatch = -1;
          for (const m of matches) {
            if (m + ngram < globalIdx && m + ngram > bestMatch) bestMatch = m + ngram;
          }
          if (bestMatch < 0) { p++; continue; }
          // Read up to `lookahead` tokens following the prior match and
          // compare against actualSeq starting at globalIdx.
          let matchLen = 0;
          for (let i = 0; i < lookahead; i++) {
            if (bestMatch + i >= actualSeq.length) break;
            if (globalIdx + i >= actualSeq.length) break;
            if (actualSeq[bestMatch + i] !== actualSeq[globalIdx + i]) break;
            matchLen++;
          }
          accepted += matchLen;
          p += 1 + matchLen; // +1 for the current token, +matchLen accepted from lookahead
        }

        return {
          ngram, lookahead,
          totalTokens: produced.length,
          acceptedExtra: accepted,
          forwardPasses,
          // Effective tokens per forward pass (including the always-accepted current token).
          tokensPerPass: produced.length / forwardPasses,
          acceptanceRate: accepted / Math.max(1, produced.length),
        };
      }

      const results: ReturnType<typeof simulate>[] = [];
      for (const ngram of [2, 3, 4]) {
        for (const lookahead of [4, 6, 8]) {
          results.push(simulate(ngram, lookahead));
        }
      }

      return { producedCount: produced.length, results };
    }, { prompt: URL_PROMPT, maxTokens: MAX_NEW_TOKENS });

    console.log("");
    console.log("==========================================================");
    console.log(`[spec] generated ${result.producedCount} tokens on URL prompt`);
    console.log("[spec] offline n-gram speculative decoding simulation:");
    console.log("[spec]   ngram  lookahead  tokens/pass  acceptance  extra");
    for (const r of result.results) {
      console.log(
        `[spec]   ${String(r.ngram).padStart(5)}  ${String(r.lookahead).padStart(9)}  ${r.tokensPerPass.toFixed(2).padStart(11)}  ${(r.acceptanceRate * 100).toFixed(1).padStart(9)}%  ${r.acceptedExtra}`,
      );
    }
    console.log("==========================================================");
    console.log("");

    // Decision gate: log the best tokens/pass figure. The test always
    // passes — this is a measurement test, not a correctness test.
    const best = result.results.reduce((a, b) => (b.tokensPerPass > a.tokensPerPass ? b : a));
    console.log(`[spec] best: ngram=${best.ngram} lookahead=${best.lookahead} → ${best.tokensPerPass.toFixed(2)} tokens/forward-pass (${(best.acceptanceRate * 100).toFixed(1)}% acceptance)`);
    console.log(`[spec] projected speedup: ${best.tokensPerPass.toFixed(2)}× on this workload`);
    console.log(`[spec] at our current 13.4 tok/s baseline, that would be ~${(13.4 * best.tokensPerPass).toFixed(1)} tok/s effective`);

    expect(result.producedCount).toBeGreaterThan(50);
  });
});
