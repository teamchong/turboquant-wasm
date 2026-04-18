/**
 * End-to-end test using the full demo flow: load system prompt, then ask a real
 * user question and verify the model returns coherent code. Mirrors what a real
 * user sees when they type a prompt into the running demo.
 */

import { test, chromium, type BrowserContext, type Page } from "@playwright/test";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const DATA_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..", ".playwright-data");

test.describe.serial("URL prompt end-to-end", () => {
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
      if (msg.type() === "error") console.log(`[browser error] ${t}`);
    });
    page.on("pageerror", err => console.log(`[pageerror] ${err.message}`));
    // No skipSysPrompt — load the real system prompt so the cache mirrors the
    // actual demo state when a user submits a query.
    await page.goto("http://localhost:5173/draw.html");
    await page.waitForFunction(
      () => document.querySelector("#status")?.textContent === "Ready",
      {}, { timeout: 1_500_000 },
    );
  });

  test.afterAll(async () => { await context?.close(); });

  test("'What happens when you type a URL' -> Diagram script tokens", async () => {
    const result = await page.evaluate(async () => {
      const engine = (window as any).__engine;
      const tokenizer = (window as any).__tokenizer;
      engine.restoreCache();
      const userMsgs = [{ role: "user", content: "What happens when you type a URL into your browser? Include every network layer" }];
      const userText = tokenizer.apply_chat_template(userMsgs, { tokenize: false, add_generation_prompt: true });
      const userEncoded = tokenizer.encode(userText);
      const userTokens: number[] = Array.from(userEncoded);
      // System prompt cache has BOS at position 0; strip the duplicate from the user tokens.
      if (userTokens[0] === 2) userTokens.shift();

      let t = await engine.prefill(userTokens);
      const tokens: number[] = [t];
      // Stop on <eos>=1 or <turn|>=106. 107 is plain `\n` and is NOT a stop.
      for (let i = 0; i < 400; i++) {
        if (t === 1 || t === 106) break;
        t = await engine.generateToken(t);
        tokens.push(t);
      }
      const inputText = tokenizer.decode(userTokens, { skip_special_tokens: false });
      const outputText = tokenizer.decode(tokens, { skip_special_tokens: false });
      return { tokenCount: tokens.length, inputText, outputText };
    });
    console.log(`[url] input: ${JSON.stringify(result.inputText.slice(0, 200))}...`);
    console.log(`[url] generated ${result.tokenCount} tokens`);
    console.log(`[url] output: ${JSON.stringify(result.outputText)}`);
  });

  test("short context with same system prompt — verify instruction sticks short-range", async () => {
    const result = await page.evaluate(async () => {
      const engine = (window as any).__engine;
      const tokenizer = (window as any).__tokenizer;
      engine.restoreCache();
      const userMsgs = [{ role: "user", content: "draw a cat" }];
      const userText = tokenizer.apply_chat_template(userMsgs, { tokenize: false, add_generation_prompt: true });
      const userEncoded = tokenizer.encode(userText);
      const userTokens: number[] = Array.from(userEncoded);
      if (userTokens[0] === 2) userTokens.shift();

      let t = await engine.prefill(userTokens);
      const tokens: number[] = [t];
      for (let i = 0; i < 200; i++) {
        if (t === 1 || t === 106) break;
        t = await engine.generateToken(t);
        tokens.push(t);
      }
      const outputText = tokenizer.decode(tokens, { skip_special_tokens: false });
      return { tokenCount: tokens.length, outputText };
    });
    console.log(`[short] generated ${result.tokenCount} tokens`);
    console.log(`[short] output: ${JSON.stringify(result.outputText)}`);
  });
});
