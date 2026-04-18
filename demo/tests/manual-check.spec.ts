import { test, chromium, type BrowserContext, type Page } from "@playwright/test";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const DATA_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..", ".playwright-data");

test.describe.serial("Manual detokenize", () => {
  test.setTimeout(600_000);
  let context: BrowserContext;
  let page: Page;

  test.beforeAll(async () => {
    test.setTimeout(600_000);
    context = await chromium.launchPersistentContext(DATA_DIR, { channel: "chrome", args: [] });
    page = context.pages()[0] || await context.newPage();
    page.on("console", msg => {
      const t = msg.text();
      if (t.startsWith("[draw]") || t.startsWith("[engine]") || t.startsWith("[worker]")) {
        console.log(t);
      }
    });
    await page.goto("http://localhost:5173/draw.html?skipSysPrompt=1");
    await page.waitForFunction(
      () => document.querySelector("#status")?.textContent === "Ready",
      {}, { timeout: 480_000 },
    );
  });

  test.afterAll(async () => { await context?.close(); });

  test("dump tensor names", async () => {
    const info = await page.evaluate(() => {
      const engine = (window as any).__engine;
      const out: Array<{ name: string; type: number; dims: number[] }> = [];
      for (const [name, t] of engine.model.tensors) out.push({ name, type: t.type, dims: t.dims });
      out.sort((a, b) => a.name.localeCompare(b.name));
      return out;
    });
    const globals = info.filter(t => !t.name.startsWith("blk."));
    console.log("[tensors] globals:");
    for (const g of globals) console.log(`  ${g.name}  type=${g.type}  dims=[${g.dims.join(",")}]`);
    const layer0 = info.filter(t => t.name.startsWith("blk.0."));
    console.log("[tensors] layer 0:");
    for (const g of layer0) console.log(`  ${g.name}  type=${g.type}  dims=[${g.dims.join(",")}]`);
    const layer15 = info.filter(t => t.name.startsWith("blk.15."));
    console.log("[tensors] layer 15 (first shared-KV layer):");
    for (const g of layer15) console.log(`  ${g.name}  type=${g.type}  dims=[${g.dims.join(",")}]`);
  });

  test("10-token batch prefill + decode, detokenized", async () => {
    const result = await page.evaluate(async () => {
      const engine = (window as any).__engine;
      const tokenizer = (window as any).__tokenizer;
      engine.resetCache();
      const inputIds = [2, 106, 108, 235371, 10, 147791, 107, 108, 106, 1645];
      const firstToken = await engine.prefill(inputIds);
      const tokens: number[] = [firstToken];
      let t = firstToken;
      for (let i = 0; i < 30; i++) {
        t = await engine.generateToken(t);
        tokens.push(t);
        if (t === 106 || t === 107) break;
      }
      const inputText = tokenizer.decode(inputIds, { skip_special_tokens: false });
      const outputText = tokenizer.decode(tokens, { skip_special_tokens: false });
      return { tokens, inputText, outputText };
    });
    console.log(`[check] input text: ${JSON.stringify(result.inputText)}`);
    console.log(`[check] output tokens: [${result.tokens.slice(0, 15).join(",")}]`);
    console.log(`[check] output text: ${JSON.stringify(result.outputText)}`);
  });

  test("prefill throughput — 256 tokens", async () => {
    const result = await page.evaluate(async () => {
      const engine = (window as any).__engine;
      const N = 256;
      const tokens: number[] = [2];
      for (let i = 1; i < N; i++) tokens.push(100 + (i * 7919) % 50000);

      // Batched prefill
      engine.resetCache();
      const tBatch0 = performance.now();
      await engine.prefill(tokens);
      const batchedMs = performance.now() - tBatch0;

      // Sequential generateToken for reference
      engine.resetCache();
      const tSeq0 = performance.now();
      for (const t of tokens) await engine.generateToken(t);
      const sequentialMs = performance.now() - tSeq0;

      return {
        N,
        batchedMs, batchedTps: (N / (batchedMs / 1000)).toFixed(1),
        sequentialMs, sequentialTps: (N / (sequentialMs / 1000)).toFixed(1),
      };
    });
    console.log(`[prefill-bench] batched:    ${result.N} tokens in ${result.batchedMs.toFixed(0)} ms (${result.batchedTps} tok/s)`);
    console.log(`[prefill-bench] sequential: ${result.N} tokens in ${result.sequentialMs.toFixed(0)} ms (${result.sequentialTps} tok/s)`);
  });

  test("user prefill after system prompt cached, detokenized", async () => {
    const result = await page.evaluate(async () => {
      const engine = (window as any).__engine;
      const tokenizer = (window as any).__tokenizer;
      engine.resetCache();
      const userMsgs = [{ role: "user", content: "draw a cat" }];
      const userText = tokenizer.apply_chat_template(userMsgs, { tokenize: false, add_generation_prompt: true });
      const userEncoded = tokenizer.encode(userText);
      const userTokens: number[] = Array.from(userEncoded);
      // Prepend BOS — the chat template doesn't include it and Gemma's tokenizer
      // doesn't add it automatically. Without BOS the model has no anchor.
      if (userTokens[0] !== 2) userTokens.unshift(2);
      let t = await engine.prefill(userTokens);
      const tokens: number[] = [t];
      for (let i = 0; i < 40; i++) {
        t = await engine.generateToken(t);
        tokens.push(t);
        if (t === 106 || t === 107) break;
      }
      const inputText = tokenizer.decode(userTokens, { skip_special_tokens: false });
      const outputText = tokenizer.decode(tokens, { skip_special_tokens: false });
      return { userTokenCount: userTokens.length, tokens, inputText, outputText };
    });
    console.log(`[user] input (${result.userTokenCount} tokens): ${JSON.stringify(result.inputText)}`);
    console.log(`[user] output tokens: [${result.tokens.slice(0, 20).join(",")}...]`);
    console.log(`[user] output text: ${JSON.stringify(result.outputText)}`);
  });
});
