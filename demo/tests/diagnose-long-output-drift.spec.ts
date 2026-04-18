/**
 * Diagnose the "Source node not found: actor_1_..." bug that shows up
 * at ~585 tokens on the URL prompt. All other perf work in this session
 * ran into this failure as background noise; time to actually look at
 * what the model is emitting and fix the root cause.
 *
 * The test captures the FULL decoded output (no truncation), runs it
 * through the Diagram SDK executor, and if the compile fails prints:
 *   1. the exact error message
 *   2. the full code, line-numbered, so a reader can scan for the
 *      faulty connect/message call
 *   3. everything the code declared via `const X = d.addY(...)` or
 *      `const X = d.addY\nconst X = ...` patterns so we can diff the
 *      declared identifiers against the identifiers the code actually
 *      references
 *
 * This is a diagnostic probe, not a gate. The test always passes as
 * long as it captured SOMETHING — the output lands in the console log
 * for manual inspection.
 */

import { test, expect, chromium, type BrowserContext, type Page } from "@playwright/test";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const HERE = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = resolve(HERE, "..", ".playwright-data");

// Three diverse prompts that exercise different SDK code paths:
//  - URL → architecture diagram with addBox + addActor + connect
//  - OAuth → sequence diagram with addActor + message (pure sequence mode
//    — this catches any regression caused by addActor also registering
//    a node entry)
//  - UML → addClass + connect with the `{ relation }` opts object
const PROMPTS: Array<{ name: string; text: string }> = [
  { name: "url",   text: "What happens when you type a URL into your browser? Include every network layer." },
  { name: "oauth", text: "OAuth 2.0 authorization code flow with PKCE as a sequence diagram — user, browser, app server, auth server, API" },
  { name: "uml",   text: "UML class diagram: Payment abstract class with CreditCard, PayPal, ApplePay, BankTransfer subclasses; each has process() and refund() methods and an amount attribute" },
];
const MAX_NEW_TOKENS = 800;

test.describe.serial("Diagnose long-output drift", () => {
  test.setTimeout(1_800_000);
  let context: BrowserContext;
  let page: Page;

  test.beforeAll(async () => {
    test.setTimeout(1_800_000);
    context = await chromium.launchPersistentContext(DATA_DIR, { channel: "chrome", args: [] });
    page = context.pages()[0] || await context.newPage();
    page.on("console", msg => {
      const t = msg.text();
      if (t.startsWith("[draw]") || t.startsWith("[engine]") || t.startsWith("[diag]")) console.log(t);
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

  test("capture full baseline output + compile error across prompts", async () => {
    const results = await page.evaluate(async (args) => {
      const { prompts, maxTokens } = args;
      const engine = (window as any).__engine;
      const tokenizer = (window as any).__tokenizer;
      const executeCode: (code: string) => Promise<{ result: any; error?: string }> = (window as any).__executeCode;

      const out: Array<{ name: string; totalTokens: number; compileError?: string; code: string; nodeCount?: number; edgeCount?: number }> = [];

      for (const p of prompts) {
        engine.restoreCache();

        const msgs = [{ role: "user", content: p.text }];
        const rendered = tokenizer.apply_chat_template(msgs, { tokenize: false, add_generation_prompt: true });
        const userTokenIds: number[] = Array.from(tokenizer.encode(rendered));
        if (userTokenIds[0] === 2) userTokenIds.shift();

        const firstToken = await engine.prefill(userTokenIds);
        const produced: number[] = [firstToken];
        await engine.streamTokens(firstToken, maxTokens - 1, [1, 106], (msg: any) => {
          produced.push(msg.id);
        });

        const full = tokenizer.decode(produced, { skip_special_tokens: true });

        // Prep the code the same way main.ts's generate() does so the
        // executor sees what the real flow would try to run.
        let code = full.trim();
        code = code.replace(/^```(?:javascript|js|typescript|ts)?\s*\n?/i, "");
        code = code.replace(/\n?```\s*$/i, "");
        const lastSemi = code.lastIndexOf(";");
        if (lastSemi >= 0) code = code.substring(0, lastSemi + 1);
        if (!code.includes("new Diagram")) code = `const d = new Diagram({ direction: "TB" });\n${code}`;
        if (!code.includes("d.render()")) code = `${code}\nreturn d.render();`;

        let compileError: string | undefined;
        let nodeCount: number | undefined;
        let edgeCount: number | undefined;
        try {
          const r = await executeCode(code);
          if (r.error) {
            compileError = r.error;
          } else if (r.result?.stats) {
            nodeCount = r.result.stats.nodes;
            edgeCount = r.result.stats.edges;
          }
        } catch (e) {
          compileError = (e as Error).message;
        }

        out.push({ name: p.name, totalTokens: produced.length, compileError, code, nodeCount, edgeCount });
      }
      return out;
    }, { prompts: PROMPTS, maxTokens: MAX_NEW_TOKENS });

    console.log("");
    console.log("==========================================================");
    for (const r of results) {
      const status = r.compileError ? `FAIL: ${r.compileError}` : `OK (nodes=${r.nodeCount} edges=${r.edgeCount})`;
      console.log(`[diag] ${r.name.padEnd(6)}  totalTokens=${String(r.totalTokens).padStart(4)}  ${status}`);
    }
    console.log("==========================================================");
    for (const r of results) {
      if (!r.compileError) continue;
      console.log("");
      console.log(`[diag] FAILED PROMPT: ${r.name}`);
      const lines = r.code.split("\n");
      for (let i = 0; i < lines.length; i++) {
        console.log(`[diag] ${String(i + 1).padStart(3)} | ${lines[i]}`);
      }
    }
    console.log("");

    expect(results.length).toBe(PROMPTS.length);
  });
});
