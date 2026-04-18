/**
 * Reproduces the "source node not found 'undefined'" failure that fired when
 * the model wrote `d.connect(d.getNode("User Input"), d.getNode("..."))`
 * against boxes that had been added with an icon set.
 *
 * Before the fix, addShape stored the icon-prefixed string ("🔍\nUser Input")
 * as node.label, so _resolveNodeRef("User Input") fell off the end and
 * getNode returned undefined — connect then threw the cryptic error.
 *
 * After the fix, addShape stores the canonical label and the rendered text
 * (with icon prefix) lives on displayLabel, so getNode-by-label resolves.
 *
 * The assertions run inside the browser via page.evaluate so they exercise
 * the actual bundled SDK (which depends on the layout WASM module).
 */

import { test, expect, chromium, type BrowserContext, type Page } from "@playwright/test";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const HERE = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = resolve(HERE, "..", ".playwright-data");

test.describe.serial("SDK label collision repro", () => {
  test.setTimeout(1_800_000);
  let context: BrowserContext;
  let page: Page;

  test.beforeAll(async () => {
    test.setTimeout(1_800_000);
    context = await chromium.launchPersistentContext(DATA_DIR, { channel: "chrome", args: [] });
    page = context.pages()[0] || await context.newPage();
    page.on("console", msg => {
      const t = msg.text();
      if (t.startsWith("[draw]") || t.startsWith("[label-test]")) console.log(t);
    });
    page.on("pageerror", err => console.log(`[pageerror] ${err.message}`));
    // skipSysPrompt=1 avoids the long system-prompt prefill so the test starts
    // generating immediately — we don't need the model loaded for SDK assertions,
    // but the page needs to reach Ready so the executor module is initialised.
    await page.goto("http://localhost:5173/draw.html?skipSysPrompt=1");
    await page.waitForFunction(
      () => document.querySelector("#status")?.textContent === "Ready",
      {}, { timeout: 1_500_000 },
    );
  });

  test.afterAll(async () => {
    test.setTimeout(120_000);
    await context?.close();
  });

  test("getNode resolves canonical label after addBox with icon (exact original repro)", async () => {
    // Mirrors the user-reported failure: model writes addBox without const,
    // then connects via getNode-by-label. With the bug, getNode returned
    // undefined and connect threw "source node not found 'undefined'".
    const result = await page.evaluate(async () => {
      const code = `
        const d = new Diagram({ direction: "TB" });
        d.addBox("User Input", { row: 1, col: 1, color: "users", icon: "search" });
        d.addBox("Orchestration Layer", { row: 3, col: 2, color: "orchestration", icon: "code" });
        d.connect(d.getNode("User Input"), d.getNode("Orchestration Layer"), "Initiate Request");
        return d.render();
      `;
      const exec = (window as any).__executeCode;
      const { result, error } = await exec(code);
      return {
        ok: !error,
        error: error ? String(error) : null,
        elementCount: result?.json?.elements?.length ?? 0,
      };
    });
    console.log("[label-test] original repro result:", JSON.stringify(result));
    expect(result.error, "the original failing snippet must execute cleanly").toBeNull();
    expect(result.ok).toBe(true);
    expect(result.elementCount, "should produce boxes + arrow elements").toBeGreaterThan(0);
  });

  test("toCode round-trip preserves canonical label without leaking the icon emoji", async () => {
    const result = await page.evaluate(async () => {
      // executeCode requires the snippet to return a RenderResult-shaped
      // object; smuggle the toCode output through an extra field so we can
      // assert on it without bypassing the existing test-host plumbing.
      const code = `
        const d = new Diagram({ direction: "TB" });
        d.addBox("API Gateway", { row: 1, col: 1, color: "external", icon: "api" });
        const tsCode = d.toCode();
        const r = await d.render();
        return { ...r, _tsCode: tsCode };
      `;
      const exec = (window as any).__executeCode;
      const { result, error } = await exec(code);
      return {
        ok: !error,
        error: error ? String(error) : null,
        tsCode: (result as any)?._tsCode ?? null,
        renderedText: (result?.json?.elements ?? []).find((e: any) => e.type === "text" && e.containerId)?.text ?? null,
      };
    });
    console.log("[label-test] toCode result:", JSON.stringify(result));
    expect(result.error).toBeNull();
    expect(result.tsCode, "toCode must return a string").toBeTruthy();
    // toCode should round-trip the canonical label and the icon opt — and
    // must NOT leak the emoji prefix into the label string.
    expect(result.tsCode).toContain('"API Gateway"');
    expect(result.tsCode).toContain('icon: "api"');
    // No "\n" inside the label literal — the emoji prefix used to land here.
    expect(result.tsCode).not.toMatch(/"[^"]*\\n[^"]*API Gateway/);
    // Sanity check: the rendered text element DOES include the emoji + label,
    // proving the displayLabel split kept render output identical.
    expect(result.renderedText, "rendered text should include the icon prefix + label").toContain("API Gateway");
  });

  test("connect with undefined from/to throws a clear, actionable error", async () => {
    const result = await page.evaluate(async () => {
      const code = `
        const d = new Diagram({ direction: "TB" });
        d.addBox("A", { row: 1, col: 1 });
        d.connect(d.getNode("does not exist"), "A");
        return d.render();
      `;
      const exec = (window as any).__executeCode;
      const { error } = await exec(code);
      return error ? String(error) : null;
    });
    console.log("[label-test] undefined-from error:", result);
    // The error message must mention the cause (undefined / not a string) so
    // the next person hitting this gets a clue, not the old cryptic cascade.
    expect(result, "passing undefined to connect must throw").not.toBeNull();
    expect(result).toMatch(/'from' must be a string/);
  });
});
