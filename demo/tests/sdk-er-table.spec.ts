/**
 * Tests for the addTable primitive (ER diagram support).
 *
 * Phase 2 of option C: addTable + cardinality. Verifies that:
 *   1. addTable produces a single layout node sized for the columns
 *   2. Render emits the outer rect + header + per-row text + dividers
 *   3. connect with cardinality opt prefixes the cardinality to the label
 *   4. toCode round-trips a table back to addTable code
 *   5. Multi-table ER diagrams lay out without errors
 *
 * Runs in the browser via __executeCode so the bundled SDK + WASM layout
 * are exercised end-to-end.
 */

import { test, expect, chromium, type BrowserContext, type Page } from "@playwright/test";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const HERE = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = resolve(HERE, "..", ".playwright-data");

test.describe.serial("SDK ER table primitive", () => {
  test.setTimeout(1_800_000);
  let context: BrowserContext;
  let page: Page;

  test.beforeAll(async () => {
    test.setTimeout(1_800_000);
    context = await chromium.launchPersistentContext(DATA_DIR, { channel: "chrome", args: [] });
    page = context.pages()[0] || await context.newPage();
    page.on("console", msg => {
      const t = msg.text();
      if (t.startsWith("[draw]") || t.startsWith("[er-test]")) console.log(t);
    });
    page.on("pageerror", err => console.log(`[pageerror] ${err.message}`));
    // ?noCache=1 + ?skipSysPrompt=1 — we don't need the model loaded for SDK tests,
    // and we want to bypass the stale system-cache.bin until phase 5 rebuilds it.
    await page.goto("http://localhost:5173/draw.html?skipSysPrompt=1&noCache=1");
    await page.waitForFunction(
      () => document.querySelector("#status")?.textContent === "Ready",
      {}, { timeout: 1_500_000 },
    );
  });

  test.afterAll(async () => {
    test.setTimeout(120_000);
    await context?.close();
  });

  test("addTable renders one outer rect + header + N rows + dividers", async () => {
    const result = await page.evaluate(async () => {
      const code = `
        const d = new Diagram({ direction: "LR" });
        const users = d.addTable("users", [
          { name: "id", type: "INT", key: "PK" },
          { name: "email", type: "VARCHAR(255)" },
          { name: "name", type: "VARCHAR(100)" },
        ], { row: 1, col: 1 });
        return d.render();
      `;
      const exec = (window as any).__executeCode;
      const { result, error } = await exec(code);
      if (error) return { ok: false, error: String(error) };
      const elements = (result?.json?.elements ?? []) as any[];
      // Expect: 1 outer rectangle (id starts with tbl_) + 1 header text + 4 lines (1 header div + 3 row dividers wait — N-1 = 2 row dividers between 3 rows)
      // Actually: 1 outer rect + 1 header text + 1 header divider + 3 row text + 2 inter-row dividers = 8
      const rects = elements.filter(e => e.type === "rectangle" && e.id.startsWith("tbl_"));
      const headers = elements.filter(e => e.type === "text" && e.id.endsWith("-header"));
      const rows = elements.filter(e => e.type === "text" && /-row\d+$/.test(e.id));
      const dividers = elements.filter(e => e.type === "line" && /-div\d+$/.test(e.id));
      return {
        ok: true,
        rectCount: rects.length,
        headerCount: headers.length,
        rowCount: rows.length,
        dividerCount: dividers.length,
        headerText: headers[0]?.text ?? null,
        rowTexts: rows.map(r => r.text),
        rectWidth: rects[0]?.width ?? null,
        rectHeight: rects[0]?.height ?? null,
      };
    });
    console.log("[er-test] addTable result:", JSON.stringify(result));
    expect(result.ok, `render must succeed (got error: ${(result as any).error})`).toBe(true);
    expect(result.rectCount, "exactly one outer table rectangle").toBe(1);
    expect(result.headerCount, "exactly one header text").toBe(1);
    expect(result.rowCount, "one text per column row").toBe(3);
    // Dividers: 1 below header + 2 between rows (3 columns → 2 inter-row dividers)
    expect(result.dividerCount, "header divider + N-1 row dividers").toBe(3);
    expect(result.headerText).toBe("users");
    expect(result.rowTexts[0], "PK column shows 🔑 prefix").toContain("🔑");
    expect(result.rowTexts[0]).toContain("id");
    expect(result.rowTexts[0]).toContain("INT");
    expect(result.rowTexts[1]).toContain("email");
    expect(result.rowTexts[1]).toContain("VARCHAR(255)");
    expect(result.rectWidth, "table width should fit the widest row").toBeGreaterThan(150);
    expect(result.rectHeight, "table height should be header + rows").toBeGreaterThan(60);
  });

  test("connect cardinality prefixes the edge label with the cardinality marker", async () => {
    const result = await page.evaluate(async () => {
      const code = `
        const d = new Diagram({ direction: "LR" });
        const users = d.addTable("users", [
          { name: "id", type: "INT", key: "PK" },
        ], { row: 1, col: 1 });
        const orders = d.addTable("orders", [
          { name: "id", type: "INT", key: "PK" },
          { name: "user_id", type: "INT", key: "FK" },
        ], { row: 1, col: 2 });
        d.connect(users, orders, "places", { cardinality: "1:N" });
        return d.render();
      `;
      const exec = (window as any).__executeCode;
      const { result, error } = await exec(code);
      if (error) return { ok: false, error: String(error) };
      const elements = (result?.json?.elements ?? []) as any[];
      const arrowLabel = elements.find(e => e.type === "text" && e.id.startsWith("arrlbl_"));
      return {
        ok: true,
        labelText: arrowLabel?.text ?? null,
      };
    });
    console.log("[er-test] cardinality result:", JSON.stringify(result));
    expect(result.ok, `render must succeed (got error: ${(result as any).error})`).toBe(true);
    expect(result.labelText, "arrow label exists").toBeTruthy();
    expect(result.labelText, "label is prefixed with cardinality").toContain("1:N");
    expect(result.labelText, "original verb still present").toContain("places");
  });

  test("toCode round-trips a table to addTable(...) source", async () => {
    const result = await page.evaluate(async () => {
      const code = `
        const d = new Diagram({ direction: "LR" });
        d.addTable("products", [
          { name: "id", type: "INT", key: "PK" },
          { name: "sku", type: "VARCHAR(50)" },
          { name: "price", type: "DECIMAL(10,2)" },
        ], { row: 1, col: 1 });
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
      };
    });
    console.log("[er-test] toCode result:", JSON.stringify(result));
    expect(result.error).toBeNull();
    expect(result.tsCode).toContain("d.addTable");
    expect(result.tsCode).toContain('"products"');
    expect(result.tsCode).toContain('name: "id"');
    expect(result.tsCode).toContain('key: "PK"');
    expect(result.tsCode).toContain('type: "VARCHAR(50)"');
    // Must NOT emit width/height literally — those are computed from columns.
    expect(result.tsCode).not.toMatch(/width:\s*\d+/);
    expect(result.tsCode).not.toMatch(/height:\s*\d+/);
  });

  test("multi-table ER diagram lays out without overlap", async () => {
    const result = await page.evaluate(async () => {
      const code = `
        const d = new Diagram({ direction: "LR" });
        const users = d.addTable("users", [
          { name: "id", type: "INT", key: "PK" },
          { name: "email", type: "VARCHAR(255)" },
        ], { row: 1, col: 1 });
        const orders = d.addTable("orders", [
          { name: "id", type: "INT", key: "PK" },
          { name: "user_id", type: "INT", key: "FK" },
          { name: "total", type: "DECIMAL(10,2)" },
        ], { row: 1, col: 2 });
        const items = d.addTable("order_items", [
          { name: "id", type: "INT", key: "PK" },
          { name: "order_id", type: "INT", key: "FK" },
          { name: "product_id", type: "INT", key: "FK" },
          { name: "qty", type: "INT" },
        ], { row: 1, col: 3 });
        const products = d.addTable("products", [
          { name: "id", type: "INT", key: "PK" },
          { name: "sku", type: "VARCHAR(50)" },
        ], { row: 1, col: 4 });
        d.connect(users, orders, "places", { cardinality: "1:N" });
        d.connect(orders, items, "contains", { cardinality: "1:N" });
        d.connect(items, products, "references", { cardinality: "N:1" });
        return d.render();
      `;
      const exec = (window as any).__executeCode;
      const { result, error } = await exec(code);
      if (error) return { ok: false, error: String(error) };
      const elements = (result?.json?.elements ?? []) as any[];
      const tableRects = elements.filter(e => e.type === "rectangle" && e.id.startsWith("tbl_"));
      // Check that no two outer table rectangles overlap.
      const overlaps: string[] = [];
      for (let i = 0; i < tableRects.length; i++) {
        for (let j = i + 1; j < tableRects.length; j++) {
          const a = tableRects[i], b = tableRects[j];
          if (a.x < b.x + b.width && a.x + a.width > b.x && a.y < b.y + b.height && a.y + a.height > b.y) {
            overlaps.push(`${a.id} overlaps ${b.id}`);
          }
        }
      }
      const arrows = elements.filter(e => e.type === "arrow");
      return { ok: true, tableCount: tableRects.length, arrowCount: arrows.length, overlaps };
    });
    console.log("[er-test] multi-table result:", JSON.stringify(result));
    expect(result.ok, `render must succeed (got error: ${(result as any).error})`).toBe(true);
    expect(result.tableCount, "4 tables").toBe(4);
    expect(result.arrowCount, "3 relationship arrows").toBe(3);
    expect(result.overlaps, "no two tables should overlap").toEqual([]);
  });
});
