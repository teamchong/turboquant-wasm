/**
 * Tests for the addClass primitive (UML class diagrams) and the new
 * relation-based arrowhead mapping on connect().
 *
 * Phase 3 of option C: addClass + relation. Verifies that:
 *   1. addClass renders a 3-section box (header + attributes + methods)
 *   2. Visibility sigils (+ - # ~) appear on member rows
 *   3. relation: "inheritance" sets the expected arrowhead pair
 *   4. relation: "composition" / "aggregation" set the diamond markers
 *   5. relation: "dependency" sets dashed style
 *   6. toCode round-trips a class back to addClass(...) source
 *   7. Multi-class layout has no overlaps
 */

import { test, expect, chromium, type BrowserContext, type Page } from "@playwright/test";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const HERE = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = resolve(HERE, "..", ".playwright-data");

test.describe.serial("SDK UML class primitive", () => {
  test.setTimeout(1_800_000);
  let context: BrowserContext;
  let page: Page;

  test.beforeAll(async () => {
    test.setTimeout(1_800_000);
    context = await chromium.launchPersistentContext(DATA_DIR, { channel: "chrome", args: [] });
    page = context.pages()[0] || await context.newPage();
    page.on("console", msg => {
      const t = msg.text();
      if (t.startsWith("[draw]") || t.startsWith("[uml-test]")) console.log(t);
    });
    page.on("pageerror", err => console.log(`[pageerror] ${err.message}`));
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

  test("addClass renders 3-section box: header + attributes + methods + 2 thick dividers", async () => {
    const result = await page.evaluate(async () => {
      const code = `
        const d = new Diagram({ direction: "TB" });
        d.addClass("User", {
          attributes: [
            { name: "id", type: "int" },
            { name: "name", type: "string" },
            { name: "passwordHash", type: "string", visibility: "private" },
          ],
          methods: [
            { name: "login()", type: "bool" },
            { name: "logout()", type: "void" },
          ],
        }, { row: 1, col: 1 });
        return d.render();
      `;
      const exec = (window as any).__executeCode;
      const { result, error } = await exec(code);
      if (error) return { ok: false, error: String(error) };
      const elements = (result?.json?.elements ?? []) as any[];
      const rects = elements.filter(e => e.type === "rectangle" && e.id.startsWith("cls_"));
      const headers = elements.filter(e => e.type === "text" && e.id.endsWith("-header"));
      const rows = elements.filter(e => e.type === "text" && /-row\d+$/.test(e.id));
      const dividers = elements.filter(e => e.type === "line" && /-div\d+$/.test(e.id));
      const thickDividers = dividers.filter(d => d.strokeWidth >= 2);
      return {
        ok: true,
        rectCount: rects.length,
        headerText: headers[0]?.text ?? null,
        rowTexts: rows.map(r => r.text),
        totalDividers: dividers.length,
        thickDividers: thickDividers.length,
      };
    });
    console.log("[uml-test] addClass result:", JSON.stringify(result));
    expect(result.ok, `render must succeed (got error: ${(result as any).error})`).toBe(true);
    expect(result.rectCount).toBe(1);
    expect(result.headerText).toBe("User");
    // 5 body rows (3 attributes + 2 methods)
    expect(result.rowTexts.length).toBe(5);
    // Public attribute → "+" sigil
    expect(result.rowTexts[0]).toMatch(/^\+\s+id/);
    // Private attribute → "-" sigil
    expect(result.rowTexts[2]).toMatch(/^-\s+passwordHash/);
    // Methods are present and use "+" by default
    expect(result.rowTexts[3]).toContain("login()");
    expect(result.rowTexts[4]).toContain("logout()");
    // 2 thick dividers: header→attrs, attrs→methods
    expect(result.thickDividers, "header divider + section divider between attrs and methods").toBe(2);
  });

  test("addClass with attributes only (no methods) draws one thick divider", async () => {
    const result = await page.evaluate(async () => {
      const code = `
        const d = new Diagram();
        d.addClass("DTO", {
          attributes: [
            { name: "x", type: "int" },
            { name: "y", type: "int" },
          ],
        }, { row: 1, col: 1 });
        return d.render();
      `;
      const exec = (window as any).__executeCode;
      const { result, error } = await exec(code);
      if (error) return { ok: false, error: String(error) };
      const elements = (result?.json?.elements ?? []) as any[];
      const dividers = elements.filter(e => e.type === "line" && /-div\d+$/.test(e.id));
      const thickDividers = dividers.filter(d => d.strokeWidth >= 2);
      return { ok: true, thickDividers: thickDividers.length };
    });
    console.log("[uml-test] attrs-only result:", JSON.stringify(result));
    expect(result.ok).toBe(true);
    // Only one thick divider: header → attributes (no methods section).
    expect(result.thickDividers).toBe(1);
  });

  test("relation: inheritance sets endArrowhead triangle", async () => {
    const result = await page.evaluate(async () => {
      const code = `
        const d = new Diagram();
        const parent = d.addClass("Animal", { methods: [{ name: "speak()", type: "void" }] }, { row: 1, col: 1 });
        const child = d.addClass("Dog", { methods: [{ name: "speak()", type: "void" }] }, { row: 2, col: 1 });
        d.connect(child, parent, "extends", { relation: "inheritance" });
        return d.render();
      `;
      const exec = (window as any).__executeCode;
      const { result, error } = await exec(code);
      if (error) return { ok: false, error: String(error) };
      const elements = (result?.json?.elements ?? []) as any[];
      const arrow = elements.find(e => e.type === "arrow");
      return {
        ok: true,
        startArrowhead: arrow?.startArrowhead ?? null,
        endArrowhead: arrow?.endArrowhead ?? null,
        strokeStyle: arrow?.strokeStyle ?? null,
      };
    });
    console.log("[uml-test] inheritance result:", JSON.stringify(result));
    expect(result.ok).toBe(true);
    expect(result.endArrowhead).toBe("triangle");
    expect(result.startArrowhead).toBeNull();
    expect(result.strokeStyle).toBe("solid");
  });

  test("relation: composition / aggregation set diamond markers; dependency is dashed", async () => {
    const result = await page.evaluate(async () => {
      const code = `
        const d = new Diagram();
        const car = d.addClass("Car", { attributes: [{ name: "vin", type: "string" }] }, { row: 1, col: 1 });
        const engine = d.addClass("Engine", { attributes: [{ name: "hp", type: "int" }] }, { row: 1, col: 2 });
        const radio = d.addClass("Radio", { attributes: [{ name: "model", type: "string" }] }, { row: 1, col: 3 });
        const fuel = d.addClass("Fuel", { attributes: [{ name: "type", type: "string" }] }, { row: 2, col: 2 });
        d.connect(car, engine, "owns",     { relation: "composition" });
        d.connect(car, radio,  "has",      { relation: "aggregation" });
        d.connect(engine, fuel, "uses",    { relation: "dependency" });
        return d.render();
      `;
      const exec = (window as any).__executeCode;
      const { result, error } = await exec(code);
      if (error) return { ok: false, error: String(error) };
      const elements = (result?.json?.elements ?? []) as any[];
      const arrows = elements.filter(e => e.type === "arrow");
      return {
        ok: true,
        composition: { start: arrows[0]?.startArrowhead, end: arrows[0]?.endArrowhead, style: arrows[0]?.strokeStyle },
        aggregation: { start: arrows[1]?.startArrowhead, end: arrows[1]?.endArrowhead, style: arrows[1]?.strokeStyle },
        dependency:  { start: arrows[2]?.startArrowhead, end: arrows[2]?.endArrowhead, style: arrows[2]?.strokeStyle },
      };
    });
    console.log("[uml-test] relations result:", JSON.stringify(result));
    expect(result.ok).toBe(true);
    expect(result.composition.start).toBe("diamond");
    expect(result.composition.end).toBe("arrow");
    expect(result.composition.style).toBe("solid");
    expect(result.aggregation.start).toBe("diamond_outline");
    expect(result.aggregation.end).toBe("arrow");
    expect(result.aggregation.style).toBe("solid");
    expect(result.dependency.start).toBeNull();
    expect(result.dependency.end).toBe("arrow");
    expect(result.dependency.style).toBe("dashed");
  });

  test("toCode round-trips a class to addClass(...) source", async () => {
    const result = await page.evaluate(async () => {
      const code = `
        const d = new Diagram();
        d.addClass("Account", {
          attributes: [
            { name: "id", type: "int" },
            { name: "balance", type: "decimal", visibility: "private" },
          ],
          methods: [
            { name: "deposit()", type: "void" },
            { name: "withdraw()", type: "bool", visibility: "protected" },
          ],
        }, { row: 1, col: 1 });
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
    console.log("[uml-test] toCode result:", JSON.stringify(result));
    expect(result.error).toBeNull();
    expect(result.tsCode).toContain("d.addClass");
    expect(result.tsCode).toContain('"Account"');
    expect(result.tsCode).toContain("attributes:");
    expect(result.tsCode).toContain("methods:");
    expect(result.tsCode).toContain('visibility: "private"');
    expect(result.tsCode).toContain('visibility: "protected"');
    // Default visibility (public) should NOT be emitted explicitly.
    expect(result.tsCode).not.toContain('visibility: "public"');
    // No raw width/height literals — addClass computes them.
    expect(result.tsCode).not.toMatch(/width:\s*\d+/);
    expect(result.tsCode).not.toMatch(/height:\s*\d+/);
  });

  test("multi-class UML diagram with inheritance lays out without overlaps", async () => {
    const result = await page.evaluate(async () => {
      const code = `
        const d = new Diagram({ direction: "TB" });
        const animal = d.addClass("Animal", {
          attributes: [{ name: "name", type: "string" }],
          methods: [{ name: "speak()", type: "void" }],
        }, { row: 1, col: 2 });
        const dog = d.addClass("Dog", {
          attributes: [{ name: "breed", type: "string" }],
          methods: [{ name: "speak()", type: "void" }],
        }, { row: 2, col: 1 });
        const cat = d.addClass("Cat", {
          attributes: [{ name: "indoor", type: "bool" }],
          methods: [{ name: "speak()", type: "void" }],
        }, { row: 2, col: 3 });
        const puppy = d.addClass("Puppy", {
          attributes: [{ name: "age", type: "int" }],
        }, { row: 3, col: 1 });
        d.connect(dog,  animal, "extends", { relation: "inheritance" });
        d.connect(cat,  animal, "extends", { relation: "inheritance" });
        d.connect(puppy, dog,    "extends", { relation: "inheritance" });
        return d.render();
      `;
      const exec = (window as any).__executeCode;
      const { result, error } = await exec(code);
      if (error) return { ok: false, error: String(error) };
      const elements = (result?.json?.elements ?? []) as any[];
      const classRects = elements.filter(e => e.type === "rectangle" && e.id.startsWith("cls_"));
      const overlaps: string[] = [];
      for (let i = 0; i < classRects.length; i++) {
        for (let j = i + 1; j < classRects.length; j++) {
          const a = classRects[i], b = classRects[j];
          if (a.x < b.x + b.width && a.x + a.width > b.x && a.y < b.y + b.height && a.y + a.height > b.y) {
            overlaps.push(`${a.id} overlaps ${b.id}`);
          }
        }
      }
      const arrows = elements.filter(e => e.type === "arrow");
      return { ok: true, classCount: classRects.length, arrowCount: arrows.length, overlaps };
    });
    console.log("[uml-test] multi-class result:", JSON.stringify(result));
    expect(result.ok).toBe(true);
    expect(result.classCount).toBe(4);
    expect(result.arrowCount).toBe(3);
    expect(result.overlaps).toEqual([]);
  });
});
