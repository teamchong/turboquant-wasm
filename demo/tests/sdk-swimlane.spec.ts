/**
 * Tests for the addLane primitive (swimlane / activity diagrams).
 *
 * Phase 4 of option C: addLane forces children to a specific row and
 * renders a labelled background band behind them. Verifies that:
 *   1. addLane locks every child to the lane's row index
 *   2. Render emits one background rect + one header rect + one label per lane
 *   3. Lane backgrounds sit BEFORE shape elements (z-ordered behind)
 *   4. Cross-lane connections route across the lane boundaries
 *   5. toCode round-trips lanes back to addLane(...) source
 *   6. addLane errors clearly on missing children
 */

import { test, expect, chromium, type BrowserContext, type Page } from "@playwright/test";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const HERE = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = resolve(HERE, "..", ".playwright-data");

test.describe.serial("SDK swimlane primitive", () => {
  test.setTimeout(1_800_000);
  let context: BrowserContext;
  let page: Page;

  test.beforeAll(async () => {
    test.setTimeout(1_800_000);
    context = await chromium.launchPersistentContext(DATA_DIR, { channel: "chrome", args: [] });
    page = context.pages()[0] || await context.newPage();
    page.on("console", msg => {
      const t = msg.text();
      if (t.startsWith("[draw]") || t.startsWith("[lane-test]")) console.log(t);
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

  test("addLane locks children to sequential rows and renders one band per lane", async () => {
    const result = await page.evaluate(async () => {
      const code = `
        const d = new Diagram({ direction: "LR" });
        const enter   = d.addBox("Enter", { col: 1, color: "frontend" });
        const submit  = d.addBox("Submit", { col: 2, color: "frontend" });
        const validate = d.addBox("Validate", { col: 3, color: "backend" });
        const respond  = d.addBox("Respond",  { col: 4, color: "backend" });
        d.addLane("User", [enter, submit]);
        d.addLane("Server", [validate, respond]);
        d.connect(submit, validate, "POST");
        d.connect(respond, enter, "200 OK");
        return d.render();
      `;
      const exec = (window as any).__executeCode;
      const { result, error } = await exec(code);
      if (error) return { ok: false, error: String(error) };
      const elements = (result?.json?.elements ?? []) as any[];
      // Lane elements: 2 bands + 2 header bgs + 2 labels = 6
      const laneBgs = elements.filter(e => e.type === "rectangle" && e.id.startsWith("lane_") && !e.id.endsWith("-header-bg"));
      const laneHeaderBgs = elements.filter(e => e.type === "rectangle" && e.id.endsWith("-header-bg"));
      const laneLabels = elements.filter(e => e.type === "text" && e.id.endsWith("-label"));
      // Box elements (4 user/server boxes)
      const boxRects = elements.filter(e => e.type === "rectangle" && e.id.startsWith("box_"));
      // Verify lane elements are emitted BEFORE box elements in the array
      // (so they sit behind in z-order). The first lane background's index
      // must be lower than the first box.
      const firstLaneIdx = elements.findIndex(e => e.type === "rectangle" && e.id.startsWith("lane_"));
      const firstBoxIdx = elements.findIndex(e => e.type === "rectangle" && e.id.startsWith("box_"));
      const arrows = elements.filter(e => e.type === "arrow");
      return {
        ok: true,
        laneBgCount: laneBgs.length,
        laneHeaderCount: laneHeaderBgs.length,
        laneLabelCount: laneLabels.length,
        boxCount: boxRects.length,
        labelTexts: laneLabels.map(l => l.text),
        zOrderCorrect: firstLaneIdx < firstBoxIdx && firstLaneIdx >= 0 && firstBoxIdx >= 0,
        arrowCount: arrows.length,
      };
    });
    console.log("[lane-test] basic result:", JSON.stringify(result));
    expect(result.ok, `render must succeed (got error: ${(result as any).error})`).toBe(true);
    expect(result.laneBgCount, "one background band per lane").toBe(2);
    expect(result.laneHeaderCount, "one header rect per lane").toBe(2);
    expect(result.laneLabelCount, "one label per lane").toBe(2);
    expect(result.labelTexts).toEqual(["User", "Server"]);
    expect(result.boxCount, "all four boxes still rendered").toBe(4);
    expect(result.zOrderCorrect, "lane backgrounds must precede box rects in element order").toBe(true);
    expect(result.arrowCount).toBe(2);
  });

  test("lane children share their lane's row, and box Y coords reflect the lane stacking", async () => {
    const result = await page.evaluate(async () => {
      const code = `
        const d = new Diagram({ direction: "LR" });
        const a = d.addBox("A", { col: 1 });
        const b = d.addBox("B", { col: 1 });
        const c = d.addBox("C", { col: 1 });
        d.addLane("Top",    [a]);
        d.addLane("Middle", [b]);
        d.addLane("Bottom", [c]);
        return d.render();
      `;
      const exec = (window as any).__executeCode;
      const { result, error } = await exec(code);
      if (error) return { ok: false, error: String(error) };
      const elements = (result?.json?.elements ?? []) as any[];
      const a = elements.find(e => e.type === "rectangle" && e.id.startsWith("box_") && e.boundElements?.some((b: any) => elements.find(t => t.id === b.id)?.text === "A"));
      const b = elements.find(e => e.type === "rectangle" && e.id.startsWith("box_") && e.boundElements?.some((b: any) => elements.find(t => t.id === b.id)?.text === "B"));
      const c = elements.find(e => e.type === "rectangle" && e.id.startsWith("box_") && e.boundElements?.some((b: any) => elements.find(t => t.id === b.id)?.text === "C"));
      return {
        ok: true,
        ay: a?.y ?? null,
        by: b?.y ?? null,
        cy: c?.y ?? null,
      };
    });
    console.log("[lane-test] row stacking result:", JSON.stringify(result));
    expect(result.ok).toBe(true);
    // Each box should be in a strictly lower row than the previous one.
    // Graphviz produces Y coordinates for TB ranks; each lane occupies
    // a separate rank so y(c) > y(b) > y(a).
    expect(result.ay).not.toBeNull();
    expect(result.by).toBeGreaterThan(result.ay!);
    expect(result.cy).toBeGreaterThan(result.by!);
  });

  test("toCode round-trips lanes back to addLane(...) source", async () => {
    const result = await page.evaluate(async () => {
      const code = `
        const d = new Diagram({ direction: "LR" });
        const validate = d.addBox("Validate", { col: 1, color: "backend" });
        const persist  = d.addBox("Persist",  { col: 2, color: "database" });
        d.addLane("API",      [validate]);
        d.addLane("Storage",  [persist]);
        d.connect(validate, persist, "INSERT");
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
    console.log("[lane-test] toCode result:", JSON.stringify(result));
    expect(result.error).toBeNull();
    expect(result.tsCode).toContain("d.addLane");
    expect(result.tsCode).toContain('"API"');
    expect(result.tsCode).toContain('"Storage"');
    // Lane order must match declaration order.
    const apiIdx = result.tsCode!.indexOf('addLane("API"');
    const storageIdx = result.tsCode!.indexOf('addLane("Storage"');
    expect(apiIdx).toBeGreaterThan(0);
    expect(storageIdx).toBeGreaterThan(apiIdx);
  });

  test("addLane silently drops unresolvable child ids instead of throwing", async () => {
    // Deferred-linkup semantics: addLane records raw refs and resolveDeferred
    // at render time filters out anything that doesn't match a real node.
    // A stale / typo'd id used to throw "child not found"; now it's
    // dropped silently so out-of-order builds + LLM typos don't nuke the
    // whole diagram. Caller still has to end up with *some* valid
    // structure — here the lane just renders with one real child and
    // the bogus id is a no-op.
    const result = await page.evaluate(async () => {
      const code = `
        const d = new Diagram();
        const a = d.addBox("A", { col: 1 });
        d.addLane("Lane", ["this-id-doesnt-exist", a]);
        return d.render();
      `;
      const exec = (window as any).__executeCode;
      const { result, error } = await exec(code);
      return { error: error ? String(error) : null, hasLane: !!(result?.json?.elements || []).find((el: any) => el.customData?._group) };
    });
    console.log("[lane-test] deferred-drop result:", result);
    expect(result.error).toBeNull();
    expect(result.hasLane).toBe(true);
  });
});
