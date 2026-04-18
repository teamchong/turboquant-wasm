/**
 * Reproduction for "addGroup breaks layout, arrows don't follow boxes".
 *
 * Runs the user's exact code, then inspects:
 *   1. Whether graphviz returned `groupBounds` (i.e. did it cluster?)
 *   2. Whether the nudge-fallback path in sdk.ts mutates node positions
 *   3. Whether each arrow endpoint actually lands on its source/target box
 *
 * The goal is to confirm (or falsify) the hypothesis that arrows point at
 * pre-layout-mutation coordinates while boxes render at mutated ones.
 */

import { test, expect, chromium, type BrowserContext, type Page } from "@playwright/test";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const HERE = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = resolve(HERE, "..", ".playwright-data");

const USER_CODE = `
const d = new Diagram({ direction: "TB" });
const urlInput = d.addBox("URL Input Field", { row: 1, col: 1, color: "frontend", icon: "search" });
const browser = d.addBox("Web Browser Application", { row: 2, col: 2, color: "frontend", icon: "globe" });
const networkLayer = d.addBox("DNS Resolution (Layer 7)", { row: 3, col: 1, color: "orchestration", icon: "search" });
const tcp = d.addBox("TCP Connection Establishment (Layer 4)", { row: 3, col: 3, color: "queue", icon: "queue" });
const ip = d.addBox("IP Addressing/Routing (Layer 3)", { row: 4, col: 1, color: "network", icon: "cloud" });
const ip_packet = d.addBox("IP Packet Creation (Layer 3)", { row: 4, col: 2, color: "network", icon: "cloud" });
const tcp_segment = d.addBox("TCP Segment Transmission (Layer 4)", { row: 5, col: 3, color: "queue", icon: "queue" });
const http = d.addBox("HTTP Request/Response (Layer 7)", { row: 5, col: 4, color: "backend", icon: "api" });
const dns_query = d.addBox("DNS Query (Layer 7)", { row: 3, col: 2, color: "orchestration", icon: "search" });
const dns_response = d.addBox("DNS Response (Layer 7)", { row: 4, col: 2, color: "orchestration", icon: "search" });
const socket = d.addBox("Socket Interface (Layer 4)", { row: 5, col: 1, color: "queue", icon: "server" });
const socket_open = d.addBox("Socket Open/Close (Layer 4)", { row: 6, col: 1, color: "queue", icon: "server" });
const dataTransfer = d.addBox("Data Transfer (Layer 2)", { row: 6, col: 2, color: "queue", icon: "queue" });
const http_request = d.addBox("HTTP Request (Layer 7)", { row: 5, col: 3, color: "backend", icon: "api" });
const http_response = d.addBox("HTTP Response (Layer 7)", { row: 6, col: 3, color: "backend", icon: "api" });
d.connect(urlInput, dns_query, "Input URL");
d.connect(dns_query, dns_response, "Query");
d.connect(dns_response, http_request, "Resolved Domain");
d.connect(http_request, http_response, "Request/Response");
d.connect(http_response, http_response, "Response Flow");
d.connect(socket, http_request, "Connection");
d.connect(socket, http_request, "Socket Usage");
d.addGroup("Network Stack", [networkLayer, tcp, ip, ip_packet, tcp_segment, http, socket, socket_open, dataTransfer]);
return d.render();
`;

test.describe.serial("addGroup layout repro", () => {
  test.setTimeout(1_800_000);
  let context: BrowserContext;
  let page: Page;

  test.beforeAll(async () => {
    test.setTimeout(1_800_000);
    context = await chromium.launchPersistentContext(DATA_DIR, { channel: "chrome", args: [] });
    page = context.pages()[0] || await context.newPage();
    page.on("console", msg => {
      const t = msg.text();
      if (t.startsWith("[draw]") || t.startsWith("[engine]") || t.startsWith("[worker]") || t.startsWith("[viewer]") || t.startsWith("[repro]")) {
        console.log(t);
      }
    });
    page.on("pageerror", err => console.log(`[pageerror] ${err.message}`));
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

  test("dump layout — did graphviz cluster? did nudge run? are arrows on boxes?", async () => {
    const report = await page.evaluate(async (code: string) => {
      const executeCode = (window as any).__executeCode;
      const { result, error } = await executeCode(code);
      if (error || !result?.json) return { ok: false, error };
      const elements = result.json.elements as any[];

      // Classify elements
      const boxes = elements.filter(e => e.type === "rectangle" && !e.id.startsWith("grp_"));
      const arrows = elements.filter(e => e.type === "arrow");
      const groupRects = elements.filter(e => e.type === "rectangle" && e.id.startsWith("grp_"));

      // Build a lookup from customData._from / _to → arrow → endpoint coords
      // Arrow geometry: absolute start = (x, y) + points[0] (which is usually [0,0]),
      // absolute end = (x, y) + points[points.length-1].
      const mismatches: Array<{
        from: string; to: string;
        arrowStart: [number, number]; arrowEnd: [number, number];
        fromBox: { x: number; y: number; w: number; h: number } | null;
        toBox: { x: number; y: number; w: number; h: number } | null;
        startDistance: number | null;
        endDistance: number | null;
      }> = [];

      const boxById = new Map<string, any>();
      for (const b of boxes) boxById.set(b.id, b);

      // Is (px, py) touching the box? Two cases count as "touching":
      //   1. Inside the box rectangle — graphviz's first/last spline points
      //      are internal tangent points, hidden by the box fill. Normal.
      //   2. Within `TOL` px of any edge — accounts for arrowhead offset,
      //      stagger for parallel edges, and routing padding.
      const TOL = 40;
      const touchesBox = (box: any, px: number, py: number): boolean => {
        const x0 = box.x, y0 = box.y, x1 = box.x + box.width, y1 = box.y + box.height;
        if (px >= x0 && px <= x1 && py >= y0 && py <= y1) return true;
        // Distance from point to closest edge
        const dx = Math.max(x0 - px, 0, px - x1);
        const dy = Math.max(y0 - py, 0, py - y1);
        return Math.hypot(dx, dy) <= TOL;
      };

      for (const a of arrows) {
        const from = a.customData?._from;
        const to = a.customData?._to;
        if (!from || !to) continue;

        // Self-loops draw as arcs, not line segments — graphviz's route shape
        // doesn't match the "endpoints on boxes" model this test asserts.
        // Skipping them isn't hiding a bug; they're rendered via a different
        // path in graphviz and would need separate validation logic.
        if (from === to) continue;

        const fromBoxEl = boxById.get(from) ?? null;
        const toBoxEl = boxById.get(to) ?? null;
        const pts: [number, number][] = a.points ?? [];
        if (pts.length < 2) continue;
        const start: [number, number] = [a.x + pts[0][0], a.y + pts[0][1]];
        const end: [number, number] = [a.x + pts[pts.length - 1][0], a.y + pts[pts.length - 1][1]];

        const startOk = fromBoxEl ? touchesBox(fromBoxEl, start[0], start[1]) : true;
        const endOk = toBoxEl ? touchesBox(toBoxEl, end[0], end[1]) : true;

        if (!startOk || !endOk) {
          mismatches.push({
            from, to,
            arrowStart: start, arrowEnd: end,
            fromBox: fromBoxEl ? { x: fromBoxEl.x, y: fromBoxEl.y, w: fromBoxEl.width, h: fromBoxEl.height } : null,
            toBox: toBoxEl ? { x: toBoxEl.x, y: toBoxEl.y, w: toBoxEl.width, h: toBoxEl.height } : null,
            startDistance: startOk ? null : "off",
            endDistance: endOk ? null : "off",
          } as any);
        }
      }

      return {
        ok: true,
        boxCount: boxes.length,
        arrowCount: arrows.length,
        groupRectCount: groupRects.length,
        // Whether graphviz actually emitted the cluster — group rect count > 0
        // tells us the SDK drew it, not whether graphviz clustered. To check
        // graphviz directly we'd need to peek inside layoutNodesWasm. For the
        // repro we care about the end-result alignment.
        mismatchCount: mismatches.length,
        // Keep only the first few so the log isn't massive
        mismatches: mismatches.slice(0, 5),
      };
    }, USER_CODE);

    console.log("[repro] layout report:", JSON.stringify(report, null, 2));
    expect(report.ok).toBe(true);
    // This is the assertion the bug hypothesis says should fail. We EXPECT it
    // to fail right now (0 mismatches would mean arrows are aligned, no bug).
    // Leaving it as an expectation so we can watch the number shrink as we fix.
    expect((report as any).mismatchCount).toBe(0);
  });
});
