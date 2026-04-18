/**
 * Reproduces the "Failed to execute 'mapAsync' on 'GPUBuffer': Buffer already
 * has an outstanding map pending" race that fired when the user clicked a
 * suggestion (which kicks off an eager prefill) and immediately clicked
 * Generate Diagram (which kicks off another prefill on top of it).
 *
 * Without the fix the worker would dispatch both messages, the second
 * prefill's _stagingBuf.mapAsync would land before the first one unmapped,
 * and the second prefill would reject with the WebGPU error.
 *
 * The fix adds a Promise queue in engine-worker.ts that serializes every
 * engine-touching message so the second prefill waits for the first to
 * finish its readback. This test posts two prefill messages back-to-back
 * (without awaiting) and asserts both complete without an error event.
 */

import { test, expect, chromium, type BrowserContext, type Page } from "@playwright/test";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const HERE = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = resolve(HERE, "..", ".playwright-data");

test.describe.serial("mapAsync race repro", () => {
  test.setTimeout(1_800_000);
  let context: BrowserContext;
  let page: Page;

  test.beforeAll(async () => {
    test.setTimeout(1_800_000);
    context = await chromium.launchPersistentContext(DATA_DIR, { channel: "chrome", args: [] });
    page = context.pages()[0] || await context.newPage();
    page.on("console", msg => {
      const t = msg.text();
      if (t.startsWith("[draw]") || t.startsWith("[engine]") || t.startsWith("[worker]") || t.startsWith("[race]")) {
        console.log(t);
      }
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

  test("two back-to-back prefill messages both succeed", async () => {
    const result = await page.evaluate(async () => {
      const w: Worker = (window as any).__engineWorker;
      // Reset cache to a known state so both prefills start from the same point.
      w.postMessage({ type: "resetCache" });

      // Two distinct prefill payloads. The exact tokens don't matter — only
      // that each one runs a forward pass that hits _stagingBuf.mapAsync.
      const tokensA = [106, 1645, 108, 147791, 107];
      const tokensB = [106, 1645, 108, 12345, 107];

      // Collect every response from the worker so we can tell whether the
      // race threw an error event (old behaviour) or both completed cleanly
      // (new behaviour).
      const events: Array<{ type: string; message?: string; firstToken?: number }> = [];
      const done = new Promise<void>((resolve, reject) => {
        let prefillDoneCount = 0;
        const handler = (e: MessageEvent) => {
          const r = e.data;
          if (r.type === "error") {
            events.push({ type: "error", message: r.message });
            w.removeEventListener("message", handler);
            // Don't reject — record the failure and let the test assert on
            // the events list so we get a clear error message about the race.
            resolve();
            return;
          }
          if (r.type === "prefillDone") {
            prefillDoneCount++;
            events.push({ type: "prefillDone", firstToken: r.firstToken });
            console.log(`[race] prefillDone #${prefillDoneCount}, firstToken=${r.firstToken}`);
            if (prefillDoneCount === 2) {
              w.removeEventListener("message", handler);
              resolve();
            }
          }
        };
        w.addEventListener("message", handler);
        // Give up after 60s so a stuck queue surfaces as a test failure
        // rather than hanging the suite.
        setTimeout(() => {
          w.removeEventListener("message", handler);
          reject(new Error(`timed out — only saw ${prefillDoneCount}/2 prefillDone events`));
        }, 60_000);
      });

      // Post both prefill messages back-to-back without awaiting either.
      // This is exactly what main.ts used to do when the user clicked a
      // suggestion and then Generate Diagram inside the same tick.
      console.log("[race] posting prefill A and B back-to-back");
      w.postMessage({ type: "prefill", tokenIds: tokensA });
      w.postMessage({ type: "prefill", tokenIds: tokensB });

      await done;
      return events;
    });

    console.log("[race] events:", JSON.stringify(result, null, 2));
    const errors = result.filter(e => e.type === "error");
    const completed = result.filter(e => e.type === "prefillDone");
    expect(errors, `worker should not error during back-to-back prefill (got: ${errors.map(e => e.message).join("; ")})`).toEqual([]);
    expect(completed.length, "both prefill messages should complete").toBe(2);
    for (const c of completed) {
      expect(c.firstToken, "each prefillDone should have a valid first token").toBeGreaterThan(0);
      expect(c.firstToken, "each first token should be in Gemma vocab").toBeLessThan(262144);
    }
  });
});
