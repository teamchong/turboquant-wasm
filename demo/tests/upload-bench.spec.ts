/**
 * Benchmark tensor-upload strategies.
 *
 * Opens draw.html with ?benchUpload=1 so the worker runs every strategy in
 * `benchmarkTensorUpload` and logs timings. Run with:
 *
 *   cd demo && npx playwright test tests/upload-bench.spec.ts
 *
 * Look for lines like `[worker] upload bench:` in the console output.
 */

import { test, chromium, type BrowserContext, type Page } from "@playwright/test";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const HERE = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = resolve(HERE, "..", ".playwright-data");

test.describe.serial("Tensor upload benchmark", () => {
  test.setTimeout(1_800_000);
  let context: BrowserContext;
  let page: Page;
  const benchLines: string[] = [];

  test.beforeAll(async () => {
    test.setTimeout(1_800_000);
    context = await chromium.launchPersistentContext(DATA_DIR, { channel: "chrome", args: [] });
    page = context.pages()[0] || await context.newPage();
    page.on("console", msg => {
      const t = msg.text();
      if (t.startsWith("[worker]") || t.startsWith("[draw]") || t.startsWith("[engine]")) console.log(t);
      if (t.startsWith("[worker] upload bench") || t.startsWith("  serial") || t.startsWith("  parallel") || t.startsWith("  chunked") || t.startsWith("  syncAccess")) {
        benchLines.push(t);
      }
    });
    page.on("pageerror", err => console.log(`[pageerror] ${err.message}`));
    // skipSysPrompt=1 so the worker doesn't wait on the ~180 s prefill just
    // to benchmark the upload path.
    await page.goto("http://localhost:5173/draw.html?benchUpload=1&skipSysPrompt=1");
    // Wait long enough for the slowest strategy (~3 GB at 0.25 GB/s ≈ 12s)
    // times 7 strategies = ~90 s, plus slack.
    await page.waitForFunction(
      () => document.querySelector("#status")?.textContent === "Ready",
      {}, { timeout: 1_500_000 },
    );
  });

  test.afterAll(async () => {
    test.setTimeout(120_000);
    await context?.close();
  });

  test("run + summarize tensor upload strategies", async () => {
    console.log("");
    console.log("=========================================");
    console.log("TENSOR UPLOAD BENCHMARK SUMMARY");
    console.log("=========================================");
    for (const line of benchLines) console.log(line);
    console.log("=========================================");
  });
});
