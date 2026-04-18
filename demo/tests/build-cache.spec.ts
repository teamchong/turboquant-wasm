/**
 * Build step: run the engine against the fixed system prompt, dump the
 * resulting KV cache, and save it to `demo/public/system-cache.bin`.
 *
 * Run manually with:
 *   cd demo && npx playwright test tests/build-cache.spec.ts
 *
 * Rerun this whenever the system prompt, model, or TQ polar config changes.
 * The runtime path verifies a hash of the prompt tokens at load time, so a
 * stale cache will be rejected (the worker then falls back to prefill).
 */

import { test, chromium, type BrowserContext, type Page } from "@playwright/test";
import { resolve, dirname } from "path";
import { writeFileSync, existsSync, readdirSync, rmSync } from "fs";
import { execSync } from "child_process";
import { fileURLToPath } from "url";

const HERE = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = resolve(HERE, "..", ".playwright-data");
const OUTPUT_PATH = resolve(HERE, "..", "public", "system-cache.bin");
// Vite auto-reallocates if 5173 is busy; the cache-regen spawner exports the
// chosen port so the test can find it regardless of who started vite.
const DEV_PORT = process.env.PLAYWRIGHT_DEV_PORT ?? "5173";
const BASE_URL = `http://localhost:${DEV_PORT}`;

/**
 * Chrome's persistent-profile model writes SingletonLock / SingletonCookie /
 * SingletonSocket at profile root to prevent two Chrome instances from
 * trampling each other. If a previous run was killed (e.g. playwright
 * webServer timeout) these files survive and the next launch fails with
 * "profile already in use" — the error from the regen log above. The OS
 * normally cleans these on process exit but abrupt kills bypass that path.
 * Drop any stale ones on startup so we're never blocked by a ghost of a
 * prior run.
 */
function clearStaleSingletons(dataDir: string): void {
  if (!existsSync(dataDir)) return;
  for (const entry of readdirSync(dataDir)) {
    if (entry.startsWith("Singleton")) {
      try { rmSync(resolve(dataDir, entry), { force: true, recursive: false }); } catch { /* ignore */ }
    }
  }
}

/**
 * Hard-kill any Chrome process that was launched with our persistent-profile
 * data dir. Belt-and-suspenders for context.close(): if close hangs or an
 * exception crashes the test between launch and afterAll, the chrome process
 * (holding the 3 GB model in memory + the SingletonLock on disk) would
 * otherwise linger and block the next regen. `pkill -9 -f "<dataDir>"` matches
 * only this test's chrome — the data-dir path is unique enough not to
 * collide with other browsers.
 */
function killOwnedBrowsers(dataDir: string): void {
  try {
    // Trailing `|| true` so a no-match (nothing to kill) doesn't throw.
    execSync(`pkill -9 -f "${dataDir}" 2>/dev/null || true`, { stdio: "ignore" });
  } catch { /* platform mismatch (Windows); ignore */ }
}

test.describe.serial("Build system prompt cache", () => {
  test.setTimeout(1_800_000);
  let context: BrowserContext;
  let page: Page;

  test.beforeAll(async () => {
    test.setTimeout(1_800_000);
    clearStaleSingletons(DATA_DIR);
    context = await chromium.launchPersistentContext(DATA_DIR, { channel: "chrome", args: [] });
    page = context.pages()[0] || await context.newPage();
    page.on("console", msg => {
      const t = msg.text();
      if (t.startsWith("[draw]") || t.startsWith("[engine]") || t.startsWith("[worker]")) console.log(t);
    });
    page.on("pageerror", err => console.log(`[pageerror] ${err.message}`));
    // ?noCache=1 skips the fetch for an existing system-cache.bin so we build
    // from scratch (otherwise the worker would just re-load whatever is there).
    await page.goto(`${BASE_URL}/draw.html?noCache=1`);

    // Heartbeat every 5s so the user watching `bun run build` output can see
    // that we're waiting on the engine, not hung. Without this, the stretch
    // between "[engine] Engine ready" and "[worker] prefill 100%" is quiet
    // enough to feel frozen — triggering Ctrl+C, which leaves orphan
    // chromes and an empty system-cache.bin.
    const heartbeatStart = Date.now();
    const heartbeat = setInterval(() => {
      const s = Math.floor((Date.now() - heartbeatStart) / 1000);
      console.log(`[build-cache] waiting for status=Ready... ${s}s elapsed`);
    }, 5000);
    try {
      await page.waitForFunction(
        () => document.querySelector("#status")?.textContent === "Ready",
        {}, { timeout: 1_500_000 },
      );
    } finally {
      clearInterval(heartbeat);
    }
  });

  test.afterAll(async () => {
    test.setTimeout(120_000);
    // Graceful close first — gives Chrome a chance to flush OPFS writes and
    // release file handles cleanly. Cap it at 30s so a hung teardown can't
    // wedge the whole build pipeline.
    if (context) {
      await Promise.race([
        context.close().catch(() => { /* already dying */ }),
        new Promise<void>((r) => setTimeout(r, 30_000)),
      ]);
    }
    // Hard kill any survivors. This catches:
    //   - a context.close() that timed out above
    //   - a test failure between launch and afterAll (pageerror, evaluate throw)
    //   - renderer / utility processes that forked off the main chrome and
    //     detached from the persistent-context lifecycle
    // All of those would otherwise leak ~1-3 GB of RSS (model weights + WebGPU
    // buffers) until the OS eventually reaps them.
    killOwnedBrowsers(DATA_DIR);
    // Once everyone is dead the SingletonLock is stale by definition — drop it
    // so the next run isn't blocked even if pkill missed something.
    clearStaleSingletons(DATA_DIR);
  });

  test("prefill system prompt and dump KV cache", async () => {
    // Describe-level setTimeout can miss in some Playwright versions — set
    // the test-body timeout explicitly. Dumping ~150 MB of KV + base64
    // encoding over the RPC boundary takes 10-30s on M1.
    test.setTimeout(1_800_000);
    console.log("[build-cache] dumping KV cache...");
    const cacheB64 = await page.evaluate(async () => {
      const engine = (window as any).__engine;
      // Snapshot is already taken inside the worker's init path right after
      // the system-prompt prefill completes; dumpSystemCache() reads from the
      // current in-flight cache (which is empty right after cache.restoreCache)
      // so we restore first to be explicit about what we're dumping.
      engine.restoreCache();
      const buf: ArrayBuffer = await engine.dumpSystemCache();
      // Return as base64 so we can carry it over the Playwright RPC boundary.
      let binary = "";
      const bytes = new Uint8Array(buf);
      const CHUNK = 0x8000;
      for (let i = 0; i < bytes.length; i += CHUNK) {
        binary += String.fromCharCode.apply(null, Array.from(bytes.subarray(i, i + CHUNK)));
      }
      return btoa(binary);
    });

    const bytes = Buffer.from(cacheB64, "base64");
    writeFileSync(OUTPUT_PATH, bytes);
    console.log(`[build-cache] wrote ${(bytes.length / 1e6).toFixed(1)} MB → ${OUTPUT_PATH}`);
  });
});
