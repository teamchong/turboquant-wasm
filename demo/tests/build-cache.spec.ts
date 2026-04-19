/**
 * Build step: run the engine against each branch's system prompt, dump the
 * resulting KV caches, pack them into a multi-branch TQKC container, and
 * save to `demo/public/system-cache.bin`.
 *
 * Run manually with:
 *   cd demo && npx playwright test tests/build-cache.spec.ts
 *
 * Rerun this whenever any branch prompt, the model, or TQ polar config
 * changes. The runtime path verifies a hash of the prompt tokens at load
 * time, so a stale cache will be rejected per-branch.
 *
 * Branches produced (in order):
 *   router       — mode-picker preamble
 *   sequence     — full sequence-mode docs + SDK typedef
 *   architecture — full arch-mode docs (arch/flowchart/class/ER/swimlane)
 */

import { test, chromium, type BrowserContext, type Page } from "@playwright/test";
import { resolve, dirname } from "path";
import { writeFileSync, existsSync, readdirSync, rmSync } from "fs";
import { execSync } from "child_process";
import { fileURLToPath } from "url";
import { packContainer, type BranchEntry } from "../src/draw/system-cache-container";

const HERE = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = resolve(HERE, "..", ".playwright-data");
const OUTPUT_PATH = resolve(HERE, "..", "public", "system-cache.bin");
// Vite auto-reallocates if 5173 is busy; the cache-regen spawner exports the
// chosen port so the test can find it regardless of who started vite.
const DEV_PORT = process.env.PLAYWRIGHT_DEV_PORT ?? "5173";
const BASE_URL = `http://localhost:${DEV_PORT}`;

const BRANCH_ORDER = ["router", "sequence", "architecture"] as const;

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

  test.beforeAll(async () => {
    clearStaleSingletons(DATA_DIR);
    context = await chromium.launchPersistentContext(DATA_DIR, { channel: "chrome", args: [] });
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

  // Dump one branch's KV in its own fresh page. Doing this per-page rather
  // than per-engine-reset avoids bleed-through: model weights stay loaded
  // across navigations (they live in OPFS + the GGUF cache), but the engine
  // + its KV cache are rebuilt from scratch each navigation. That's exactly
  // what we want — each branch gets a clean prefill of JUST its own prompt.
  async function dumpBranch(branch: string): Promise<Uint8Array> {
    console.log(`[build-cache] building branch "${branch}"...`);
    const page = await context.newPage();
    try {
      page.on("console", (msg) => {
        const t = msg.text();
        if (t.startsWith("[draw]") || t.startsWith("[engine]") || t.startsWith("[worker]")) {
          console.log(`[${branch}] ${t}`);
        }
      });
      page.on("pageerror", (err) => console.log(`[${branch}] [pageerror] ${err.message}`));

      await page.goto(`${BASE_URL}/draw.html?noCache=1&branch=${branch}`);

      const heartbeatStart = Date.now();
      const heartbeat = setInterval(() => {
        const s = Math.floor((Date.now() - heartbeatStart) / 1000);
        console.log(`[build-cache] ${branch}: waiting for status=Ready... ${s}s elapsed`);
      }, 5000);
      try {
        await page.waitForFunction(
          () => document.querySelector("#status")?.textContent === "Ready",
          {}, { timeout: 1_500_000 },
        );
      } finally {
        clearInterval(heartbeat);
      }

      const cacheB64 = await page.evaluate(async () => {
        const engine = (window as any).__engine;
        engine.restoreCache();
        const buf: ArrayBuffer = await engine.dumpSystemCache();
        let binary = "";
        const bytes = new Uint8Array(buf);
        const CHUNK = 0x8000;
        for (let i = 0; i < bytes.length; i += CHUNK) {
          binary += String.fromCharCode.apply(null, Array.from(bytes.subarray(i, i + CHUNK)));
        }
        return btoa(binary);
      });
      return Uint8Array.from(Buffer.from(cacheB64, "base64"));
    } finally {
      await page.close().catch(() => { /* ignore */ });
    }
  }

  test("prefill each branch and pack into container", async () => {
    test.setTimeout(1_800_000);

    // The cached blob also records the branch's token count (first u32 after
    // magic/version). Read it back out so the container index has it without
    // having to plumb it separately over the page.evaluate RPC.
    function readTokenCount(blob: Uint8Array): number {
      const dv = new DataView(blob.buffer, blob.byteOffset, blob.byteLength);
      const magic = dv.getUint32(0, true);
      if (magic !== 0x564b5154) throw new Error(`expected TQKV magic, got 0x${magic.toString(16)}`);
      const version = dv.getUint32(4, true);
      if (version !== 1) throw new Error(`unsupported TQKV version ${version}`);
      return dv.getUint32(8, true);
    }

    const entries: BranchEntry[] = [];
    for (const branch of BRANCH_ORDER) {
      const blob = await dumpBranch(branch);
      const tokenCount = readTokenCount(blob);
      console.log(`[build-cache] "${branch}": ${tokenCount} tokens, ${(blob.byteLength / 1e6).toFixed(1)} MB`);
      entries.push({ name: branch, blob, tokenCount });
    }

    const packed = packContainer(entries);
    writeFileSync(OUTPUT_PATH, packed);
    const total = (packed.byteLength / 1e6).toFixed(1);
    const breakdown = entries
      .map((e) => `${e.name}=${(e.blob.byteLength / 1e6).toFixed(1)}MB`)
      .join(", ");
    console.log(`[build-cache] wrote ${total} MB TQKC container → ${OUTPUT_PATH} (${breakdown})`);
  });
});
