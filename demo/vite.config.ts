import { defineConfig } from "vite";
import { resolve } from "path";
import { existsSync, readFileSync, writeFileSync, statSync, readdirSync, rmSync } from "fs";
import { createHash } from "crypto";
import { spawn, execSync } from "child_process";

// Hash the files that determine the system-prompt token stream. If any of
// these change, the baked system-cache.bin is stale and must be rebuilt.
function promptSourceHash(root: string): string {
  const srcs = [
    resolve(root, "src/draw/main.ts"),
    resolve(root, "src/draw/drawmode/sdk-types.ts"),
  ];
  const h = createHash("sha256");
  for (const p of srcs) {
    if (existsSync(p)) h.update(readFileSync(p));
  }
  return h.digest("hex").slice(0, 16);
}

// Track the last regen attempt so we don't re-spawn playwright on every HMR.
let regenInFlight: Promise<void> | null = null;

/**
 * Nuke any Chrome process still holding the persistent-profile data dir and
 * drop stale `Singleton*` lock files. Runs both BEFORE spawning playwright
 * (so a prior crashed run can't block us with "profile already in use") and
 * AFTER playwright exits (so chrome processes orphaned by a killed test
 * don't keep the ~2 GB model in RSS forever).
 *
 * The `pkill -f <dataDir>` filter is specific enough that it only matches
 * chromes launched with our data dir — it won't touch the user's regular
 * browser. Silent on Windows and when nothing matches.
 */
function cleanupPlaywrightChromes(root: string): void {
  const dataDir = resolve(root, ".playwright-data");
  try {
    execSync(`pkill -9 -f "${dataDir}" 2>/dev/null || true`, { stdio: "ignore" });
  } catch { /* Windows or no pkill — skip */ }
  if (existsSync(dataDir)) {
    for (const entry of readdirSync(dataDir)) {
      if (entry.startsWith("Singleton")) {
        try { rmSync(resolve(dataDir, entry), { force: true }); } catch { /* ignore */ }
      }
    }
  }
}

async function regenerateCache(root: string, port: number): Promise<void> {
  return new Promise<void>((resolveFn) => {
    console.log(`[vite] system prompt changed → regenerating public/system-cache.bin via playwright (~30-60s)`);
    const cachePath = resolve(root, "public", "system-cache.bin");
    const mtimeBefore = existsSync(cachePath) ? statSync(cachePath).mtimeMs : -1;
    // Record the cache file's mtime before: if the file is rewritten during
    // the playwright run, treat the run as successful even if teardown times
    // out. The test's afterAll has a close-context hang that produces exit
    // code 1 despite the cache being correctly written in the main body.

    // Proactive cleanup before launch — prior run may have been killed with a
    // chrome still running and holding the SingletonLock.
    cleanupPlaywrightChromes(root);

    // VITE_SKIP_REGEN breaks the recursion: playwright's webServer config
    // spawns ANOTHER `vite --port 5173` (dev), which fires its own
    // configureServer listening hook, which would call checkAndRegen and
    // spawn yet another playwright — each level leaks a chrome. The child
    // vite inherits this env and short-circuits at the top of checkAndRegen.
    const child = spawn("npx", ["playwright", "test", "tests/build-cache.spec.ts"], {
      cwd: root,
      stdio: "inherit",
      env: { ...process.env, PLAYWRIGHT_DEV_PORT: String(port), VITE_SKIP_REGEN: "1" },
    });
    child.on("exit", (code) => {
      // Reactive cleanup after exit — if the test or webServer timed out,
      // playwright may have leaked a chrome (holding 2-3 GB of weights).
      // Always sweep, even on success — harmless when there's nothing to kill.
      cleanupPlaywrightChromes(root);

      const mtimeAfter = existsSync(cachePath) ? statSync(cachePath).mtimeMs : -1;
      const cacheWasWritten = mtimeAfter > 0 && mtimeAfter > mtimeBefore;
      if (code === 0 || cacheWasWritten) {
        const hashPath = resolve(root, "public", ".system-cache.prompt-hash");
        writeFileSync(hashPath, promptSourceHash(root));
        if (code === 0) {
          console.log("[vite] cache regenerated, prompt-hash updated");
        } else {
          console.log(`[vite] cache regenerated (playwright exit ${code} from teardown, ignored — file was written)`);
        }
      } else {
        console.warn(`[vite] cache regen failed (exit ${code}, cache not written). Fallback: first page load will prefill at runtime.`);
      }
      resolveFn();
    });
    child.on("error", (e) => {
      cleanupPlaywrightChromes(root);
      console.warn(`[vite] could not spawn playwright: ${e.message}`);
      resolveFn();
    });
  });
}

function checkAndRegen(root: string, port: number): void {
  // Break regen recursion. When a regen is in flight in the *parent* process,
  // it spawns playwright, playwright starts this vite via webServer, and this
  // vite's listening hook would otherwise call checkAndRegen again. Each
  // recursion leaks a chrome (the nested playwright tries to launch its own
  // persistent context, fails on SingletonLock, and the orphaned process
  // sits on 2-3 GB of weights). VITE_SKIP_REGEN is set by regenerateCache
  // when spawning playwright; child vites inherit it.
  if (process.env.VITE_SKIP_REGEN === "1") return;

  const hashPath = resolve(root, "public", ".system-cache.prompt-hash");
  const cachePath = resolve(root, "public", "system-cache.bin");
  const current = promptSourceHash(root);
  const stored = existsSync(hashPath) ? readFileSync(hashPath, "utf8").trim() : null;
  const cacheExists = existsSync(cachePath);

  // Fresh clone: hash file is not committed but cache is. Trust the committed
  // cache and seed the hash. Subsequent edits will flip stored ≠ current and
  // trigger regen. (Worker still runs the token-hash check at load time, so a
  // genuinely stale committed cache would still be caught at runtime.)
  if (!stored && cacheExists) {
    writeFileSync(hashPath, current);
    return;
  }

  if (cacheExists && stored === current) return;
  if (regenInFlight) return;
  regenInFlight = regenerateCache(root, port).finally(() => { regenInFlight = null; });
}

// `publicDir` is hijacked by the top-level `dist/` for gguf-parser.wasm, so the
// prebuilt system-prompt KV cache needs its own plugin: it lives at
// `demo/public/system-cache.bin` (committed, stays out of the npm package) and
// this plugin serves it in dev + emits it as an asset in the production build.
//
// Dev-only: also accepts POST /api/write-system-cache. The draw demo hits it
// after the cached hash doesn't match the freshly-tokenised SYSTEM_PROMPT —
// the worker prefills from scratch, dumps the new KV, and hands it here to
// overwrite public/system-cache.bin. Next reload is fast. No playwright
// command, no manual step.
const systemCacheAsset = (root: string) => ({
  name: "system-cache-asset",
  // `vite build` path: if the prompt source changed since the last cache
  // build, regen before the build completes so `dist/` ships a fresh cache.
  // Playwright's webServer config auto-starts vite dev for us, so this hook
  // just spawns the test and waits for it to finish.
  async buildStart() {
    // Same recursion guard as checkAndRegen — if this vite was itself spawned
    // by a parent regen, skip.
    if (process.env.VITE_SKIP_REGEN === "1") return;
    const hashPath = resolve(root, "public", ".system-cache.prompt-hash");
    const cachePath = resolve(root, "public", "system-cache.bin");
    const current = promptSourceHash(root);
    const stored = existsSync(hashPath) ? readFileSync(hashPath, "utf8").trim() : null;
    const cacheExists = existsSync(cachePath);
    if (!stored && cacheExists) {
      writeFileSync(hashPath, current);
      return;
    }
    if (cacheExists && stored === current) return;
    console.log("[vite build] system prompt changed → regenerating cache before build");
    await regenerateCache(root, 5173);
  },
  configureServer(server: any) {
    // On dev server startup, detect whether SYSTEM_PROMPT (or SDK_TYPES, which
    // is appended into it) has changed since the last cache build. If so,
    // spawn playwright to regenerate the cache. User never sees slow prefill
    // in browser.
    server.httpServer?.once("listening", () => {
      const addr = server.httpServer.address();
      const port = typeof addr === "object" && addr ? addr.port : 5173;
      checkAndRegen(root, port);
    });
    server.middlewares.use((req: any, res: any, next: any) => {
      if (req.url === "/system-cache.bin" || req.url === "/turboquant-wasm/system-cache.bin") {
        const filepath = resolve(root, "public", "system-cache.bin");
        if (existsSync(filepath)) {
          const data = readFileSync(filepath);
          res.setHeader("Content-Type", "application/octet-stream");
          res.setHeader("Content-Length", String(data.length));
          res.end(data);
          return;
        }
      }
      if (req.url === "/api/write-system-cache" && req.method === "POST") {
        const chunks: Buffer[] = [];
        req.on("data", (c: Buffer) => chunks.push(c));
        req.on("end", () => {
          try {
            const body = Buffer.concat(chunks);
            // First 4 bytes must be "TQKV" magic — reject anything else so
            // a misrouted request can't corrupt the cache file.
            if (body.length < 4 || body.readUInt32LE(0) !== 0x564B5154) {
              res.statusCode = 400;
              res.end("bad magic (expected TQKV)");
              return;
            }
            const filepath = resolve(root, "public", "system-cache.bin");
            writeFileSync(filepath, body);
            console.log(`[vite] wrote ${filepath} (${(body.length / 1e6).toFixed(1)} MB)`);
            res.statusCode = 200;
            res.end("ok");
          } catch (e) {
            res.statusCode = 500;
            res.end((e as Error).message);
          }
        });
        req.on("error", (e: Error) => {
          res.statusCode = 500;
          res.end(e.message);
        });
        return;
      }
      next();
    });
  },
  generateBundle(this: any) {
    const filepath = resolve(root, "public", "system-cache.bin");
    if (existsSync(filepath)) {
      this.emitFile({
        type: "asset",
        fileName: "system-cache.bin",
        source: readFileSync(filepath),
      });
    }
  },
});

export default defineConfig(({ command }) => ({
  base: command === "serve" ? "/" : "/turboquant-wasm/",
  assetsInclude: ["**/*.wgsl"],
  plugins: [systemCacheAsset(__dirname)],
  build: {
    outDir: "dist",
    emptyOutDir: true,
    target: "esnext",
    rollupOptions: {
      input: {
        index: resolve(__dirname, "index.html"),
        "3dgs": resolve(__dirname, "3dgs.html"),
        search: resolve(__dirname, "search.html"),
        images: resolve(__dirname, "images.html"),
        draw: resolve(__dirname, "draw.html"),
        "gpu-test": resolve(__dirname, "gpu-test.html"),
        "gpu-debug": resolve(__dirname, "gpu-debug.html"),
      },
    },
  },
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
    fs: {
      allow: [resolve(__dirname, "..")],
    },
  },
  publicDir: resolve(__dirname, "../dist"),
  optimizeDeps: {
    exclude: ["turboquant-wasm"],
  },
}));
