import { defineConfig } from "vite";
import { resolve } from "path";
import { existsSync, readFileSync } from "fs";

// `publicDir` is hijacked by the top-level `dist/` for gguf-parser.wasm, so the
// prebuilt system-prompt KV cache needs its own plugin: it lives at
// `demo/public/system-cache.bin` (committed, stays out of the npm package),
// this plugin serves it in dev and emits it as an asset in the production build.
const systemCacheAsset = (root: string) => ({
  name: "system-cache-asset",
  // No auto-regen anywhere — not in `vite build` (would require WebGPU on
  // CI runners) and not in `vite dev` either (would recurse when
  // `bun run rebuild-cache` starts its own vite via playwright's
  // webServer — nested regen kills the outer test's chrome via
  // SingletonLock cleanup). The committed public/system-cache.bin is
  // the source of truth. If prompts change, run `bun run rebuild-cache`
  // explicitly on a local dev machine with WebGPU.
  configureServer(server: any) {
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
  // Use default `demo/public/` — everything vite needs to ship to the
  // browser lives here (gguf-parser.wasm, system-cache.bin, etc). This
  // used to point at `../dist/` back when the root npm-package build's
  // turboquant.wasm was imported via a vite alias, but that alias was
  // removed — demo now consumes the published npm package from
  // node_modules, so there's no reason to peek at the root dist.
  optimizeDeps: {
    exclude: ["turboquant-wasm"],
  },
}));
