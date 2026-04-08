import { defineConfig } from "vite";
import { resolve } from "path";

export default defineConfig(({ command }) => ({
  base: command === "serve" ? "/" : "/turboquant-wasm/",
  resolve: {
    alias: {
      "turboquant-wasm": resolve(__dirname, "../src/js/index.ts"),
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: {
      input: {
        index: resolve(__dirname, "index.html"),
        "3dgs": resolve(__dirname, "3dgs.html"),
        search: resolve(__dirname, "search.html"),
        images: resolve(__dirname, "images.html"),
      },
    },
  },
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
    },
  },
  optimizeDeps: {
    exclude: ["turboquant-wasm"],
  },
}));
