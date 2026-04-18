import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  testMatch: "**/*.spec.ts",
  timeout: 30000,
  use: {
    baseURL: "http://localhost:5173",
    channel: "chrome",
    launchOptions: {
      args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan"],
    },
  },
  // webServer.timeout must be larger than vite's cold-start time — on M1 with a
  // warm dep cache vite reports "ready in ~60-70s" for the draw bundle (large
  // Excalidraw + transformers.js deps), which blew past the 60s default and
  // caused playwright to abort mid-boot, killing chrome before it could release
  // the persistent-profile SingletonLock.
  webServer: {
    command: "npx vite --port 5173",
    port: 5173,
    reuseExistingServer: true,
    timeout: 180_000,
  },
});
