// Compute the hash that vite.config.ts's systemCacheAsset plugin uses to
// decide whether public/system-cache.bin is stale. Call this after a
// successful rebuild to record the matching hash; call it without a
// rebuild if the cache content is already current but the hash drifted.
//
// Usage: node scripts/write-prompt-hash.mjs

import { createHash } from "crypto";
import { readFileSync, writeFileSync, existsSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "..");

// Must stay in sync with promptSourceHash() in demo/vite.config.ts.
// Any file here that influences a branch's token stream invalidates
// the cache when edited.
const srcs = [
  resolve(ROOT, "src/draw/main.ts"),
  resolve(ROOT, "src/draw/drawmode/sdk-types.ts"),
  resolve(ROOT, "src/draw/prompts/preamble.ts"),
  resolve(ROOT, "src/draw/prompts/router.ts"),
  resolve(ROOT, "src/draw/prompts/sequence.ts"),
  resolve(ROOT, "src/draw/prompts/architecture.ts"),
  resolve(ROOT, "src/draw/prompts/flowchart.ts"),
  resolve(ROOT, "src/draw/prompts/state.ts"),
  resolve(ROOT, "src/draw/prompts/orgchart.ts"),
  resolve(ROOT, "src/draw/prompts/er.ts"),
  resolve(ROOT, "src/draw/prompts/class.ts"),
  resolve(ROOT, "src/draw/prompts/swimlane.ts"),
];
const h = createHash("sha256");
for (const p of srcs) {
  if (existsSync(p)) h.update(readFileSync(p));
}
const hash = h.digest("hex").slice(0, 16);
const hashPath = resolve(ROOT, "public/.system-cache.prompt-hash");
writeFileSync(hashPath, hash);
console.log(`wrote ${hash} → ${hashPath}`);
