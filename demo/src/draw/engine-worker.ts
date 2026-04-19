/**
 * Engine Web Worker — all GPU work runs here, off the main thread.
 *
 * Protocol:
 *   Main → Worker: init, prefill, decode, stream, restoreCache, snapshotCache,
 *                  resetCache, getStats, abort, wipe, dumpCache
 *   Worker → Main: status, ready, token, stats, profile, error, prefillDone,
 *                  progress, streamDone, cacheDump
 */

import { fetchGGUF, parseGGUF, uploadTensorsToGPU, benchmarkTensorUpload, wipeModels } from "./model-loader.js";
import { InferenceEngine } from "./engine.js";
import { GrammarState, S_FREE, S_IN_THINK } from "./grammar.js";

let engine: InferenceEngine | null = null;
let aborted = false;
// Set once by `initGrammar`; `streamConstrained` uses it to advance state.
let grammarTransitions: Uint8Array | null = null;
let grammarVocabSize = 0;

function post(msg: any, transfer?: Transferable[]) {
  if (transfer) (self as any).postMessage(msg, transfer);
  else self.postMessage(msg);
}
function status(text: string) { post({ type: "status", text }); }

// Serialize every engine-state operation. onmessage is called for each posted
// message in arrival order, but the dispatched handlers are async — without a
// queue, message N+1 starts running while N is still awaiting GPU work, which
// races the shared `_stagingBuf.mapAsync` (only one outstanding map allowed)
// and lets `restoreCache` mutate state mid-prefill. The queue makes "process
// in order, one-at-a-time" the actual contract.
//
// `abort` is the only message that bypasses the queue: it's a signal that
// must take effect immediately so an in-flight stream can stop early.
let workQueue: Promise<void> = Promise.resolve();
function enqueue(fn: () => Promise<void>): Promise<void> {
  const next = workQueue.then(fn).catch((e) => {
    post({ type: "error", message: (e as Error).message });
  });
  workQueue = next;
  return next;
}

interface ExtraBranchInit {
  name: string;
  blob: ArrayBuffer;
  tokenCount: number;
  tokenIds: number[];
}

async function init(
  ggufUrl: string,
  systemTokenIds: number[],
  enableProfile: boolean,
  cacheBlob: ArrayBuffer | null,
  benchUpload = false,
  extraBranches: ExtraBranchInit[] = [],
) {
  try {
    const phases: Array<{ name: string; ms: number }> = [];
    let phaseStart = performance.now();
    const mark = (name: string) => {
      const now = performance.now();
      phases.push({ name, ms: now - phaseStart });
      phaseStart = now;
    };

    // 1. WebGPU
    const gpu = (navigator as any).gpu;
    if (!gpu) throw new Error("navigator.gpu not available in worker — check chrome://gpu");
    const adapter = await gpu.requestAdapter();
    if (!adapter) throw new Error("GPU adapter request failed — try closing other GPU tabs or check chrome://gpu");
    if (!adapter.features.has("shader-f16")) throw new Error("shader-f16 not supported");
    if (!adapter.features.has("subgroups")) throw new Error("subgroups not supported");
    const features: GPUFeatureName[] = ["shader-f16", "subgroups"];
    const hasTimestamp = adapter.features.has("timestamp-query");
    if (hasTimestamp) features.push("timestamp-query");
    const device = await adapter.requestDevice({
      requiredFeatures: features,
      requiredLimits: {
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
        maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
      },
    });
    console.log(`[worker] WebGPU device ready (shader-f16${hasTimestamp ? ", timestamp-query" : ""})`);
    mark("webgpu_device");

    // 2. Download GGUF
    status("Downloading model...");
    const blob = await fetchGGUF(ggufUrl, (loaded, total) => {
      const pct = total > 0 ? ((loaded / total) * 100).toFixed(0) : "?";
      const gb = (loaded / 1e9).toFixed(2);
      const totalGb = total > 0 ? (total / 1e9).toFixed(2) : "?";
      status(`Downloading: ${pct}% (${gb} / ${totalGb} GB)`);
    });
    console.log("[worker] GGUF on disk:", (blob.size / 1e9).toFixed(2), "GB");
    mark("gguf_download");

    // 3. Parse
    status("Parsing model...");
    const model = await parseGGUF(blob);
    console.log("[worker] Parsed:", model.metadata.architecture, model.metadata.nLayers, "layers");

    // Survey tensor name prefixes so we can see if the GGUF bundles a
    // vision tower we're paying GPU memory for. unsloth's Gemma 4 E2B
    // GGUFs are tagged "Image-Text-to-Text" and the HF repo card says
    // every quant variant bundles the ~150M-param vision encoder. We
    // never feed images through the engine (text-only draw demo), so
    // any vision tensors are pure dead weight in GPU.
    const prefixCounts = new Map<string, number>();
    const prefixBytes = new Map<string, number>();
    for (const [name, t] of model.tensors) {
      const prefix = name.split(".")[0];
      prefixCounts.set(prefix, (prefixCounts.get(prefix) ?? 0) + 1);
      prefixBytes.set(prefix, (prefixBytes.get(prefix) ?? 0) + t.size);
    }
    const prefixSummary = [...prefixCounts.entries()]
      .sort((a, b) => (prefixBytes.get(b[0]) ?? 0) - (prefixBytes.get(a[0]) ?? 0))
      .map(([p, c]) => `${p}:${c}(${((prefixBytes.get(p) ?? 0) / 1e6).toFixed(0)}MB)`)
      .join(" ");
    console.log(`[worker] tensor prefixes: ${prefixSummary}`);
    mark("gguf_parse");

    // 4. Upload tensors — filter out vision-related tensors we don't use.
    // Prefixes taken from llama.cpp's gemma3n/gemma4 convention:
    //   v.* / v_* / vision.* — vision tower weights (SigLIP variant)
    //   mmproj.* / mm_proj.* — multimodal projector (vision→text bridge)
    //   audio.* / a.* / whisper.* — audio encoder (if gemma4 ships one)
    // Filter is permissive on naming; we log what got skipped so regressions
    // are obvious.
    const skippedTensors: string[] = [];
    const textOnlyFilter = (name: string): boolean => {
      const lc = name.toLowerCase();
      const isVision = /^(v|vision|vision_tower|mmproj|mm_proj|mm_projector|siglip|clip|visual)\./.test(lc)
        || lc.startsWith("img_") || lc.startsWith("image_")
        || lc.startsWith("audio.") || lc.startsWith("whisper.");
      if (isVision) { skippedTensors.push(name); return false; }
      return true;
    };

    status("Uploading to GPU...");
    if (benchUpload) {
      await benchmarkTensorUpload(device, model);
    } else {
      await uploadTensorsToGPU(device, model, (n, total) => {
        status(`Uploading tensors: ${n}/${total}`);
      }, textOnlyFilter);
    }
    if (skippedTensors.length > 0) {
      const skippedBytes = skippedTensors.reduce((sum, n) => sum + (model.tensors.get(n)?.size ?? 0), 0);
      console.log(`[worker] skipped ${skippedTensors.length} non-text tensors (${(skippedBytes / 1e6).toFixed(0)}MB saved): ${skippedTensors.slice(0, 8).join(", ")}${skippedTensors.length > 8 ? " ..." : ""}`);
    } else {
      console.log("[worker] no vision/audio tensors detected — GGUF is already text-only");
    }
    mark("tensor_upload");

    // 5. Create engine
    device.pushErrorScope("validation");
    engine = await InferenceEngine.create(device, model);
    const createErr = await device.popErrorScope();
    if (createErr) throw new Error(`engine create validation error: ${createErr.message}`);
    // Profiler is only enabled on explicit ?profile=1 — otherwise its
    // one-pass-per-category behaviour blocks the compute-pass coalescing
    // in engine.ensureComputePass(). Without profiler we collapse ~900
    // dispatches/step into a handful of passes and recover ~9ms/step of
    // beginComputePass/end overhead.
    if (hasTimestamp && enableProfile) engine.enableProfiling();
    console.log("[worker] Engine ready");
    mark("engine_create");

    // Send tensor metadata to main thread (for tests)
    const tensorMeta: any[] = [];
    for (const [name, t] of model.tensors) {
      tensorMeta.push({ name, type: t.type, dims: t.dims });
    }
    post({ type: "tensorMeta", data: tensorMeta });

    // 6. System prompt setup — prefer a prebuilt KV cache blob if main passed
    // one, otherwise fall back to the slow prefill path.
    //
    // After this block the "active" cache is the router branch (either
    // loadCache-ed from the prebuilt blob or freshly prefilled). Any
    // extraBranches (sequence/architecture) are registered on the engine
    // via registerBranch so mountKV(name) can swap them in later without
    // reparsing any blob — they stay CPU-resident in engine.branches.
    if (cacheBlob) {
      try {
        status("Loading prebuilt system cache...");
        const t0 = performance.now();
        // Register router first, then mountKV — this path sets
        // engine.activeBranch = "router" so the snapshot we take next
        // correctly records "router" as the baseline branch. restoreCache
        // uses that to put activeBranch back in sync after a mid-stream
        // mount followed by retry rollback.
        engine.registerBranch("router", cacheBlob, systemTokenIds);
        engine.mountKV("router");
        const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
        console.log(`[worker] loadCache done: ${systemTokenIds.length} tokens in ${elapsed}s`);
        engine.snapshotCache();
      } catch (e) {
        console.warn(`[worker] loadCache failed (${(e as Error).message}), falling back to prefill`);
        cacheBlob = null;
      }
    }
    if (!cacheBlob) {
      status("Preparing system prompt...");
      console.log(`[worker] prefill start: ${systemTokenIds.length} tokens`);
      const t0 = performance.now();
      // Emit a console.log at 10% steps too. status() posts a message to main
      // which updates the DOM, but the build-cache playwright test forwards
      // `[worker]` console logs to the parent shell — without this, the
      // prefill phase runs silently for 30-60s during `bun run build` and
      // looks like the build is hung.
      let lastLoggedPct = -1;
      await engine.prefill(
        systemTokenIds,
        (done, total) => {
          const pct = ((done / total) * 100).toFixed(0);
          const elapsed = ((performance.now() - t0) / 1000).toFixed(0);
          status(`Preparing: ${pct}% (${done}/${total} tokens, ${elapsed}s)`);
          const pctInt = Math.floor((done / total) * 10) * 10;
          if (pctInt > lastLoggedPct) {
            lastLoggedPct = pctInt;
            console.log(`[worker] prefill ${pctInt}% (${done}/${total}, ${elapsed}s)`);
          }
        },
        (msg) => { status(msg); },
      );
      const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
      console.log(`[worker] prefill done: ${systemTokenIds.length} tokens in ${elapsed}s`);
      status("Caching system prompt...");

      // Dump the freshly-computed KV before snapshot so we can register
      // it as "router" (snapshotCache will then capture activeBranch =
      // "router" correctly). The dump itself is also posted back to main
      // so the dev server can write public/system-cache.bin for the next
      // reload. In production (no POST endpoint) main silently drops it.
      let dumpedRouterBlob: ArrayBuffer | null = null;
      try {
        const blob = await engine.dumpCache(systemTokenIds);
        const buf = blob.buffer.slice(blob.byteOffset, blob.byteOffset + blob.byteLength) as ArrayBuffer;
        engine.registerBranch("router", buf, systemTokenIds);
        // Set activeBranch explicitly — we got here via prefill, not
        // mountKV, so activeBranch was never assigned. Going through
        // mountKV("router") again would redundantly re-upload the same
        // bytes we just dumped; assigning directly avoids that.
        engine.mountKV("router");
        dumpedRouterBlob = buf;
      } catch (e) {
        console.warn(`[worker] cache auto-dump failed: ${(e as Error).message}`);
      }

      engine.snapshotCache();
      console.log("[worker] System prompt prefilled and cached");

      if (dumpedRouterBlob) {
        post({ type: "cacheRebuilt", data: dumpedRouterBlob }, [dumpedRouterBlob]);
      }
    }

    // Register non-active branches (sequence, architecture) on the engine
    // so mountKV(name) can swap them into the live cache later. We do this
    // AFTER the router is in place so engine.activeBranch stays "router"
    // (registerBranch alone doesn't mount; mountKV does).
    for (const b of extraBranches) {
      try {
        engine.registerBranch(b.name, b.blob, b.tokenIds);
        console.log(`[worker] registered branch "${b.name}": ${(b.blob.byteLength / 1e6).toFixed(1)}MB, ${b.tokenCount} tokens`);
      } catch (e) {
        console.warn(`[worker] failed to register branch "${b.name}": ${(e as Error).message}`);
      }
    }
    if (extraBranches.length > 0) {
      console.log(`[worker] branches available: ${engine.registeredBranches.join(", ")}`);
    }

    mark("prefill_or_load");
    const total = phases.reduce((s, p) => s + p.ms, 0);
    const summary = phases.map(p => `${p.name}=${(p.ms / 1000).toFixed(1)}s`).join(", ");
    console.log(`[worker] init phases (${(total / 1000).toFixed(1)}s total): ${summary}`);

    post({ type: "ready" });
  } catch (e) {
    post({ type: "error", message: (e as Error).message });
  }
}

async function prefill(tokenIds: number[]) {
  if (!engine) { post({ type: "error", message: "Engine not initialized" }); return; }
  try {
    const firstToken = await engine.prefill(tokenIds);
    post({ type: "prefillDone", firstToken });
  } catch (e) {
    post({ type: "error", message: (e as Error).message });
  }
}

async function decode(tokenId: number) {
  if (!engine) { post({ type: "error", message: "Engine not initialized" }); return; }
  try {
    const nextToken = await engine.generateToken(tokenId);
    const stats = engine.getStats();
    post({
      type: "token",
      id: nextToken,
      stats: {
        positions: stats.positions,
        compressedMB: stats.compressedMB,
        uncompressedMB: stats.uncompressedMB,
        ratio: stats.ratio,
      },
      profile: engine.lastProfile,
    });
  } catch (e) {
    post({ type: "error", message: (e as Error).message });
  }
}

async function decodeBatch(tokenIds: number[]) {
  if (!engine) { post({ type: "error", message: "Engine not initialized" }); return; }
  try {
    const argmaxes = await engine.decodeBatch(tokenIds);
    post({ type: "decodeBatchDone", argmaxes });
  } catch (e) {
    post({ type: "error", message: (e as Error).message });
  }
}

async function rollbackKV(targetPosition: number) {
  if (!engine) { post({ type: "error", message: "Engine not initialized" }); return; }
  try {
    engine.rollbackKV(targetPosition);
    post({ type: "rollbackKVDone", ok: true });
  } catch (e) {
    post({ type: "error", message: (e as Error).message });
  }
}

/**
 * Grammar-triggered branch swap. Replace the active system cache with
 * the named pre-baked branch, then prefill the user-turn tokens on top
 * of the fresh branch so the model has the actual request in context.
 *
 * Semantics after this call (on success):
 *   - engine.activeBranch === branch
 *   - engine.position === branch.tokenCount + userTurnTokens.length
 *   - last-token argmax (firstToken in the response) is the model's
 *     next prediction, ready to be handed to the decode loop
 *
 * On failure the caller should fall back to a full-restart retry — the
 * engine may be left in a partial state (cache swapped, prefill failed).
 */
async function mountBranchAndPrefill(branch: string, userTurnTokens: number[]) {
  if (!engine) { post({ type: "error", message: "Engine not initialized" }); return; }
  try {
    const t0 = performance.now();
    engine.mountKV(branch);
    const firstToken = await engine.prefill(userTurnTokens);
    const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
    console.log(`[worker] mountBranchAndPrefill "${branch}" + ${userTurnTokens.length} user tokens in ${elapsed}s, firstToken=${firstToken}`);
    post({ type: "mountBranchDone", firstToken, branch });
  } catch (e) {
    post({ type: "error", message: (e as Error).message });
  }
}

async function stream(tokenId: number, maxTokens: number, eosIds: number[]) {
  if (!engine) { post({ type: "error", message: "Engine not initialized" }); return; }
  try {
    const eosSet = new Set<number>(eosIds);
    aborted = false;
    await engine.streamTokens(tokenId, maxTokens, eosSet, (tok, stats) => {
      if (aborted) return true; // cancel the rest of the pipeline
      post({
        type: "token",
        id: tok,
        stats: {
          positions: stats.positions,
          compressedMB: stats.compressedMB,
          uncompressedMB: stats.uncompressedMB,
          ratio: stats.ratio,
        },
        profile: null,
      });
      return false;
    });
    post({ type: "streamDone" });
  } catch (e) {
    post({ type: "error", message: (e as Error).message });
  }
}

function initGrammar(masks: Uint32Array, transitions: Uint8Array, vocabSize: number, maskCount: number) {
  if (!engine) { post({ type: "error", message: "Engine not initialized" }); return; }
  try {
    engine.uploadMaskBitmaps(masks, maskCount);
    grammarTransitions = transitions;
    grammarVocabSize = vocabSize;
    post({ type: "grammarReady" });
  } catch (e) {
    post({ type: "error", message: (e as Error).message });
  }
}

// K=1 constrained stream: we need the emitted token id BEFORE submitting
// the next encoder, so we can look up the next state and pick the right
// mask bitmap. Falls back to plain `stream` behaviour if grammar isn't
// initialised.
//
// Two-phase thinking-mode flow: when startState == S_IN_THINK, the stream
// stops as soon as a token transitions grammar back out of S_IN_THINK
// (i.e. `<channel|>` just landed). The caller then prefills a reminder
// and calls streamConstrained again with startState = S_FREE to produce
// the code.
async function streamConstrained(initialTokenId: number, maxTokens: number, eosIds: number[], startState: number = S_FREE) {
  if (!engine) { post({ type: "error", message: "Engine not initialized" }); return; }
  if (!grammarTransitions) { post({ type: "error", message: "Grammar not initialized" }); return; }
  try {
    const eosSet = new Set<number>(eosIds);
    const grammar = new GrammarState(grammarTransitions, grammarVocabSize);
    grammar.state = startState;
    aborted = false;

    let nextToken = initialTokenId;
    const stateBeforeInitial = grammar.state;
    grammar.advance(nextToken);

    let reason: "eos" | "maxTokens" | "aborted" | "thinkingEnded" = "maxTokens";
    let lastTokenId = initialTokenId;

    // Edge case: the prefill-returned first token itself already contains
    // `<channel|>`. grammar flipped out of S_IN_THINK on the initial advance
    // (above) and the loop's transition check would never fire. Bail now
    // so main can run the reminder-injection handoff.
    if (stateBeforeInitial === S_IN_THINK && grammar.state !== S_IN_THINK) {
      engine.disableLogitMask();
      post({ type: "streamDone", reason: "thinkingEnded", lastTokenId });
      return;
    }

    for (let i = 0; i < maxTokens; i++) {
      if (aborted) { reason = "aborted"; break; }
      // S_IN_THINK's mask allows every token (built by grammar.ts: every
      // id passes simulateToken's fast-path return). Dispatching the mask
      // kernel in that state is a full 262k-lane sweep that performs zero
      // semantic work — skip it entirely. The mask re-enables the moment
      // grammar transitions to a code-syntax state.
      if (grammar.state === S_IN_THINK) engine.disableLogitMask();
      else engine.setLogitMaskIndex(grammar.state);
      const tok = await engine.generateToken(nextToken);
      lastTokenId = tok;
      const stats = engine.getStats();
      post({
        type: "token",
        id: tok,
        stats: {
          positions: stats.positions,
          compressedMB: stats.compressedMB,
          uncompressedMB: stats.uncompressedMB,
          ratio: stats.ratio,
        },
        profile: null,
      });
      if (eosSet.has(tok)) { reason = "eos"; break; }
      const prevState = grammar.state;
      grammar.advance(tok);
      nextToken = tok;
      // S_IN_THINK → anything-else means `<channel|>` just fired. Hand
      // control back to main so it can inject the reminder prompt before
      // the code phase starts.
      if (prevState === S_IN_THINK && grammar.state !== S_IN_THINK) {
        reason = "thinkingEnded";
        break;
      }
    }
    engine.disableLogitMask();
    post({ type: "streamDone", reason, lastTokenId });
  } catch (e) {
    engine?.disableLogitMask();
    post({ type: "error", message: (e as Error).message });
  }
}

async function dumpCache(systemTokenIds: number[]) {
  if (!engine) { post({ type: "error", message: "Engine not initialized" }); return; }
  try {
    // Cache dump assumes the engine is in the state produced by system prompt
    // prefill — restore the snapshot so repeated dumps always see the same
    // positions regardless of any decode work that may have happened since.
    engine.restoreCache();
    const blob = await engine.dumpCache(systemTokenIds);
    const buf = blob.buffer.slice(blob.byteOffset, blob.byteOffset + blob.byteLength);
    post({ type: "cacheDump", data: buf }, [buf]);
  } catch (e) {
    post({ type: "error", message: (e as Error).message });
  }
}

async function wipe() {
  try {
    engine?.resetCache();
    engine = null;
    await wipeModels();
    post({ type: "status", text: "All data wiped." });
    // Explicit ack so main can terminate the worker ONLY after the async
    // OPFS deletion has actually run. Without this, main was racing
    // postMessage({wipe}) against an immediate terminate() and the worker
    // got killed before wipeModels() touched disk — the model stayed on
    // OPFS and reloads reused the stale download.
    post({ type: "wipeDone" });
  } catch (e) {
    post({ type: "error", message: (e as Error).message });
  }
}

async function conformanceBatchedMatmul(weightName: string, batchSize: number) {
  if (!engine) { post({ type: "error", message: "Engine not initialized" }); return; }
  try {
    const result = await engine.conformanceBatchedMatmul(weightName, batchSize);
    post({ type: "batchedMatmulResult", data: result });
  } catch (e) {
    post({ type: "error", message: (e as Error).message });
  }
}

async function benchmarkBatchedMatmul(weightName: string, batchSize: number, iters: number) {
  if (!engine) { post({ type: "error", message: "Engine not initialized" }); return; }
  try {
    const result = await engine.benchmarkBatchedMatmul(weightName, batchSize, iters);
    post({ type: "benchmarkBatchedMatmulResult", data: result });
  } catch (e) {
    post({ type: "error", message: (e as Error).message });
  }
}

async function decodeBatchGenuine(tokenIds: number[]) {
  if (!engine) { post({ type: "error", message: "Engine not initialized" }); return; }
  try {
    const result = await engine.decodeBatchGenuine(tokenIds);
    post({ type: "decodeBatchGenuineResult", data: result });
  } catch (e) {
    post({ type: "error", message: (e as Error).message });
  }
}

async function conformanceForward(tokenIds: number[]) {
  if (!engine) { post({ type: "error", message: "Engine not initialized" }); return; }
  try {
    engine.resetCache();
    engine.device.pushErrorScope("validation");
    engine.device.pushErrorScope("out-of-memory");
    // Warm up cache with all but the last token — no debug dumps needed for those.
    for (let i = 0; i < tokenIds.length - 1; i++) {
      await engine.generateToken(tokenIds[i]);
    }
    // Enable dumps only for the final position so we capture its forward pass.
    engine.debugDumps = new Map();
    await engine.generateToken(tokenIds[tokenIds.length - 1]);
    const oom = await engine.device.popErrorScope();
    const val = await engine.device.popErrorScope();
    if (oom) console.error("[worker] OOM error:", oom.message);
    if (val) console.error("[worker] validation error:", val.message);
    const data: Record<string, { first3: number[]; sum: number; full: Float32Array }> = {};
    for (const [k, v] of engine.debugDumps) data[k] = v;
    engine.debugDumps = null;
    post({ type: "debugDumps", data });
  } catch (e) {
    post({ type: "error", message: (e as Error).message });
  }
}

self.onmessage = (e: MessageEvent) => {
  const msg = e.data;
  switch (msg.type) {
    case "init": enqueue(() => init(msg.ggufUrl, msg.systemTokenIds, !!msg.enableProfile, msg.cacheBlob ?? null, !!msg.benchUpload, Array.isArray(msg.extraBranches) ? msg.extraBranches : [])); break;
    case "mountBranchAndPrefill": enqueue(() => mountBranchAndPrefill(msg.branch, msg.userTurnTokens ?? [])); break;
    case "dumpCache": enqueue(() => dumpCache(msg.systemTokenIds)); break;
    case "prefill": enqueue(() => prefill(msg.tokenIds)); break;
    case "decode": enqueue(() => decode(msg.tokenId)); break;
    case "decodeBatch": enqueue(() => decodeBatch(msg.tokenIds)); break;
    case "rollbackKV": enqueue(() => rollbackKV(msg.targetPosition)); break;
    case "stream": enqueue(() => stream(msg.tokenId, msg.maxTokens, msg.eosIds)); break;
    case "initGrammar": enqueue(async () => initGrammar(new Uint32Array(msg.masks), new Uint8Array(msg.transitions), msg.vocabSize, msg.maskCount)); break;
    case "streamConstrained": enqueue(() => streamConstrained(msg.tokenId, msg.maxTokens, msg.eosIds, msg.startInThinking ? S_IN_THINK : S_FREE)); break;
    case "restoreCache": enqueue(async () => { engine?.restoreCache(); }); break;
    case "snapshotCache": enqueue(async () => { engine?.snapshotCache(); }); break;
    case "resetCache": enqueue(async () => { engine?.resetCache(); post({ type: "stats", data: engine?.getStats() }); }); break;
    case "getStats": enqueue(async () => { post({ type: "stats", data: engine?.getStats() }); }); break;
    // abort bypasses the queue on purpose: it's a cancel signal that the
    // in-flight stream loop reads on its next iteration. Queueing it would
    // make abort wait for the very stream it's trying to cancel.
    case "abort": aborted = true; break;
    case "wipe": enqueue(() => wipe()); break;
    case "conformanceForward": enqueue(() => conformanceForward(Array.isArray(msg.tokenIds) ? msg.tokenIds : [msg.tokenId])); break;
    case "conformanceBatchedMatmul": enqueue(() => conformanceBatchedMatmul(msg.weightName, msg.batchSize)); break;
    case "benchmarkBatchedMatmul": enqueue(() => benchmarkBatchedMatmul(msg.weightName, msg.batchSize, msg.iters)); break;
    case "decodeBatchGenuine": enqueue(() => decodeBatchGenuine(msg.tokenIds)); break;
  }
};
