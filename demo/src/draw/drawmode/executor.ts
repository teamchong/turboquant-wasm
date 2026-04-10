/**
 * LocalExecutor — runs LLM-generated diagram code in the current process.
 *
 * Defense-in-depth: common globals (fetch, eval, Function, etc.) are shadowed
 * so LLM code can only use the Diagram API in the normal case. This is NOT a
 * security sandbox — constructor chain escapes and dynamic import() bypass
 * shadowing. For true isolation, use Cloudflare's Dynamic Worker Loader.
 *
 * Acceptable because: locally the LLM already has full system access, and on
 * Workers the env only contains non-secret bindings (WASM module, browser).
 */

import { Diagram } from "./sdk.js";
import { EXCALIDRAW_VERSION } from "./types.js";
import type { RenderResult, RenderOpts, ExcalidrawFile } from "./types.js";

const EMPTY_FILE: ExcalidrawFile = { type: "excalidraw", version: EXCALIDRAW_VERSION, elements: [] };

export interface ExecuteResult {
  result: RenderResult;
  error?: string;
}

/**
 * Execute LLM-generated TypeScript code that uses the Diagram SDK.
 * The code receives `Diagram` as a global and must return a `RenderResult`
 * (i.e., call `d.render()`).
 *
 * `formatMap` coerces output formats — e.g. `{ excalidraw: "url" }` forces
 * excalidraw format to url (useful when there's no filesystem).
 */
export async function executeCode(
  code: string,
  renderOpts?: RenderOpts,
  formatMap?: Partial<Record<string, string>>,
): Promise<ExecuteResult> {
  try {
    // Create a per-execution Diagram subclass that merges renderOpts as defaults.
    // This avoids mutating Diagram.prototype which would stack across concurrent requests.
    class ConfiguredDiagram extends Diagram {
      override async render(opts?: RenderOpts): Promise<RenderResult> {
        const merged = { ...renderOpts, ...opts };
        if (formatMap && typeof merged.format === "string" && merged.format in formatMap) {
          merged.format = formatMap[merged.format] as RenderOpts["format"];
        }
        return super.render(merged);
      }
    }

    const wrappedCode = `
      return (async () => {
        ${code}
      })();
    `;

    // Shadow common globals as defense-in-depth. Not bulletproof — constructor
    // chain escapes (Diagram.constructor.constructor) and dynamic import() still
    // work. But blocks the obvious vectors (fetch, eval, process, etc.).
    const fn = new Function(
      "Diagram",
      "fetch", "globalThis", "self", "process", "require",
      "eval", "Function",
      wrappedCode,
    );

    // 60s timeout — prevents infinite loops / stuck awaits from hanging forever
    const TIMEOUT_MS = 60_000;
    let timer: ReturnType<typeof setTimeout>;
    try {
      const result = await Promise.race([
        fn(ConfiguredDiagram, undefined, undefined, undefined, undefined, undefined, undefined, undefined),
        new Promise((_, reject) => {
          timer = setTimeout(() => reject(new Error(`Execution timed out after ${TIMEOUT_MS / 1000}s`)), TIMEOUT_MS);
        }),
      ]);

      if (!result || typeof result !== "object") {
        return {
          result: { json: EMPTY_FILE },
          error: "Code did not return a RenderResult. Make sure to return d.render().",
        };
      }

      return { result };
    } finally {
      clearTimeout(timer!);
    }
  } catch (e: unknown) {
    const message = e instanceof Error ? e.message : String(e);
    return {
      result: { json: EMPTY_FILE },
      error: message,
    };
  }
}
