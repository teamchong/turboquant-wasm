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
 * Execute LLM-generated diagram code. The LLM writes ONLY the body —
 * no `const d = new Diagram(...)`, no `d.` prefix, no `return d.render()`.
 * All SDK methods (addBox, connect, addGroup, etc.) are injected as globals.
 * The executor creates the Diagram internally, runs the code, and renders.
 *
 * Best-effort: if the body throws, nodes added before the throw are still
 * rendered (so streaming partials animate and the final view shows what
 * compiled successfully). The error is returned alongside the result.
 *
 * Backward compat: if the LLM includes `const d = new Diagram(...)` or
 * `d.methodName(...)` or `return d.render()`, those are stripped/rewritten
 * before execution so old-style output still works.
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
    class ConfiguredDiagram extends Diagram {
      override async render(opts?: RenderOpts): Promise<RenderResult> {
        const merged = { ...renderOpts, ...opts };
        if (formatMap && typeof merged.format === "string" && merged.format in formatMap) {
          merged.format = formatMap[merged.format] as RenderOpts["format"];
        }
        return super.render(merged);
      }
    }

    // Strip legacy boilerplate if the LLM includes it.
    let cleaned = code;
    // Remove `const d = new Diagram(...);\n` — extract type/direction if present.
    const ctorMatch = cleaned.match(/const\s+d\s*=\s*new\s+Diagram\s*\(\s*(\{[^}]*\})?\s*\)\s*;?/);
    let ctorOpts: Record<string, string> = {};
    if (ctorMatch) {
      cleaned = cleaned.replace(ctorMatch[0], "").trimStart();
      if (ctorMatch[1]) {
        try { ctorOpts = JSON.parse(ctorMatch[1].replace(/'/g, '"').replace(/(\w+)\s*:/g, '"$1":')); } catch {}
      }
    }
    // Rewrite `d.methodName(` → `methodName(` for all SDK methods.
    cleaned = cleaned.replace(/\bd\.(addBox|addEllipse|addDiamond|addTable|addClass|addText|addLine|addGroup|addFrame|connect|addLane|addActor|message|setDirection|setType|setTheme|render)\s*\(/g, "$1(");
    // Remove trailing `return d.render(...)` or `return render(...)`.
    cleaned = cleaned.replace(/\breturn\s+(d\.)?render\s*\([^)]*\)\s*;?\s*$/m, "");

    const d = new ConfiguredDiagram(ctorOpts as any);

    // Build globals object — bind all public SDK methods to `d`.
    const globals: Record<string, Function> = {
      addBox: d.addBox.bind(d),
      addEllipse: d.addEllipse.bind(d),
      addDiamond: d.addDiamond.bind(d),
      addTable: d.addTable.bind(d),
      addClass: d.addClass.bind(d),
      addText: d.addText.bind(d),
      addLine: d.addLine.bind(d),
      addGroup: d.addGroup.bind(d),
      addFrame: d.addFrame.bind(d),
      connect: d.connect.bind(d),
      addLane: d.addLane.bind(d),
      addActor: d.addActor.bind(d),
      message: d.message.bind(d),
      setDirection: d.setDirection.bind(d),
      setType: d.setType.bind(d),
      setTheme: d.setTheme.bind(d),
    };

    const paramNames = Object.keys(globals);
    const paramValues = Object.values(globals);

    const wrappedCode = `
      return (async () => {
        ${cleaned}
      })();
    `;

    // Shadow dangerous globals + inject SDK method globals.
    const fn = new Function(
      ...paramNames,
      "Diagram",
      "fetch", "globalThis", "self", "process", "require",
      "eval", "Function",
      wrappedCode,
    );

    const TIMEOUT_MS = 60_000;
    let timer: ReturnType<typeof setTimeout>;
    // Best-effort: if the body throws (e.g. connect(null)), the nodes added
    // BEFORE the throw still live in `d`. Render them anyway so streaming
    // partials animate and the final view shows what compiled successfully.
    let codeError: string | undefined;
    try {
      await Promise.race([
        fn(...paramValues, ConfiguredDiagram, undefined, undefined, undefined, undefined, undefined, undefined, undefined),
        new Promise((_, reject) => {
          timer = setTimeout(() => reject(new Error(`Execution timed out after ${TIMEOUT_MS / 1000}s`)), TIMEOUT_MS);
        }),
      ]);
    } catch (e) {
      codeError = e instanceof Error ? e.message : String(e);
    } finally {
      clearTimeout(timer!);
    }
    try {
      const result = await d.render(renderOpts);
      return codeError ? { result, error: codeError } : { result };
    } catch (e) {
      const renderErr = e instanceof Error ? e.message : String(e);
      return { result: { json: EMPTY_FILE }, error: codeError ?? renderErr };
    }
  } catch (e: unknown) {
    const message = e instanceof Error ? e.message : String(e);
    return {
      result: { json: EMPTY_FILE },
      error: message,
    };
  }
}
