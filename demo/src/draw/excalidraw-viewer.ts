/**
 * Streaming diagram renderer.
 *
 * Two modes:
 *  - `svg` (default): renders elements via `exportToSvg` into a static SVG,
 *    diffs the DOM with morphdom so geometry changes animate smoothly, and
 *    lerps a scene-space viewport toward the bounding box of the current
 *    elements every frame. No live Excalidraw component is mounted — which
 *    means `updateScene`'s cached stroke canvases (the root cause of "arrows
 *    don't follow boxes after addGroup") cannot exist at this layer.
 *  - `live`: mounts the full Excalidraw React component on demand so the
 *    user can interactively edit the final diagram. Only entered via
 *    `enterEditMode()`; `exitEditMode()` tears it back down and returns to
 *    SVG streaming.
 *
 * Cribbed in structure from github.com/excalidraw/excalidraw-mcp-app
 * (mcp-app.tsx), adapted to keep our SDK + graphviz + TypeScript layer.
 */

import React from "react";
import { createRoot, type Root } from "react-dom/client";
import morphdom from "morphdom";
import "@excalidraw/excalidraw/index.css";

interface ViewportRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

const LERP_SPEED = 0.05;        // 0..1; higher = snappier camera
const EXPORT_PADDING = 20;      // matches exportToSvg's exportPadding arg
const FIT_PADDING = 60;         // extra scene-space padding around elements

// ── Viewer state ────────────────────────────────────────────────────────────

let container: HTMLElement | null = null;
let mode: "svg" | "live" = "svg";

// SVG mode
let svgWrapper: HTMLElement | null = null;
let currentElements: any[] = [];
let animatedVP: ViewportRect | null = null;
let targetVP: ViewportRect | null = null;
let rafId = 0;

// Live mode
let liveRoot: Root | null = null;
let excalidrawAPI: any = null;

// Lazy-loaded Excalidraw exports
let Excalidraw: any = null;
let exportToSvg: any = null;
let fontsReadyPromise: Promise<void> | null = null;

async function loadExcalidraw(): Promise<void> {
  if (Excalidraw && exportToSvg) return;
  const mod = await import("@excalidraw/excalidraw");
  Excalidraw = mod.Excalidraw;
  exportToSvg = mod.exportToSvg;
}

function ensureFontsLoaded(): Promise<void> {
  if (!fontsReadyPromise) {
    // Excalidraw's SVG export needs Virgil/Excalifont to measure text. Load
    // them once so the first render has correct bounding boxes.
    fontsReadyPromise = (async () => {
      if (typeof document !== "undefined" && (document as any).fonts) {
        try { await (document as any).fonts.load("20px Excalifont"); } catch { /* ignore */ }
        try { await (document as any).fonts.load("20px Virgil"); } catch { /* ignore */ }
      }
    })();
  }
  return fontsReadyPromise;
}

// ── Mount ───────────────────────────────────────────────────────────────────

export async function mountExcalidraw(host: HTMLElement): Promise<void> {
  container = host;
  mode = "svg";
  container.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.className = "svg-host";
  wrap.style.cssText = "width:100%;height:100%;position:relative;background:#ffffff;";
  container.appendChild(wrap);
  svgWrapper = wrap;
  // Preload the heavy Excalidraw import + fonts in the background so the
  // first updateDiagram call doesn't pay the full cost.
  loadExcalidraw().catch(() => {});
  ensureFontsLoaded().catch(() => {});
}

// ── Viewport math ───────────────────────────────────────────────────────────

/** Scene-space bounding box of every element, including arrow polyline points. */
function computeSceneBBox(elements: any[]): { minX: number; minY: number; maxX: number; maxY: number } | null {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const el of elements) {
    if (el.x == null || el.y == null) continue;
    const w = el.width ?? 0;
    const h = el.height ?? 0;
    minX = Math.min(minX, el.x);
    minY = Math.min(minY, el.y);
    maxX = Math.max(maxX, el.x + w);
    maxY = Math.max(maxY, el.y + h);
    // Arrow/line polyline points are relative to (x, y) — walk them too.
    if (el.points && Array.isArray(el.points)) {
      for (const pt of el.points) {
        minX = Math.min(minX, el.x + pt[0]);
        minY = Math.min(minY, el.y + pt[1]);
        maxX = Math.max(maxX, el.x + pt[0]);
        maxY = Math.max(maxY, el.y + pt[1]);
      }
    }
  }
  if (!isFinite(minX)) return null;
  return { minX, minY, maxX, maxY };
}

/** Compute a "fit-all" viewport for the current elements, plus padding. */
function computeFitViewport(elements: any[]): ViewportRect | null {
  const bb = computeSceneBBox(elements);
  if (!bb) return null;
  return {
    x: bb.minX - FIT_PADDING,
    y: bb.minY - FIT_PADDING,
    width: (bb.maxX - bb.minX) + FIT_PADDING * 2,
    height: (bb.maxY - bb.minY) + FIT_PADDING * 2,
  };
}

/**
 * Convert a scene-space viewport rect into the coordinate system used by the
 * SVG produced by `exportToSvg`. exportToSvg shifts every element so the top
 * left of the scene bounding box sits at (exportPadding, exportPadding) inside
 * the SVG viewBox, so the same shift applies to any viewport we want to show.
 */
function sceneToSvgViewBox(
  vp: ViewportRect,
  sceneMinX: number,
  sceneMinY: number,
): { x: number; y: number; w: number; h: number } {
  return {
    x: vp.x - sceneMinX + EXPORT_PADDING,
    y: vp.y - sceneMinY + EXPORT_PADDING,
    w: vp.width,
    h: vp.height,
  };
}

function applyViewBox(): void {
  if (!svgWrapper || !animatedVP) return;
  const svg = svgWrapper.querySelector("svg");
  if (!svg) return;
  const bb = computeSceneBBox(currentElements);
  const vb = sceneToSvgViewBox(animatedVP, bb?.minX ?? 0, bb?.minY ?? 0);
  // Force a 4:3 aspect ratio by padding whichever dimension is short. Keeps
  // the SVG element size stable and avoids squash/stretch across partials.
  const hostRatio = 4 / 3;
  const vpRatio = vb.w / vb.h;
  let outW = vb.w, outH = vb.h, outX = vb.x, outY = vb.y;
  if (vpRatio > hostRatio) {
    // Too wide: grow height
    outH = vb.w / hostRatio;
    outY = vb.y - (outH - vb.h) / 2;
  } else if (vpRatio < hostRatio) {
    // Too tall: grow width
    outW = vb.h * hostRatio;
    outX = vb.x - (outW - vb.w) / 2;
  }
  svg.setAttribute("viewBox", `${outX} ${outY} ${outW} ${outH}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
}

function animateStep(): void {
  if (!animatedVP || !targetVP) { rafId = 0; return; }
  const a = animatedVP;
  const t = targetVP;
  a.x += (t.x - a.x) * LERP_SPEED;
  a.y += (t.y - a.y) * LERP_SPEED;
  a.width += (t.width - a.width) * LERP_SPEED;
  a.height += (t.height - a.height) * LERP_SPEED;
  applyViewBox();
  const delta = Math.abs(t.x - a.x) + Math.abs(t.y - a.y)
              + Math.abs(t.width - a.width) + Math.abs(t.height - a.height);
  if (delta > 0.5) {
    rafId = requestAnimationFrame(animateStep);
  } else {
    // Snap to final target to avoid sub-pixel drift then stop.
    animatedVP = { ...t };
    applyViewBox();
    rafId = 0;
  }
}

function kickAnimation(): void {
  if (rafId) cancelAnimationFrame(rafId);
  rafId = requestAnimationFrame(animateStep);
}

// ── SVG render path ─────────────────────────────────────────────────────────

export async function updateDiagram(elements: readonly any[]): Promise<void> {
  if (mode === "live") {
    // Edit mode — pass through to the live component.
    excalidrawAPI?.updateScene({
      elements: elements as any,
      appState: { viewModeEnabled: false, viewBackgroundColor: "#ffffff" },
    });
    return;
  }
  if (!svgWrapper) {
    // mountExcalidraw wasn't called yet. Shouldn't happen in the normal
    // flow but log so we notice if it does — updateDiagram silently failing
    // is exactly the class of bug that ate three commits on arrow IDs.
    console.warn(`[viewer] updateDiagram called with no svgWrapper (mode=${mode})`);
    return;
  }
  currentElements = [...elements];

  try {
    await loadExcalidraw();
    await ensureFontsLoaded();

    const svg: SVGSVGElement = await exportToSvg({
      elements: currentElements as any,
      appState: { viewBackgroundColor: "#ffffff", exportBackground: true } as any,
      files: null,
      exportPadding: EXPORT_PADDING,
      skipInliningFonts: true,
    });

    // Re-check the wrapper after the awaits above: enterEditMode can null it
    // out between the top-of-function check and here, which produced a
    // `Cannot read properties of null (reading 'querySelector')` crash when
    // auto-edit fires right after a streaming updateDiagram.
    if (!svgWrapper || mode !== "svg") return;

    // Normalize the SVG's sizing so it always fills the host div.
    svg.style.width = "100%";
    svg.style.height = "100%";
    svg.removeAttribute("width");
    svg.removeAttribute("height");

    const existing = svgWrapper.querySelector("svg");
    if (existing) {
      // morphdom updates attrs + children in place without tearing down the
      // DOM, which means the browser can smoothly transition changed
      // elements. Passing the new svg root directly works because morphdom
      // handles any node (not just documents).
      morphdom(existing, svg, { childrenOnly: false });
    } else {
      svgWrapper.appendChild(svg);
    }

    // Compute the new camera target (fit-all + padding) and kick the lerp.
    const vp = computeFitViewport(currentElements);
    if (vp) {
      targetVP = vp;
      if (!animatedVP) {
        // First render — snap the camera immediately so the initial frame
        // isn't a weird zoom-from-origin.
        animatedVP = { ...vp };
      }
      applyViewBox();
      kickAnimation();
    }
  } catch (e) {
    // exportToSvg throws on some partial/malformed scenes — swallow so the
    // streaming flow keeps feeding us frames.
    console.warn("[viewer] exportToSvg failed:", e);
  }
}

export function fitToScreen(): void {
  if (mode === "live") {
    try { excalidrawAPI?.scrollToContent(undefined, { fitToContent: true }); } catch { /* ignore */ }
    return;
  }
  const vp = computeFitViewport(currentElements);
  if (!vp) return;
  targetVP = vp;
  if (!animatedVP) animatedVP = { ...vp };
  kickAnimation();
}

export function resetDiagram(): void {
  currentElements = [];
  animatedVP = null;
  targetVP = null;
  if (rafId) { cancelAnimationFrame(rafId); rafId = 0; }
  if (mode === "live") {
    try { excalidrawAPI?.resetScene(); } catch { /* ignore */ }
    return;
  }
  if (svgWrapper) {
    while (svgWrapper.firstChild) svgWrapper.removeChild(svgWrapper.firstChild);
  }
}

// ── Thinking-cloud overlay ──────────────────────────────────────────────────
//
// While the LLM is inside the thinking channel, we render its reasoning as
// a single Excalidraw scene: a dashed ellipse ("cloud") with a text element
// bound inside. rAF coalesces bursts of token updates so we don't thrash
// exportToSvg on every tok (~30ms each).

let thinkingRafId = 0;
let thinkingPendingText: string | null = null;
const THINK_WRAP_WIDTH = 72;      // chars per line before soft wrap
const THINK_MAX_LINES = 14;       // tail the display so the cloud stays readable

function wrapThinkingText(text: string): string {
  // Preserve the model's own newlines; only break lines that are themselves
  // longer than THINK_WRAP_WIDTH, at word boundaries.
  const out: string[] = [];
  for (const raw of text.split("\n")) {
    if (raw.length <= THINK_WRAP_WIDTH) { out.push(raw); continue; }
    let line = "";
    for (const word of raw.split(/(\s+)/)) {
      if (line.length + word.length > THINK_WRAP_WIDTH && line.trim().length > 0) {
        out.push(line);
        line = word.replace(/^\s+/, "");
      } else {
        line += word;
      }
    }
    if (line) out.push(line);
  }
  return out.slice(-THINK_MAX_LINES).join("\n");
}

function buildThinkingElements(text: string): any[] {
  const display = wrapThinkingText(text);
  const lines = display.split("\n");
  const fontSize = 16;
  const lineHeight = 1.25;
  const charWidth = 9;  // rough Excalifont 16pt average glyph width
  const padX = 40;
  const padY = 34;
  const longestLineWidth = lines.reduce((m, l) => Math.max(m, l.length * charWidth), 0);
  const textWidth = Math.max(360, Math.min(THINK_WRAP_WIDTH * charWidth, longestLineWidth));
  const textHeight = Math.max(80, Math.ceil(lines.length * fontSize * lineHeight) + 12);
  const cloudWidth = textWidth + padX * 2;
  const cloudHeight = textHeight + padY * 2;
  const now = Date.now();
  // Standalone text (no containerId) avoids Excalidraw re-computing the text
  // position based on container bounds. Use a sharp-cornered rectangle
  // (roundness: null) — an ellipse or rounded rect clipped the first/last
  // lines of multi-line reasoning text against the curved boundary.
  const cloud: any = {
    id: "thinking-cloud",
    type: "rectangle",
    x: 0, y: 0,
    width: cloudWidth, height: cloudHeight,
    angle: 0,
    strokeColor: "#8892a6",
    backgroundColor: "#f5f8ff",
    fillStyle: "solid",
    strokeWidth: 2,
    strokeStyle: "dashed",
    roughness: 2,
    opacity: 100,
    groupIds: [],
    frameId: null,
    roundness: null,
    seed: 12345,
    version: 1,
    versionNonce: 12345,
    isDeleted: false,
    boundElements: null,
    updated: now,
    link: null,
    locked: false,
    customData: { _thinking: true },
  };
  const textEl: any = {
    id: "thinking-text",
    type: "text",
    x: padX,
    y: padY,
    width: textWidth,
    height: textHeight,
    angle: 0,
    strokeColor: "#333",
    backgroundColor: "transparent",
    fillStyle: "solid",
    strokeWidth: 1,
    strokeStyle: "solid",
    roughness: 0,
    opacity: 100,
    groupIds: [],
    frameId: null,
    roundness: null,
    seed: 12346,
    version: 1,
    versionNonce: 12346,
    isDeleted: false,
    boundElements: null,
    updated: now,
    link: null,
    locked: false,
    containerId: null,
    originalText: display,
    autoResize: false,
    text: display,
    fontSize,
    fontFamily: 1,
    textAlign: "left",
    verticalAlign: "top",
    baseline: 16,
    lineHeight,
    customData: { _thinking: true },
  };
  return [cloud, textEl];
}

export function showThinkingCloud(text: string): void {
  thinkingPendingText = text;
  if (thinkingRafId !== 0) return;
  thinkingRafId = requestAnimationFrame(() => {
    thinkingRafId = 0;
    if (thinkingPendingText === null) return;
    const t = thinkingPendingText;
    thinkingPendingText = null;
    void updateDiagram(buildThinkingElements(t));
  });
}

export function clearThinkingCloud(): void {
  if (thinkingRafId !== 0) {
    cancelAnimationFrame(thinkingRafId);
    thinkingRafId = 0;
  }
  thinkingPendingText = null;
  resetDiagram();
}

// ── Edit-mode (live Excalidraw) ─────────────────────────────────────────────

export async function enterEditMode(): Promise<void> {
  if (mode === "live" || !container) return;
  // Cancel any in-flight camera animation.
  if (rafId) { cancelAnimationFrame(rafId); rafId = 0; }
  const finalElements = [...currentElements];
  container.innerHTML = "";
  svgWrapper = null;
  mode = "live";

  await loadExcalidraw();
  liveRoot = createRoot(container);
  liveRoot.render(
    React.createElement(Excalidraw, {
      initialData: {
        elements: finalElements as any,
        appState: { viewBackgroundColor: "#ffffff", zenModeEnabled: false } as any,
      },
      excalidrawAPI: (ref: any) => {
        excalidrawAPI = ref;
        if (ref) {
          setTimeout(() => {
            try { ref.scrollToContent(undefined, { fitToContent: true }); } catch { /* ignore */ }
          }, 50);
        }
      },
    }),
  );
}

export function exitEditMode(): void {
  if (mode === "svg" || !container) return;
  if (liveRoot) { liveRoot.unmount(); liveRoot = null; }
  excalidrawAPI = null;
  container.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.className = "svg-host";
  wrap.style.cssText = "width:100%;height:100%;position:relative;background:#ffffff;";
  container.appendChild(wrap);
  svgWrapper = wrap;
  mode = "svg";
  // Re-render whatever's in currentElements so the user doesn't drop back
  // to an empty canvas.
  if (currentElements.length > 0) {
    void updateDiagram(currentElements);
  }
}

export function getMode(): "svg" | "live" { return mode; }
