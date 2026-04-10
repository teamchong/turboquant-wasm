/**
 * Mount Excalidraw React component into a DOM element.
 * Keeps the rest of the page vanilla TS — React is only used here.
 */

import React from "react";
import { createRoot, type Root } from "react-dom/client";
import "@excalidraw/excalidraw/index.css";

let root: Root | null = null;
let excalidrawAPI: any = null;
let Excalidraw: any = null;

export async function mountExcalidraw(container: HTMLElement) {
  if (root) return;

  // Dynamic import — Excalidraw is large, load only when needed
  const mod = await import("@excalidraw/excalidraw");
  Excalidraw = mod.Excalidraw;

  root = createRoot(container);
  root.render(
    React.createElement(Excalidraw, {
      excalidrawAPI: (ref: any) => { excalidrawAPI = ref; },
      initialData: { appState: { viewModeEnabled: true, zenModeEnabled: true } },
      viewModeEnabled: true,
      zenModeEnabled: true,
    }),
  );
}

export function updateDiagram(elements: readonly any[]) {
  if (!excalidrawAPI) return;
  // Reset scene completely then set new elements
  excalidrawAPI.resetScene();
  excalidrawAPI.updateScene({
    elements,
    appState: { viewModeEnabled: true, zenModeEnabled: true },
  });
  setTimeout(() => {
    excalidrawAPI.scrollToContent(undefined, { fitToContent: true });
  }, 200);
}
