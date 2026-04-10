/**
 * Layout bridge — Zig WASM with statically-linked Graphviz C for graph layout,
 * plus validation. Graphviz `dot` engine handles layout and edge routing.
 */

import { z } from "zod";

const encoder = new TextEncoder();
const decoder = new TextDecoder();

class WasiExit extends Error {
  code: number;
  constructor(code: number) { super(`WASI exit: ${code}`); this.code = code; }
}

let _dirname: string | undefined;
async function getDirname(): Promise<string> {
  if (!_dirname) {
    const { dirname } = await import("node:path");
    const { fileURLToPath } = await import("node:url");
    _dirname = dirname(fileURLToPath(import.meta.url));
  }
  return _dirname;
}

// ── WASM Layout Result Types ──

export interface EdgeRoute {
  /** Absolute Excalidraw-coordinate points for the arrow path */
  points: [number, number][];
  /** Label position in Excalidraw coordinates */
  labelPos?: { x: number; y: number };
  /** Where arrow meets source node edge (0-1 normalized), computed by Graphviz */
  startFixedPoint?: [number, number];
  /** Where arrow meets target node edge (0-1 normalized), computed by Graphviz */
  endFixedPoint?: [number, number];
}

export interface GroupBounds {
  id: string;
  x: number; y: number;
  width: number; height: number;
}

interface WasmLayoutResult {
  nodes: { id: string; x: number; y: number }[];
  edgeRoutes: Map<string, EdgeRoute>;
  groupBounds?: GroupBounds[];
}

// ── WASM Instance ──

interface WasmLayoutExports {
  memory: WebAssembly.Memory;
  alloc: (size: number) => number;
  resetHeap: () => void;
  layoutGraph: (nodesPtr: number, nodesLen: number, edgesPtr: number, edgesLen: number, groupsPtr: number, groupsLen: number, outPtr: number, outCap: number, optsPtr: number, optsLen: number) => number;
  validate: (elemPtr: number, elemLen: number, outPtr: number, outCap: number) => number;
  zlibCompress: (inPtr: number, inLen: number, outPtr: number, outCap: number) => number;
  svgToPng: (svgPtr: number, svgLen: number, width: number, height: number, outPtr: number, outCap: number) => number;
}

let wasmInstance: WasmLayoutExports | null = null;
let wasmLock: Promise<void> = Promise.resolve();
let wasmLoadPromise: Promise<void> | null = null;

/** Serialize access to WASM — bump allocator is not concurrent-safe. */
async function withWasmLock<T>(fn: () => T): Promise<T> {
  const prev = wasmLock;
  let resolve!: () => void;
  wasmLock = new Promise(r => { resolve = r; });
  await prev;
  try {
    return fn();
  } finally {
    resolve();
  }
}

const WasmLayoutOutputSchema = z.object({
  nodes: z.array(z.object({ id: z.string(), x: z.number(), y: z.number() })),
  edges: z.array(z.object({
    from: z.string(),
    to: z.string(),
    points: z.array(z.tuple([z.number(), z.number()])),
    startFixedPoint: z.tuple([z.number(), z.number()]).optional(),
    endFixedPoint: z.tuple([z.number(), z.number()]).optional(),
    labelX: z.number().optional(),
    labelY: z.number().optional(),
  })).optional(),
  groups: z.array(z.object({
    id: z.string(),
    x: z.number(), y: z.number(),
    width: z.number(), height: z.number(),
  })).optional(),
});

/** WASI shim — minimal syscall implementations for layout-only WASM. */
function makeWasiImports(memRef: { memory: WebAssembly.Memory | null }) {
  return {
    wasi_snapshot_preview1: {
      environ_get: () => 0,
      environ_sizes_get: (_countPtr: number, _sizePtr: number) => {
        if (memRef.memory) {
          const view = new DataView(memRef.memory.buffer);
          view.setUint32(_countPtr, 0, true);
          view.setUint32(_sizePtr, 0, true);
        }
        return 0;
      },
      clock_time_get: (_id: number, _precision: bigint, resultPtr: number) => {
        if (memRef.memory) {
          new DataView(memRef.memory.buffer).setBigUint64(resultPtr, BigInt(0), true);
        }
        return 0;
      },
      fd_close: () => 0,
      fd_fdstat_get: () => 0,
      fd_fdstat_set_flags: () => 0,
      fd_filestat_get: () => 8,
      fd_prestat_get: () => 8,
      fd_prestat_dir_name: () => 8,
      fd_pwrite: () => 0,
      fd_read: () => 0,
      fd_seek: () => 0,
      path_open: () => 8,
      fd_write: (_fd: number, iovs: number, iovsLen: number, nwrittenPtr: number) => {
        if (memRef.memory) {
          // Sum iov lengths and report all bytes as written (discard output).
          // Returning 0 would cause musl stdio to retry infinitely.
          const view = new DataView(memRef.memory.buffer);
          let total = 0;
          for (let i = 0; i < iovsLen; i++) {
            total += view.getUint32(iovs + i * 8 + 4, true);
          }
          view.setUint32(nwrittenPtr, total, true);
        }
        return 0;
      },
      path_filestat_get: () => 8,
      proc_exit: (code: number) => { throw new WasiExit(code); },
    },
  };
}

/** Initialize WASM instance and wire up exports. */
function wireInstance(instance: WebAssembly.Instance, ref: { memory: WebAssembly.Memory | null }): void {
  ref.memory = (instance.exports as Record<string, unknown>).memory as WebAssembly.Memory;
  const start = (instance.exports as Record<string, unknown>)._start as (() => void) | undefined;
  if (start) {
    try { start(); } catch (e) {
      if (!(e instanceof WasiExit)) throw e;
    }
  }
  wasmInstance = instance.exports as unknown as WasmLayoutExports;
}

/** Initialize WASM from compiled module or bytes. */
async function initWasm(source: BufferSource | WebAssembly.Module): Promise<void> {
  const ref: { memory: WebAssembly.Memory | null } = { memory: null };
  const wasiImports = makeWasiImports(ref);
  if (source instanceof WebAssembly.Module) {
    const instance = await WebAssembly.instantiate(source, wasiImports);
    wireInstance(instance, ref);
  } else {
    const result = await WebAssembly.instantiate(source, wasiImports) as unknown as { instance: WebAssembly.Instance };
    wireInstance(result.instance, ref);
  }
}

/**
 * Load the WASM module.
 * - No args: reads from filesystem (local Node.js)
 * - WebAssembly.Module: instantiates directly (Cloudflare Worker)
 * - BufferSource: compiles and instantiates (any runtime)
 */
export async function loadWasm(source?: string | WebAssembly.Module | BufferSource): Promise<void> {
  if (wasmLoadPromise) return wasmLoadPromise;
  wasmLoadPromise = (async () => {
    try {
      if (source instanceof WebAssembly.Module || source instanceof ArrayBuffer || ArrayBuffer.isView(source)) {
        await initWasm(source);
      } else {
        const { readFile } = await import("node:fs/promises");
        const { join } = await import("node:path");
        const dir = await getDirname();
        const path = source ?? join(dir, "..", "wasm", "zig-out", "bin", "drawmode.wasm");
        const bytes = await readFile(path);
        await initWasm(bytes);
      }
    } catch (e) {
      if (typeof process !== "undefined") process.stderr.write(`[drawmode] WASM load failed: ${e}\n`);
      wasmInstance = null;
      wasmLoadPromise = null; // allow retry on failure
    }
  })();
  return wasmLoadPromise;
}

export function isWasmLoaded(): boolean {
  return wasmInstance !== null;
}

function writeToWasm(data: Uint8Array): number {
  if (!wasmInstance) throw new Error("WASM not loaded");
  const ptr = wasmInstance.alloc(data.byteLength);
  new Uint8Array(wasmInstance.memory.buffer, ptr, data.byteLength).set(data);
  return ptr;
}

function readFromWasm(ptr: number, len: number): Uint8Array {
  if (!wasmInstance) throw new Error("WASM not loaded");
  return new Uint8Array(wasmInstance.memory.buffer, ptr, len).slice();
}

/** Call a single-input WASM function: encode JSON → call → decode result. */
function callWasmSync(
  fn: (inPtr: number, inLen: number, outPtr: number, outCap: number) => number,
  inputJson: string,
  outCap: number,
): string | null {
  if (!wasmInstance) throw new Error("WASM not loaded");
  wasmInstance.resetHeap();
  const inBytes = encoder.encode(inputJson);
  const inPtr = writeToWasm(inBytes);
  const outPtr = wasmInstance.alloc(outCap);
  const written = fn(inPtr, inBytes.byteLength, outPtr, outCap);
  return written > 0 ? decoder.decode(readFromWasm(outPtr, written)) : null;
}

/**
 * Run WASM Sugiyama layout on nodes and edges.
 * Returns positioned nodes and edge routes with orthogonal routing points.
 */
export async function layoutGraphWasm(
  nodes: { id: string; width: number; height: number; row?: number; col?: number; absX?: number; absY?: number; type?: string }[],
  edges: { from: string; to: string; label?: string }[],
  groups?: { id: string; label: string; children: string[]; parent?: string }[],
  options?: { rankdir?: string; engine?: string },
): Promise<WasmLayoutResult | null> {
  if (!wasmInstance) throw new Error("WASM not loaded — run loadWasm() first");
  return withWasmLock(() => layoutGraphWasmInner(nodes, edges, groups, options));
}

function layoutGraphWasmInner(
  nodes: { id: string; width: number; height: number; row?: number; col?: number; absX?: number; absY?: number; type?: string }[],
  edges: { from: string; to: string; label?: string }[],
  groups?: { id: string; label: string; children: string[]; parent?: string }[],
  options?: { rankdir?: string; engine?: string },
): WasmLayoutResult | null {
  if (!wasmInstance) throw new Error("WASM not loaded");

  const nodesJson = JSON.stringify(
    nodes.map(n => ({
      id: n.id,
      width: n.width,
      height: n.height,
      row: n.row ?? null,
      col: n.col ?? null,
      absX: n.absX ?? null,
      absY: n.absY ?? null,
    })),
  );
  const edgesJson = JSON.stringify(
    edges.map(e => ({ from: e.from, to: e.to, label: e.label ?? "" })),
  );
  const groupsJson = JSON.stringify(
    (groups ?? []).map(g => ({ id: g.id, label: g.label, children: g.children, parent: g.parent ?? "" })),
  );

  const optsJson = JSON.stringify({ rankdir: options?.rankdir ?? "TB", engine: options?.engine ?? "dot" });

  const nodesBytes = encoder.encode(nodesJson);
  const edgesBytes = encoder.encode(edgesJson);
  const groupsBytes = encoder.encode(groupsJson);
  const optsBytes = encoder.encode(optsJson);

  // Scale output buffer based on input size (~300 bytes per node + ~250 per edge)
  const inputSize = nodesBytes.byteLength + edgesBytes.byteLength + groupsBytes.byteLength;
  const outCap = Math.max(128 * 1024, inputSize * 4);

  wasmInstance.resetHeap();
  const nodesPtr = writeToWasm(nodesBytes);
  const edgesPtr = writeToWasm(edgesBytes);
  const groupsPtr = writeToWasm(groupsBytes);
  const optsPtr = writeToWasm(optsBytes);
  const outPtr = wasmInstance.alloc(outCap);
  const written = wasmInstance.layoutGraph(
    nodesPtr, nodesBytes.byteLength,
    edgesPtr, edgesBytes.byteLength,
    groupsPtr, groupsBytes.byteLength,
    outPtr, outCap,
    optsPtr, optsBytes.byteLength,
  );

  if (written === 0) return null;

  try {
    const resultStr = decoder.decode(readFromWasm(outPtr, written));
    const result = WasmLayoutOutputSchema.parse(JSON.parse(resultStr));

    // Build edge routes map
    const edgeRoutes = new Map<string, EdgeRoute>();
    const edgePairCounts = new Map<string, number>();

    for (const edge of result.edges ?? []) {
      const baseKey = `${edge.from}->${edge.to}`;
      const pairIdx = edgePairCounts.get(baseKey) ?? 0;
      edgePairCounts.set(baseKey, pairIdx + 1);
      const key = pairIdx === 0 ? baseKey : `${baseKey}#${pairIdx}`;

      if (edge.points && edge.points.length >= 2) {
        // Use Zig-computed label position (with collision avoidance),
        // fall back to midpoint of longest segment
        let labelPos: { x: number; y: number };
        if (edge.labelX !== undefined && edge.labelY !== undefined) {
          labelPos = { x: edge.labelX, y: edge.labelY };
        } else {
          let bestLen = 0, bestSeg = 0;
          for (let s = 0; s < edge.points.length - 1; s++) {
            const dx = edge.points[s + 1][0] - edge.points[s][0];
            const dy = edge.points[s + 1][1] - edge.points[s][1];
            const segLen = Math.abs(dx) + Math.abs(dy);
            if (segLen > bestLen) { bestLen = segLen; bestSeg = s; }
          }
          const p1 = edge.points[bestSeg], p2 = edge.points[bestSeg + 1];
          labelPos = { x: Math.round((p1[0] + p2[0]) / 2), y: Math.round((p1[1] + p2[1]) / 2) };
        }

        edgeRoutes.set(key, {
          points: edge.points,
          labelPos,
          startFixedPoint: edge.startFixedPoint,
          endFixedPoint: edge.endFixedPoint,
        });
      }
    }

    return { nodes: result.nodes, edgeRoutes, groupBounds: result.groups };
  } catch (e) {
    if (typeof process !== "undefined") process.stderr.write(`[drawmode] WASM layout failed: ${e}\n`);
    return null;
  }
}

/** Validate Excalidraw elements. Returns validation errors JSON, or null. */
export async function validateElements(elementsJson: string): Promise<string | null> {
  if (!wasmInstance) throw new Error("WASM not loaded — run loadWasm() first");
  return withWasmLock(() =>
    callWasmSync(wasmInstance!.validate.bind(wasmInstance), elementsJson, 16 * 1024),
  );
}

/** Compress data using zlib format (matching pako.deflate). Returns compressed bytes or null. */
export async function zlibCompress(data: Uint8Array): Promise<Uint8Array | null> {
  if (!wasmInstance) throw new Error("WASM not loaded — run loadWasm() first");
  return withWasmLock(() => {
    wasmInstance!.resetHeap();
    const inPtr = writeToWasm(data);
    const outCap = data.byteLength + 1024; // compressed + zlib overhead
    const outPtr = wasmInstance!.alloc(outCap);
    const written = wasmInstance!.zlibCompress(inPtr, data.byteLength, outPtr, outCap);
    return written > 0 ? readFromWasm(outPtr, written) : null;
  });
}

/** Convert SVG string to PNG bytes via PlutoSVG in WASM. Returns PNG Uint8Array or null. */
export async function svgToPngWasm(svgString: string, width = 0, height = 0): Promise<Uint8Array | null> {
  if (!wasmInstance) throw new Error("WASM not loaded — run loadWasm() first");
  return withWasmLock(() => {
    wasmInstance!.resetHeap();
    const svgBytes = encoder.encode(svgString);
    const svgPtr = writeToWasm(svgBytes);
    // PNG output is typically smaller than SVG input, but allocate generously
    // for large diagrams: width * height * 4 (RGBA) as upper bound, minimum 2MB
    const outCap = Math.max(2 * 1024 * 1024, width * height * 4);
    const outPtr = wasmInstance!.alloc(outCap);
    if (outPtr === 0) return null;
    const written = wasmInstance!.svgToPng(svgPtr, svgBytes.byteLength, width, height, outPtr, outCap);
    return written > 0 ? readFromWasm(outPtr, written) : null;
  });
}

