/**
 * Diagram SDK — high-level API for building Excalidraw diagrams.
 * Hides all Excalidraw JSON complexity (bound text, arrow routing, edge math).
 */

import type {
  ColorPreset, ShapeOpts, ConnectOpts, RenderOpts, RenderResult,
  GraphNode, GraphEdge, FillStyle, StrokeStyle, FontFamily,
  Arrowhead, TextAlign, VerticalAlign, ColorPair,
  ExcalidrawElement, ExcalidrawFile, OutputFormat,
  ThemePreset, ThemeOpts, LayoutDirection, DiagramType, GroupOpts,
} from "./types.js";
import { COLOR_PALETTE, EXCALIDRAW_VERSION, ExcalidrawFileSchema } from "./types.js";
import {
  validateElements, isWasmLoaded, loadWasm,
  layoutGraphWasm,
  type EdgeRoute,
  type GroupBounds,
} from "./layout.js";

interface PositionedNode extends GraphNode {
  x?: number;
  y?: number;
}

/** Compare old and new Excalidraw elements to produce a human-readable change summary. */
function computeChangeSummary(oldElements: ExcalidrawElement[], newElements: ExcalidrawElement[]): string | undefined {
  // Index old elements: shapes/text by id, arrows by _from+_to
  const oldShapes = new Map<string, ExcalidrawElement>();
  const newShapes = new Map<string, ExcalidrawElement>();
  let oldArrowCount = 0, newArrowCount = 0;

  for (const el of oldElements) {
    if (el.type === "arrow") { oldArrowCount++; continue; }
    // Skip bound text (they follow their container)
    if (el.type === "text" && el.containerId) continue;
    // Skip group labels
    if (el.type === "text" && el.id.endsWith("-label")) continue;
    oldShapes.set(el.id, el);
  }
  for (const el of newElements) {
    if (el.type === "arrow") { newArrowCount++; continue; }
    if (el.type === "text" && el.containerId) continue;
    if (el.type === "text" && el.id.endsWith("-label")) continue;
    newShapes.set(el.id, el);
  }

  // Pre-index bound text by containerId for O(1) lookup
  const newTextByContainer = new Map<string, ExcalidrawElement>();
  for (const e of newElements) {
    if (e.type === "text" && e.containerId) newTextByContainer.set(e.containerId, e);
  }
  const oldTextByContainer = new Map<string, ExcalidrawElement>();
  for (const e of oldElements) {
    if (e.type === "text" && e.containerId) oldTextByContainer.set(e.containerId, e);
  }

  const added: string[] = [];
  const removed: string[] = [];
  const moved: string[] = [];
  let unchanged = 0;

  const modified: string[] = [];

  // Find added, modified, moved, unchanged
  for (const [id, el] of newShapes) {
    const old = oldShapes.get(id);
    if (!old) {
      const boundText = newTextByContainer.get(id);
      const label = el.text ?? boundText?.text ?? id;
      added.push(`"${label}" (${el.type})`);
    } else {
      const boundTextNew = newTextByContainer.get(id);
      const boundTextOld = oldTextByContainer.get(id);
      const newLabel = el.text ?? boundTextNew?.text ?? "";
      const oldLabel = old.text ?? boundTextOld?.text ?? "";
      const dx = Math.abs((el.x ?? 0) - (old.x ?? 0));
      const dy = Math.abs((el.y ?? 0) - (old.y ?? 0));
      if (newLabel !== oldLabel) {
        modified.push(`"${oldLabel}" → "${newLabel}"`);
      } else if (dx > 5 || dy > 5) {
        moved.push(`"${newLabel}"`);
      } else {
        unchanged++;
      }
    }
  }

  // Find removed
  for (const [id, el] of oldShapes) {
    if (!newShapes.has(id)) {
      const boundText = oldTextByContainer.get(id);
      const label = el.text ?? boundText?.text ?? id;
      removed.push(`"${label}" (${el.type})`);
    }
  }

  const edgeDiff = newArrowCount - oldArrowCount;

  // No changes at all
  if (added.length === 0 && removed.length === 0 && moved.length === 0 && modified.length === 0 && edgeDiff === 0) return undefined;

  const lines: string[] = [];
  if (added.length > 0) lines.push(`+ Added: ${added.join(", ")}`);
  if (removed.length > 0) lines.push(`- Removed: ${removed.join(", ")}`);
  if (modified.length > 0) lines.push(`~ Modified: ${modified.join(", ")}`);
  if (moved.length > 0) lines.push(`~ Moved: ${moved.join(", ")}`);
  if (edgeDiff > 0) lines.push(`+ ${edgeDiff} edge(s) added`);
  if (edgeDiff < 0) lines.push(`- ${Math.abs(edgeDiff)} edge(s) removed`);
  if (unchanged > 0) lines.push(`  Unchanged: ${unchanged} node(s)`);
  return lines.join("\n");
}

const ICON_PRESETS: Record<string, string> = {
  lambda: "λ", docker: "🐳", database: "🗄️", db: "🗄️",
  cloud: "☁️", lock: "🔒", globe: "🌐", server: "🖥️",
  api: "🔌", queue: "📨", cache: "⚡", storage: "💾",
  user: "👤", users: "👥", warning: "⚠️", check: "✅",
  fire: "🔥", key: "🔑", mail: "📧", search: "🔍",
  kubernetes: "☸️", k8s: "☸️",
};

const THEME_PRESETS: Record<ThemePreset, ThemeOpts> = {
  default: {},
  sketch: { fillStyle: "hachure", roughness: 2, fontFamily: 1 },
  blueprint: { fillStyle: "solid", roughness: 0, strokeWidth: 1, fontFamily: 3 },
  minimal: { fillStyle: "solid", roughness: 0, strokeWidth: 1, fontFamily: 2, opacity: 90 },
};

const DEFAULT_WIDTH = 180;
const DEFAULT_HEIGHT = 80;
const EXTRA_LINE_PX = 24; // extra height per additional line of text
const COL_SPACING = 280;
const ROW_SPACING = 220;
const BASE_X = 100;
const BASE_Y = 100;

/** Excalidraw lineHeight per font family: 1=Virgil→1.25, 2=Helvetica→1.15, 3=Cascadia→1.2 */
function getLineHeight(fontFamily: FontFamily): number {
  if (fontFamily === 2) return 1.15;
  if (fontFamily === 3) return 1.2;
  return 1.25; // Virgil (1) and default
}

/** Average character width as a fraction of fontSize, per font family */
const CHAR_WIDTH_FACTOR: Record<FontFamily, number> = {
  1: 0.60,  // Virgil (handwritten)
  2: 0.55,  // Helvetica (proportional)
  3: 0.60,  // Cascadia (monospace)
};

/** Measure text dimensions accounting for font family and size.
 *  Emoji/wide chars (codepoint > 0x1F00) count as 2x width. */
function measureText(text: string, fontSize = 16, fontFamily: FontFamily = 1): { width: number; height: number } {
  const lines = text.split("\n");
  const charWidth = fontSize * CHAR_WIDTH_FACTOR[fontFamily];
  const maxLineLen = Math.max(...lines.map(l => {
    let w = 0;
    for (const ch of l) {
      w += (ch.codePointAt(0)! > 0x1F00) ? 2 : 1;
    }
    return w;
  }));
  return {
    width: maxLineLen * charWidth,
    height: lines.length * fontSize * getLineHeight(fontFamily),
  };
}

const SESSION_SEED = Date.now().toString(36);

function randSeed(): number {
  return Math.floor(Math.random() * 2000000000);
}

function computeNodeBounds(nodes: { x?: number; y?: number; width: number; height: number }[]): { minX: number; minY: number; maxX: number; maxY: number } {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const n of nodes) {
    const nx = n.x ?? 0, ny = n.y ?? 0;
    if (nx < minX) minX = nx;
    if (ny < minY) minY = ny;
    if (nx + n.width > maxX) maxX = nx + n.width;
    if (ny + n.height > maxY) maxY = ny + n.height;
  }
  return { minX, minY, maxX, maxY };
}

function computeBounds(points: number[][]): { minX: number; minY: number; maxX: number; maxY: number } {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of points) {
    if (p[0] < minX) minX = p[0];
    if (p[1] < minY) minY = p[1];
    if (p[0] > maxX) maxX = p[0];
    if (p[1] > maxY) maxY = p[1];
  }
  return { minX, minY, maxX, maxY };
}

/** AABB overlap check with optional padding */
function rectsOverlap(
  ax: number, ay: number, aw: number, ah: number,
  bx: number, by: number, bw: number, bh: number,
  pad = 0,
): boolean {
  return ax - pad < bx + bw && ax + aw + pad > bx &&
         ay - pad < by + bh && ay + ah + pad > by;
}

type Rect = { x: number; y: number; width: number; height: number };

/**
 * Resolve overlapping arrow labels against all static elements.
 * Iterates until no label moves or budget is exhausted.
 *
 * Checks labels against: other labels, node shapes, group boundaries, arrow segments.
 */
function resolveOverlaps(elements: ExcalidrawElement[], budget = 10): void {
  // Moveable: arrow labels (free-standing text with "arrlbl_" prefix)
  const labels = elements.filter(
    el => el.type === "text" && el.id.startsWith("arrlbl_"),
  );
  if (labels.length === 0) return;

  // Static obstacles: shapes (not groups, not group labels, not bound text)
  const nodeRects: Rect[] = [];
  for (const el of elements) {
    if ((el.type === "rectangle" || el.type === "ellipse" || el.type === "diamond") &&
        !el.id.startsWith("grp_") && !el.id.startsWith("frm_")) {
      nodeRects.push(el);
    }
  }

  // Group boundaries (dashed rectangles)
  const groupRects: Rect[] = [];
  for (const el of elements) {
    if (el.id.startsWith("grp_") && !el.id.endsWith("-label")) {
      groupRects.push(el);
    }
  }

  // Arrow segments, tagged with their label ID so we can skip self-overlap
  const arrowSegs: { x1: number; y1: number; x2: number; y2: number; labelId?: string }[] = [];
  for (const el of elements) {
    if (el.type === "arrow" && el.points && el.points.length >= 2) {
      const labelId = (el.customData as Record<string, unknown> | undefined)?._labelId as string | undefined;
      for (let s = 0; s < el.points.length - 1; s++) {
        arrowSegs.push({
          x1: el.x + el.points[s][0], y1: el.y + el.points[s][1],
          x2: el.x + el.points[s + 1][0], y2: el.y + el.points[s + 1][1],
          labelId,
        });
      }
    }
  }

  // Check if a label rect overlaps any obstacle (skip own arrow segments)
  const hasCollision = (idx: number, lx: number, ly: number, lw: number, lh: number): boolean => {
    const selfId = labels[idx].id;
    // vs other labels
    for (let j = 0; j < labels.length; j++) {
      if (j === idx) continue;
      const o = labels[j];
      if (rectsOverlap(lx, ly, lw, lh, o.x, o.y, o.width, o.height)) return true;
    }
    // vs nodes
    for (const n of nodeRects) {
      if (rectsOverlap(lx, ly, lw, lh, n.x, n.y, n.width, n.height)) return true;
    }
    // vs group boundaries (only the border strip, not the interior)
    for (const g of groupRects) {
      const borderW = 4; // approximate stroke hit area
      // top edge
      if (rectsOverlap(lx, ly, lw, lh, g.x, g.y, g.width, borderW)) return true;
      // bottom edge
      if (rectsOverlap(lx, ly, lw, lh, g.x, g.y + g.height - borderW, g.width, borderW)) return true;
      // left edge
      if (rectsOverlap(lx, ly, lw, lh, g.x, g.y, borderW, g.height)) return true;
      // right edge
      if (rectsOverlap(lx, ly, lw, lh, g.x + g.width - borderW, g.y, borderW, g.height)) return true;
    }
    // vs arrow segments (AABB approximation) — skip own arrow
    // Pad segments by 10px to account for visual stroke width + arrowheads
    const SEG_PAD = 10;
    for (const seg of arrowSegs) {
      if (seg.labelId === selfId) continue; // don't collide with own arrow
      const sx = Math.min(seg.x1, seg.x2) - SEG_PAD, sy = Math.min(seg.y1, seg.y2) - SEG_PAD;
      const sw = (Math.abs(seg.x2 - seg.x1) || 2) + SEG_PAD * 2;
      const sh = (Math.abs(seg.y2 - seg.y1) || 2) + SEG_PAD * 2;
      if (rectsOverlap(lx, ly, lw, lh, sx, sy, sw, sh)) return true;
    }
    return false;
  };

  // Shift offsets: small nudges first, then larger jumps in 8 directions
  const makeOffsets = (w: number, h: number) => [
    // Small nudges (just clear the overlap)
    { dx: 0, dy: -(h + 4) },
    { dx: 0, dy: h + 4 },
    { dx: -(w / 2 + 4), dy: 0 },
    { dx: w / 2 + 4, dy: 0 },
    // Medium shifts
    { dx: 0, dy: -(h + 10) },
    { dx: 0, dy: h + 10 },
    { dx: -(w + 6), dy: 0 },
    { dx: w + 6, dy: 0 },
    // Diagonal shifts
    { dx: -(w + 6), dy: -(h + 6) },
    { dx: w + 6, dy: -(h + 6) },
    { dx: -(w + 6), dy: h + 6 },
    { dx: w + 6, dy: h + 6 },
    // Large shifts
    { dx: 0, dy: -(h + 24) },
    { dx: 0, dy: h + 24 },
    { dx: -(w + 24), dy: 0 },
    { dx: w + 24, dy: 0 },
  ];

  // Iterate until stable or budget exhausted
  for (let iter = 0; iter < budget; iter++) {
    let moved = false;
    for (let i = 0; i < labels.length; i++) {
      const cur = labels[i];
      if (!hasCollision(i, cur.x, cur.y, cur.width, cur.height)) continue;

      const offsets = makeOffsets(cur.width, cur.height);
      for (const { dx, dy } of offsets) {
        const nx = cur.x + dx, ny = cur.y + dy;
        if (!hasCollision(i, nx, ny, cur.width, cur.height)) {
          cur.x = nx;
          cur.y = ny;
          moved = true;
          break;
        }
      }
    }
    if (!moved) break; // stable
  }
}

/**
 * Nudge text nodes that partially overlap box node edges.
 * Skips text fully contained within a box (intentional content/descriptions).
 * Only pushes text that straddles a box boundary.
 */
function resolveTextBoxOverlaps(positioned: Map<string, PositionedNode>, gap = 8, budget = 10): void {
  const textNodes: PositionedNode[] = [];
  const boxNodes: PositionedNode[] = [];
  for (const n of positioned.values()) {
    if (n.type === "text") textNodes.push(n);
    else if (n.type === "rectangle" || n.type === "ellipse" || n.type === "diamond") boxNodes.push(n);
  }
  if (textNodes.length === 0 || boxNodes.length === 0) return;

  for (let iter = 0; iter < budget; iter++) {
    let moved = false;
    for (const t of textNodes) {
      const tx = t.x ?? 0, ty = t.y ?? 0;
      const tr = tx + t.width, tb = ty + t.height;
      for (const b of boxNodes) {
        const bx = b.x ?? 0, by = b.y ?? 0;
        const br = bx + b.width, bb = by + b.height;
        if (!rectsOverlap(tx, ty, t.width, t.height, bx, by, b.width, b.height, 0)) continue;
        // Skip text fully inside the box (intentional content)
        if (tx >= bx && ty >= by && tr <= br && tb <= bb) continue;
        // Text partially overlaps box edge — push to nearest edge
        const pushLeft = bx - gap - t.width - tx;
        const pushRight = br + gap - tx;
        const pushUp = by - gap - t.height - ty;
        const pushDown = bb + gap - ty;
        const candidates = [
          { d: Math.abs(pushLeft), dx: pushLeft, dy: 0 },
          { d: Math.abs(pushRight), dx: pushRight, dy: 0 },
          { d: Math.abs(pushUp), dx: 0, dy: pushUp },
          { d: Math.abs(pushDown), dx: 0, dy: pushDown },
        ];
        candidates.sort((a, c) => a.d - c.d);
        t.x = tx + candidates[0].dx;
        t.y = ty + candidates[0].dy;
        moved = true;
        break;
      }
    }
    if (!moved) break;
  }
}

export class Diagram {
  private nodes = new Map<string, GraphNode>();
  private edges: GraphEdge[] = [];
  private groups = new Map<string, { label: string; children: string[]; opts?: GroupOpts }>();
  private frames = new Map<string, { name: string; children: string[] }>();
  /** Passthrough elements from fromFile() — re-emitted unchanged */
  private passthrough: ExcalidrawElement[] = [];
  private idCounter = 0;
  private themeDefaults: ThemeOpts = {};
  private direction: LayoutDirection = "TB";
  private diagramType: DiagramType = "architecture";
  private sequenceActors: { id: string; label: string; index: number; opts?: ShapeOpts }[] = [];
  private sequenceMessages: { from: string; to: string; label?: string; index: number; opts?: ConnectOpts }[] = [];

  constructor(opts?: { theme?: ThemePreset; direction?: LayoutDirection; type?: DiagramType }) {
    if (opts?.theme) {
      this.themeDefaults = THEME_PRESETS[opts.theme] ?? {};
    }
    if (opts?.direction) {
      this.direction = opts.direction;
    }
    if (opts?.type) {
      this.diagramType = opts.type;
    }
  }

  /** Internal access to the nodes map — used by static methods like fromMermaid. */
  private _getNodes(): Map<string, GraphNode> {
    return this.nodes;
  }

  /** Set a theme preset that applies defaults to all subsequently added shapes. */
  setTheme(theme: ThemePreset): void {
    this.themeDefaults = THEME_PRESETS[theme] ?? {};
  }

  /** Set the layout direction (rankdir). */
  setDirection(direction: LayoutDirection): void {
    this.direction = direction;
  }

  private nextId(prefix: string): string {
    return `${prefix}_${++this.idCounter}_${SESSION_SEED}`;
  }

  /** Add a rectangle to the diagram. Returns the element ID. */
  addBox(label: string, opts?: ShapeOpts): string {
    return this.addShape("box", "rectangle", label, opts);
  }

  /** Add an ellipse to the diagram. Returns the element ID. */
  addEllipse(label: string, opts?: ShapeOpts): string {
    return this.addShape("ell", "ellipse", label, opts, "users");
  }

  /** Add a diamond to the diagram (for flowchart decisions). Returns the element ID. */
  addDiamond(label: string, opts?: ShapeOpts): string {
    return this.addShape("dia", "diamond", label, opts);
  }

  private addShape(prefix: string, type: GraphNode["type"], label: string, opts?: ShapeOpts, defaultPreset: ColorPreset = "backend"): string {
    const id = this.nextId(prefix);
    // Merge theme defaults under per-node opts (per-node wins)
    const mergedOpts: ShapeOpts | undefined = Object.keys(this.themeDefaults).length > 0
      ? { ...this.themeDefaults, ...opts } as ShapeOpts
      : opts;
    let displayLabel = label;
    if (mergedOpts?.icon) {
      const emoji = ICON_PRESETS[mergedOpts.icon] ?? mergedOpts.icon;
      displayLabel = `${emoji}\n${label}`;
    }
    const extraLines = displayLabel.split("\n").length - 1;
    const measured = measureText(displayLabel, mergedOpts?.fontSize ?? 16, mergedOpts?.fontFamily ?? 1);
    const autoWidth = Math.max(DEFAULT_WIDTH, measured.width + 40);
    // Minimum height must fit the text content + padding (20px top/bottom)
    const minTextHeight = measured.height + 20;
    const computedHeight = mergedOpts?.height ?? (DEFAULT_HEIGHT + extraLines * EXTRA_LINE_PX);
    this.nodes.set(id, {
      id, label: displayLabel, type,
      row: mergedOpts?.row, col: mergedOpts?.col,
      width: mergedOpts?.width ?? autoWidth,
      height: Math.max(computedHeight, minTextHeight),
      color: resolveColor(mergedOpts, defaultPreset),
      opts: mergedOpts,
      absX: mergedOpts?.x,
      absY: mergedOpts?.y,
    });
    return id;
  }

  /** Add standalone text (no container shape). Returns the element ID. */
  addText(text: string, opts?: {
    x?: number; y?: number;
    fontSize?: number; fontFamily?: FontFamily;
    color?: ColorPreset; strokeColor?: string;
  }): string {
    const id = this.nextId("txt");
    const preset = opts?.color ?? "backend";
    const paletteColor = COLOR_PALETTE[preset];
    const strokeColor = opts?.strokeColor ?? paletteColor.stroke;
    const fontSize = opts?.fontSize ?? 16;
    const textMeasured = measureText(text, fontSize, opts?.fontFamily ?? 1);
    this.nodes.set(id, {
      id, label: text, type: "text",
      width: textMeasured.width + 16,
      height: textMeasured.height + 8,
      color: { background: "transparent", stroke: strokeColor },
      opts: { x: opts?.x, y: opts?.y, fontSize: opts?.fontSize, fontFamily: opts?.fontFamily },
      absX: opts?.x,
      absY: opts?.y,
    });
    return id;
  }

  /** Add a line element (for dividers/boundaries). Returns the element ID. */
  addLine(points: [number, number][], opts?: {
    strokeColor?: string; strokeWidth?: number; strokeStyle?: StrokeStyle;
  }): string {
    if (points.length < 2) throw new Error("addLine requires at least two points");
    const id = this.nextId("line");
    const { minX, minY, maxX, maxY } = computeBounds(points);
    this.nodes.set(id, {
      id, label: "", type: "line",
      width: maxX - minX || 1,
      height: maxY - minY || 1,
      color: { background: "transparent", stroke: opts?.strokeColor ?? "#868e96" },
      opts: { strokeWidth: opts?.strokeWidth, strokeStyle: opts?.strokeStyle },
      absX: minX,
      absY: minY,
      linePoints: points.map(p => [p[0] - minX, p[1] - minY] as [number, number]),
    });
    return id;
  }

  /** Group elements together with a dashed boundary and label.
   *  Children can be node IDs or other group IDs (for nesting). */
  addGroup(label: string, children: string[], opts?: GroupOpts): string {
    for (const cid of children) {
      if (!this.nodes.has(cid) && !this.groups.has(cid)) {
        if (this.frames.has(cid)) throw new Error(`Cannot add frame "${cid}" to a group. Groups can contain nodes or other groups.`);
        throw new Error(`Child not found: "${cid}". Add nodes before grouping them.`);
      }
    }
    const id = this.nextId("grp");
    this.groups.set(id, { label, children, opts });
    return id;
  }

  /** Add a native Excalidraw frame container. Returns the frame ID. */
  addFrame(name: string, children: string[]): string {
    for (const cid of children) {
      if (!this.nodes.has(cid) && !this.groups.has(cid)) {
        if (this.frames.has(cid)) throw new Error(`Cannot nest frame "${cid}" inside another frame.`);
        throw new Error(`Child not found: "${cid}". Add nodes before framing them.`);
      }
    }
    const id = this.nextId("frm");
    this.frames.set(id, { name, children });
    return id;
  }

  /** Remove a group container. Children are kept. */
  removeGroup(id: string): void {
    if (!this.groups.has(id)) throw new Error(`Group not found: "${id}"`);
    this.groups.delete(id);
  }

  /** Remove a frame container. Children are kept. */
  removeFrame(id: string): void {
    if (!this.frames.has(id)) throw new Error(`Frame not found: "${id}"`);
    this.frames.delete(id);
  }

  /** Connect two elements with an arrow. */
  connect(from: string, to: string, label?: string, opts?: ConnectOpts): void {
    if (!this.nodes.has(from)) {
      if (this.groups.has(from)) throw new Error(`Cannot connect from group "${from}". Connect from a node inside the group instead.`);
      if (this.frames.has(from)) throw new Error(`Cannot connect from frame "${from}". Connect from a node inside the frame instead.`);
      throw new Error(`Source node not found: "${from}". Add the node before connecting it.`);
    }
    if (!this.nodes.has(to)) {
      if (this.groups.has(to)) throw new Error(`Cannot connect to group "${to}". Connect to a node inside the group instead.`);
      if (this.frames.has(to)) throw new Error(`Cannot connect to frame "${to}". Connect to a node inside the frame instead.`);
      throw new Error(`Target node not found: "${to}". Add the node before connecting it.`);
    }
    this.edges.push({
      from, to, label,
      style: opts?.style ?? "solid",
      opts,
    });
  }

  // ── Editing / Query Methods ──

  /** Load an existing .excalidraw file for editing. */
  static async fromFile(this: new () => Diagram, path: string): Promise<Diagram> {
    const { readFile } = await import("node:fs/promises");
    const raw = await readFile(path, "utf-8");
    const parsed = ExcalidrawFileSchema.safeParse(JSON.parse(raw));
    if (!parsed.success) throw new Error(`Invalid .excalidraw file: ${parsed.error.message}`);
    return (this as typeof Diagram).fromElements(parsed.data.elements);
  }

  /** Reconstruct a Diagram from raw Excalidraw elements (no filesystem needed). */
  static fromElements(this: new () => Diagram, elements: ExcalidrawElement[]): Diagram {
    const d = new this();

    // Index text elements by containerId for label lookup
    const textByContainer = new Map<string, ExcalidrawElement>();
    for (const el of elements) {
      if (el.type === "text" && el.containerId) {
        textByContainer.set(el.containerId, el);
      }
    }

    // Index label text elements by their parent ID (e.g. "grp_1-label" → keyed by "grp_1")
    const labelTextById = new Map<string, ExcalidrawElement>();
    for (const el of elements) {
      if (el.type === "text" && el.id.endsWith("-label")) {
        labelTextById.set(el.id.replace(/-label$/, ""), el);
      }
    }

    // Index all elements by ID for cross-referencing (e.g. arrow _labelId → text element)
    const elemById = new Map<string, ExcalidrawElement>();
    for (const el of elements) elemById.set(el.id, el);

    // Collect arrow label text IDs so they can be skipped when processing standalone text
    const arrowLabelIds = new Set<string>();
    for (const el of elements) {
      if (el.type === "arrow") {
        const labelId = (el.customData as Record<string, unknown> | undefined)?._labelId as string | undefined;
        if (labelId) arrowLabelIds.add(labelId);
      }
    }

    // Pre-detect group IDs so label text nodes can be skipped regardless of element order
    const groupIds = new Set<string>();
    for (const el of elements) {
      if ((el.type === "rectangle" || el.type === "ellipse") &&
          el.strokeStyle === "dashed" && el.backgroundColor === "transparent" &&
          (el.opacity ?? 100) <= 70) {
        if (labelTextById.has(el.id)) groupIds.add(el.id);
      }
    }

    // Reconstruct nodes from shapes
    for (const el of elements) {
      if (el.type === "rectangle" || el.type === "ellipse" || el.type === "diamond") {
        // Detect group boundaries: must have companion "-label" text, dashed stroke,
        // transparent background, and low opacity. The "-label" check distinguishes
        // drawmode groups from user shapes that happen to be dashed + low opacity.
        const labelEl = labelTextById.get(el.id);
        if (el.type !== "diamond" && labelEl && el.strokeStyle === "dashed" &&
            el.backgroundColor === "transparent" && (el.opacity ?? 100) <= 70) {
          d.groups.set(el.id, {
            label: labelEl?.text ?? "",
            children: [], // Reconstructed below after all nodes are loaded
            _bounds: { x: el.x, y: el.y, w: el.width, h: el.height },
          } as { label: string; children: string[]; _bounds?: { x: number; y: number; w: number; h: number } });
          continue;
        }

        const boundText = textByContainer.get(el.id);
        const label = boundText?.text ?? "";

        const node: GraphNode = {
          id: el.id,
          label,
          type: el.type as GraphNode["type"],
          width: el.width,
          height: el.height,
          color: {
            background: el.backgroundColor ?? "",
            stroke: el.strokeColor ?? "",
          },
          opts: {
            fillStyle: el.fillStyle as FillStyle | undefined,
            strokeWidth: el.strokeWidth,
            strokeStyle: el.strokeStyle as StrokeStyle | undefined,
            roughness: el.roughness,
            opacity: el.opacity,
            roundness: el.roundness,
            strokeColor: el.strokeColor,
            backgroundColor: el.backgroundColor,
            fontSize: boundText?.fontSize,
            fontFamily: boundText?.fontFamily as FontFamily | undefined,
            textAlign: boundText?.textAlign as TextAlign | undefined,
            verticalAlign: boundText?.verticalAlign as VerticalAlign | undefined,
            link: el.link,
            ...(el.customData !== undefined ? { customData: el.customData as Record<string, unknown> } : {}),
          },
          absX: el.x,
          absY: el.y,
        };
        d.nodes.set(el.id, node);
      } else if (el.type === "arrow") {
        // Recover edge endpoints from customData (preferred) or legacy bindings
        const startId = (el.customData as Record<string, unknown> | undefined)?._from as string | undefined
          ?? el.startBinding?.elementId;
        const endId = (el.customData as Record<string, unknown> | undefined)?._to as string | undefined
          ?? el.endBinding?.elementId;
        const isDrawmodeArrow = !!(el.customData as Record<string, unknown> | undefined)?._from;
        if (startId && endId && isDrawmodeArrow) {
          // Drawmode-generated arrow — reconstruct as edge for re-routing
          const labelId = (el.customData as Record<string, unknown> | undefined)?._labelId as string | undefined;
          const arrowLabel = labelId ? elemById.get(labelId) : textByContainer.get(el.id);
          d.edges.push({
            from: startId,
            to: endId,
            label: arrowLabel?.text,
            style: (el.strokeStyle as StrokeStyle) ?? "solid",
            opts: {
              strokeColor: el.strokeColor,
              strokeWidth: el.strokeWidth,
              roughness: el.roughness,
              opacity: el.opacity,
              startArrowhead: el.startArrowhead as Arrowhead | undefined,
              endArrowhead: el.endArrowhead as Arrowhead | undefined,
              elbowed: el.elbowed,
              labelFontSize: arrowLabel?.fontSize,
            },
          });
        } else {
          // External arrow (not drawmode-generated) — preserve original routing as passthrough.
          // Also preserve its bound label text element.
          d.passthrough.push(el);
          const boundLabel = textByContainer.get(el.id);
          if (boundLabel) d.passthrough.push(boundLabel);
        }
      } else if (el.type === "line") {
        // Reconstruct line element
        d.nodes.set(el.id, {
          id: el.id,
          label: "",
          type: "line",
          width: el.width,
          height: el.height,
          color: { background: "transparent", stroke: el.strokeColor ?? "" },
          opts: {
            strokeWidth: el.strokeWidth,
            strokeStyle: el.strokeStyle as StrokeStyle | undefined,
          },
          linePoints: (el.points as [number, number][]) ?? [],
          absX: el.x,
          absY: el.y,
        });
      } else if (el.type === "text" && !el.containerId) {
        // Skip group label text elements (detected in pre-scan above)
        if (el.id.endsWith("-label") && groupIds.has(el.id.replace(/-label$/, ""))) continue;
        // Skip arrow label text elements (identified via arrow customData._labelId)
        if (arrowLabelIds.has(el.id)) continue;

        // Standalone text — add as text node
        d.nodes.set(el.id, {
          id: el.id,
          label: el.text ?? "",
          type: "text",
          width: el.width,
          height: el.height,
          color: { background: "transparent", stroke: el.strokeColor ?? "" },
          opts: { fontSize: el.fontSize, fontFamily: el.fontFamily as FontFamily | undefined },
          absX: el.x,
          absY: el.y,
        });
      } else if (el.type === "frame") {
        // Native Excalidraw frame — reconstruct into frames map
        d.frames.set(el.id, {
          name: el.name ?? "",
          children: [], // Populated below using frameId references
        });
      } else if (el.type === "text" && el.containerId) {
        // Bound text — already handled via textByContainer, skip
      } else {
        // Unknown element type — passthrough
        d.passthrough.push(el);
      }
    }

    // Reconstruct group children: nodes whose position falls within group bounds
    for (const [groupId, group] of d.groups) {
      const gb = (group as { _bounds?: { x: number; y: number; w: number; h: number } })._bounds;
      if (!gb) continue;
      for (const node of d.nodes.values()) {
        const nx = node.absX ?? 0;
        const ny = node.absY ?? 0;
        if (nx >= gb.x && ny >= gb.y && nx + node.width <= gb.x + gb.w && ny + node.height <= gb.y + gb.h) {
          group.children.push(node.id);
        }
      }
      delete (group as Record<string, unknown>)._bounds;
    }

    // Reconstruct frame children: nodes whose element had frameId set
    for (const el of elements) {
      if (el.frameId && d.frames.has(el.frameId)) {
        // Only add shape nodes (not bound text elements) as frame children
        if (d.nodes.has(el.id)) {
          d.frames.get(el.frameId)!.children.push(el.id);
        }
      }
    }

    // Advance idCounter past any loaded IDs to avoid collisions when adding new nodes
    for (const id of [...d.nodes.keys(), ...d.groups.keys(), ...d.frames.keys()]) {
      const match = id.match(/^[a-z]+_(\d+)_/);
      if (match) {
        const num = parseInt(match[1], 10);
        if (num >= d.idCounter) d.idCounter = num + 1;
      }
    }

    return d;
  }

  /**
   * Generate TypeScript SDK code that reproduces this diagram.
   * Reverse-maps hex colors to presets, generates readable variable names,
   * and emits only non-default options to keep code minimal.
   */
  toCode(opts?: { path?: string }): string {
    const lines: string[] = [];

    // Build reverse color lookup: hex pair → preset name
    const reverseColor = new Map<string, ColorPreset>();
    for (const [preset, pair] of Object.entries(COLOR_PALETTE)) {
      reverseColor.set(`${pair.background}|${pair.stroke}`, preset as ColorPreset);
    }

    // Generate readable variable names from labels
    const varNames = new Map<string, string>();
    const usedVars = new Set<string>();
    const reserved = new Set([
      "break", "case", "catch", "class", "const", "continue", "debugger", "default",
      "delete", "do", "else", "export", "extends", "false", "finally", "for",
      "function", "if", "import", "in", "instanceof", "let", "new", "null",
      "return", "super", "switch", "this", "throw", "true", "try", "typeof",
      "var", "void", "while", "with", "yield", "await", "enum",
    ]);
    const toVarName = (label: string, fallback: string): string => {
      // Strip icon emoji prefix (first line if multi-line with emoji)
      const cleanLabel = label.replace(/^[^\w\n]*\n/, "");
      let name = cleanLabel
        .replace(/[^a-zA-Z0-9\s]/g, "")
        .trim()
        .split(/\s+/)
        .map((w, i) => i === 0 ? w.toLowerCase() : w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
        .join("");
      if (!name || /^\d/.test(name) || reserved.has(name)) name = fallback;
      // Deduplicate
      const base = name;
      let suffix = 2;
      while (usedVars.has(name)) { name = `${base}${suffix++}`; }
      usedVars.add(name);
      return name;
    };

    // Detect color preset from node's color pair
    const detectPreset = (color: { background: string; stroke: string }): ColorPreset | undefined => {
      return reverseColor.get(`${color.background}|${color.stroke}`);
    };

    // Build minimal ShapeOpts (only non-default fields)
    const buildShapeOpts = (node: GraphNode): Record<string, unknown> => {
      const o: Record<string, unknown> = {};
      const storedOpts = node.opts;
      if (storedOpts?.row !== undefined) o.row = storedOpts.row;
      if (storedOpts?.col !== undefined) o.col = storedOpts.col;
      const preset = detectPreset(node.color);
      if (preset) o.color = preset;
      if (storedOpts?.x !== undefined) o.x = storedOpts.x;
      if (storedOpts?.y !== undefined) o.y = storedOpts.y;
      if (storedOpts?.width) o.width = storedOpts.width;
      if (storedOpts?.height) o.height = storedOpts.height;
      // Only emit hex colors if no preset matched
      if (!preset) {
        if (storedOpts?.backgroundColor) o.backgroundColor = storedOpts.backgroundColor;
        if (storedOpts?.strokeColor) o.strokeColor = storedOpts.strokeColor;
      }
      if (storedOpts?.fillStyle && storedOpts.fillStyle !== "solid") o.fillStyle = storedOpts.fillStyle;
      if (storedOpts?.strokeWidth && storedOpts.strokeWidth !== 2) o.strokeWidth = storedOpts.strokeWidth;
      if (storedOpts?.strokeStyle && storedOpts.strokeStyle !== "solid") o.strokeStyle = storedOpts.strokeStyle;
      if (storedOpts?.roughness !== undefined && storedOpts.roughness !== 1) o.roughness = storedOpts.roughness;
      if (storedOpts?.opacity !== undefined && storedOpts.opacity !== 100) o.opacity = storedOpts.opacity;
      if (storedOpts?.fontSize && storedOpts.fontSize !== 16) o.fontSize = storedOpts.fontSize;
      if (storedOpts?.fontFamily && storedOpts.fontFamily !== 1) o.fontFamily = storedOpts.fontFamily;
      if (storedOpts?.textAlign && storedOpts.textAlign !== "center") o.textAlign = storedOpts.textAlign;
      if (storedOpts?.verticalAlign && storedOpts.verticalAlign !== "middle") o.verticalAlign = storedOpts.verticalAlign;
      if (storedOpts?.link) o.link = storedOpts.link;
      if (storedOpts?.icon) o.icon = storedOpts.icon;
      return o;
    };

    // Build minimal ConnectOpts
    const buildConnectOpts = (edge: GraphEdge): Record<string, unknown> => {
      const o: Record<string, unknown> = {};
      const eo = edge.opts;
      if (edge.style !== "solid") o.style = edge.style;
      if (eo?.strokeColor) o.strokeColor = eo.strokeColor;
      if (eo?.strokeWidth && eo.strokeWidth !== 2) o.strokeWidth = eo.strokeWidth;
      if (eo?.startArrowhead !== undefined && eo.startArrowhead !== null) o.startArrowhead = eo.startArrowhead;
      if (eo?.endArrowhead !== undefined && eo.endArrowhead !== "arrow") o.endArrowhead = eo.endArrowhead;
      if (eo?.elbowed === false) o.elbowed = false;
      if (eo?.labelFontSize && eo.labelFontSize !== 16) o.labelFontSize = eo.labelFontSize;
      if (eo?.labelPosition && eo.labelPosition !== "middle") o.labelPosition = eo.labelPosition;
      return o;
    };

    const stringify = (obj: Record<string, unknown>): string => {
      const entries = Object.entries(obj);
      if (entries.length === 0) return "";
      // Pretty-print simple objects on one line
      const parts = entries.map(([k, v]) => `${k}: ${JSON.stringify(v)}`);
      return `{ ${parts.join(", ")} }`;
    };

    // Constructor
    const ctorOpts: Record<string, string> = {};
    if (this.direction !== "TB") ctorOpts.direction = this.direction;
    if (this.diagramType !== "architecture") ctorOpts.type = this.diagramType;
    const ctorStr = Object.keys(ctorOpts).length > 0 ? stringify(ctorOpts) : "";
    lines.push(`const d = new Diagram(${ctorStr});`);

    // Sequence diagrams
    if (this.diagramType === "sequence") {
      lines.push("");
      for (const actor of this.sequenceActors) {
        const varName = toVarName(actor.label, `actor${actor.index}`);
        varNames.set(actor.id, varName);
        const shapeOpts = actor.opts ? buildShapeOpts({ id: actor.id, label: actor.label, type: "rectangle", width: 0, height: 0, color: { background: "", stroke: "" }, opts: actor.opts }) : {};
        const optsStr = Object.keys(shapeOpts).length > 0 ? `, ${stringify(shapeOpts)}` : "";
        lines.push(`const ${varName} = d.addActor(${JSON.stringify(actor.label)}${optsStr});`);
      }
      lines.push("");
      for (const msg of this.sequenceMessages) {
        const from = varNames.get(msg.from) ?? JSON.stringify(msg.from);
        const to = varNames.get(msg.to) ?? JSON.stringify(msg.to);
        const labelStr = msg.label ? `, ${JSON.stringify(msg.label)}` : "";
        const co = msg.opts ? buildConnectOpts({ from: msg.from, to: msg.to, label: msg.label, style: msg.opts?.style ?? "solid", opts: msg.opts }) : {};
        const optsStr = Object.keys(co).length > 0 ? `, ${stringify(co)}` : "";
        // Need empty label arg if we have opts but no label
        const args = optsStr && !labelStr ? `, undefined${optsStr}` : `${labelStr}${optsStr}`;
        lines.push(`d.message(${from}, ${to}${args});`);
      }
    } else {
      // Architecture diagrams: nodes, then groups, then edges
      lines.push("");

      // Extract shared styles: group nodes by their non-positional opts (color, icon, etc.)
      // Only extract when 3+ nodes share the same style (otherwise spread syntax adds tokens)
      const styleGroups = new Map<string, { style: Record<string, unknown>; name: string }>();
      const nodeStyleKeys = new Map<string, string>(); // node id → style key
      const shapeNodes = [...this.nodes.entries()].filter(([, n]) => n.type !== "line" && n.type !== "text");

      for (const [id, node] of shapeNodes) {
        const full = buildShapeOpts(node);
        // Separate positional from style properties
        const style: Record<string, unknown> = {};
        for (const [k, v] of Object.entries(full)) {
          if (k !== "row" && k !== "col" && k !== "x" && k !== "y" && k !== "width" && k !== "height") {
            style[k] = v;
          }
        }
        if (Object.keys(style).length > 0) {
          const key = JSON.stringify(style);
          nodeStyleKeys.set(id, key);
          if (!styleGroups.has(key)) {
            // Generate style variable name from the most common property
            const colorVal = style.color as string | undefined;
            const styleName = colorVal ? `${colorVal}Style` : `style${styleGroups.size + 1}`;
            styleGroups.set(key, { style, name: styleName });
          }
        }
      }

      // Emit shared style variables (only for groups with 3+ members)
      const styleKeyCounts = new Map<string, number>();
      for (const key of nodeStyleKeys.values()) {
        styleKeyCounts.set(key, (styleKeyCounts.get(key) ?? 0) + 1);
      }
      const emittedStyles = new Set<string>();
      for (const [key, count] of styleKeyCounts) {
        if (count >= 3) {
          const sg = styleGroups.get(key)!;
          lines.push(`const ${sg.name} = ${stringify(sg.style)};`);
          emittedStyles.add(key);
        }
      }
      if (emittedStyles.size > 0) lines.push("");

      // Emit nodes
      for (const [id, node] of this.nodes) {
        if (node.type === "line") {
          const varName = toVarName(node.label || "line", "line");
          varNames.set(id, varName);
          const pts = node.linePoints ?? [];
          const lineOpts: Record<string, unknown> = {};
          if (node.opts?.strokeColor) lineOpts.strokeColor = node.opts.strokeColor;
          if (node.opts?.strokeWidth) lineOpts.strokeWidth = node.opts.strokeWidth;
          if (node.opts?.strokeStyle) lineOpts.strokeStyle = node.opts.strokeStyle;
          const optsStr = Object.keys(lineOpts).length > 0 ? `, ${stringify(lineOpts)}` : "";
          lines.push(`const ${varName} = d.addLine(${JSON.stringify(pts)}${optsStr});`);
          continue;
        }

        const varName = toVarName(node.label, `node${this.idCounter}`);
        varNames.set(id, varName);

        if (node.type === "text") {
          const textOpts: Record<string, unknown> = {};
          if (node.absX !== undefined) textOpts.x = node.absX;
          if (node.absY !== undefined) textOpts.y = node.absY;
          if (node.opts?.fontSize && node.opts.fontSize !== 16) textOpts.fontSize = node.opts.fontSize;
          if (node.opts?.fontFamily && node.opts.fontFamily !== 1) textOpts.fontFamily = node.opts.fontFamily;
          const preset = detectPreset(node.color);
          if (preset) textOpts.color = preset;
          else if (node.color.stroke && node.color.stroke !== "#1e1e1e") textOpts.strokeColor = node.color.stroke;
          const optsStr = Object.keys(textOpts).length > 0 ? `, ${stringify(textOpts)}` : "";
          lines.push(`const ${varName} = d.addText(${JSON.stringify(node.label)}${optsStr});`);
          continue;
        }

        const method = node.type === "ellipse" ? "addEllipse" : node.type === "diamond" ? "addDiamond" : "addBox";
        // Strip icon emoji prefix from label (icon is emitted as an opt)
        let codeLabel = node.label;
        if (node.opts?.icon && codeLabel.includes("\n")) {
          codeLabel = codeLabel.split("\n").slice(1).join("\n");
        }

        // Use shared style variable if this node's style was extracted
        const styleKey = nodeStyleKeys.get(id);
        if (styleKey && emittedStyles.has(styleKey)) {
          const sg = styleGroups.get(styleKey)!;
          const full = buildShapeOpts(node);
          // Keep only positional properties inline
          const positional: Record<string, unknown> = {};
          for (const [k, v] of Object.entries(full)) {
            if (k === "row" || k === "col" || k === "x" || k === "y" || k === "width" || k === "height") {
              positional[k] = v;
            }
          }
          const posStr = Object.keys(positional).length > 0
            ? Object.entries(positional).map(([k, v]) => `${k}: ${JSON.stringify(v)}`).join(", ")
            : "";
          const optsStr = posStr
            ? `, { ...${sg.name}, ${posStr} }`
            : `, ${sg.name}`;
          lines.push(`const ${varName} = d.${method}(${JSON.stringify(codeLabel)}${optsStr});`);
        } else {
          const shapeOpts = buildShapeOpts(node);
          const optsStr = Object.keys(shapeOpts).length > 0 ? `, ${stringify(shapeOpts)}` : "";
          lines.push(`const ${varName} = d.${method}(${JSON.stringify(codeLabel)}${optsStr});`);
        }
      }

      // Emit groups
      if (this.groups.size > 0) {
        lines.push("");
        for (const [id, group] of this.groups) {
          const childVars = group.children.map(c => varNames.get(c) ?? `"${c}"`);
          const groupOpts: Record<string, unknown> = {};
          if (group.opts?.padding && group.opts.padding !== 30) groupOpts.padding = group.opts.padding;
          if (group.opts?.strokeColor) groupOpts.strokeColor = group.opts.strokeColor;
          if (group.opts?.strokeStyle && group.opts.strokeStyle !== "dashed") groupOpts.strokeStyle = group.opts.strokeStyle;
          if (group.opts?.opacity !== undefined && group.opts.opacity !== 60) groupOpts.opacity = group.opts.opacity;
          const optsStr = Object.keys(groupOpts).length > 0 ? `, ${stringify(groupOpts)}` : "";
          const varName = toVarName(group.label, `group${this.idCounter}`);
          varNames.set(id, varName);
          lines.push(`const ${varName} = d.addGroup(${JSON.stringify(group.label)}, [${childVars.join(", ")}]${optsStr});`);
        }
      }

      // Emit frames
      if (this.frames.size > 0) {
        lines.push("");
        for (const [id, frame] of this.frames) {
          const childVars = frame.children.map(c => varNames.get(c) ?? `"${c}"`);
          const varName = toVarName(frame.name, `frame${this.idCounter}`);
          varNames.set(id, varName);
          lines.push(`d.addFrame(${JSON.stringify(frame.name)}, [${childVars.join(", ")}]);`);
        }
      }

      // Emit edges
      if (this.edges.length > 0) {
        lines.push("");
        for (const edge of this.edges) {
          const from = varNames.get(edge.from) ?? `"${edge.from}"`;
          const to = varNames.get(edge.to) ?? `"${edge.to}"`;
          const co = buildConnectOpts(edge);
          const labelStr = edge.label ? `, ${JSON.stringify(edge.label)}` : "";
          const optsStr = Object.keys(co).length > 0 ? `, ${stringify(co)}` : "";
          // Need empty label arg if we have opts but no label
          const args = optsStr && !labelStr ? `, undefined${optsStr}` : `${labelStr}${optsStr}`;
          lines.push(`d.connect(${from}, ${to}${args});`);
        }
      }
    }

    // Render
    lines.push("");
    const renderOpts: Record<string, unknown> = {};
    if (opts?.path) renderOpts.path = opts.path;
    const renderStr = Object.keys(renderOpts).length > 0 ? stringify(renderOpts) : "";
    lines.push(`return d.render(${renderStr});`);

    return lines.join("\n");
  }

  /** Import a Mermaid graph definition. Supports graph TD/LR, nodes, edges, subgraphs. */
  static fromMermaid(syntax: string): Diagram {
    const d = new Diagram();
    const lines = syntax.split("\n").map(l => l.trim()).filter(l => l && !l.startsWith("%%"));

    // Parse direction — may be on its own line or before a semicolon
    let direction: LayoutDirection = "TB";
    if (lines[0]) {
      const dirMatch = lines[0].match(/^(?:graph|flowchart)\s+(TD|TB|LR|RL)\b/i);
      if (dirMatch) {
        const raw = dirMatch[1].toUpperCase();
        direction = (raw === "TD" ? "TB" : raw) as LayoutDirection;
        // Remove the direction prefix; keep anything after the semicolon
        const afterDir = lines[0].replace(/^(?:graph|flowchart)\s+(?:TD|TB|LR|RL)\s*;?\s*/i, "");
        if (afterDir) {
          lines[0] = afterDir;
        } else {
          lines.shift();
        }
      } else if (lines[0].match(/^(?:graph|flowchart)\s*$/i)) {
        lines.shift();
      }
    }

    // Track created node IDs by mermaid name
    const nodeMap = new Map<string, string>();
    // Track subgraph stack
    const subgraphStack: { label: string; children: string[] }[] = [];

    // Helper: ensure a node exists, create if needed
    const ensureNode = (name: string, label?: string, shape?: string): string => {
      if (nodeMap.has(name)) return nodeMap.get(name)!;
      const nodeLabel = label ?? name;
      let id: string;
      if (shape === "circle") {
        id = d.addEllipse(nodeLabel);
      } else if (shape === "diamond") {
        id = d.addDiamond(nodeLabel);
      } else if (shape === "database") {
        id = d.addBox(nodeLabel, { color: "database" });
      } else {
        id = d.addBox(nodeLabel);
      }
      nodeMap.set(name, id);
      // Add to current subgraph if any
      if (subgraphStack.length > 0) {
        subgraphStack[subgraphStack.length - 1].children.push(id);
      }
      return id;
    };

    // Parse node definition: extract name, label, and shape from patterns like A[Label], B{Label}, C((Label)), D[(Label)]
    const parseNodeDef = (raw: string): { name: string; label?: string; shape?: string } => {
      raw = raw.trim();
      // (( )) — circle
      let m = raw.match(/^([A-Za-z0-9_-]+)\(\((.+?)\)\)$/);
      if (m) return { name: m[1], label: m[2], shape: "circle" };
      // [( )] — database
      m = raw.match(/^([A-Za-z0-9_-]+)\[\((.+?)\)\]$/);
      if (m) return { name: m[1], label: m[2], shape: "database" };
      // { } — diamond
      m = raw.match(/^([A-Za-z0-9_-]+)\{(.+?)\}$/);
      if (m) return { name: m[1], label: m[2], shape: "diamond" };
      // ( ) — rounded box (just a box for us)
      m = raw.match(/^([A-Za-z0-9_-]+)\((.+?)\)$/);
      if (m) return { name: m[1], label: m[2], shape: "box" };
      // [ ] — box
      m = raw.match(/^([A-Za-z0-9_-]+)\[(.+?)\]$/);
      if (m) return { name: m[1], label: m[2], shape: "box" };
      // Plain name
      m = raw.match(/^([A-Za-z0-9_-]+)$/);
      if (m) return { name: m[1] };
      return { name: raw };
    };

    // Process lines (may contain multiple statements separated by ;)
    const statements: string[] = [];
    for (const line of lines) {
      for (const stmt of line.split(";")) {
        const s = stmt.trim();
        if (s) statements.push(s);
      }
    }

    // Edge pattern: captures source, arrow, optional label, and the rest (may chain)
    // Arrows: -->, ---, -.->  , ==>
    // The source is a non-greedy match of a node def (name + optional shape brackets)
    const edgeRegex = /^([A-Za-z0-9_-]+(?:\[.*?\]|\{.*?\}|\(\(.*?\)\)|\[\(.*?\)\]|\(.*?\))?)\s*(==>|-->|---|-\.->)\s*(?:\|([^|]*)\|\s*)?(.+)$/;

    const parseEdgeOpts = (arrow: string): ConnectOpts | undefined => {
      const connectOpts: ConnectOpts = {};
      if (arrow === "-.->") connectOpts.style = "dashed";
      if (arrow === "==>") connectOpts.strokeWidth = 4;
      if (arrow === "---") connectOpts.endArrowhead = null;
      return Object.keys(connectOpts).length > 0 ? connectOpts : undefined;
    };

    for (const stmt of statements) {
      // Subgraph
      const subMatch = stmt.match(/^subgraph\s+(.+)$/i);
      if (subMatch) {
        subgraphStack.push({ label: subMatch[1].trim(), children: [] });
        continue;
      }
      if (stmt.toLowerCase() === "end" && subgraphStack.length > 0) {
        const sg = subgraphStack.pop()!;
        const groupId = d.addGroup(sg.label, sg.children);
        // If nested, add group to parent subgraph
        if (subgraphStack.length > 0) {
          subgraphStack[subgraphStack.length - 1].children.push(groupId);
        }
        continue;
      }

      // Try edge parse — handle chains like A-->B-->C
      let remaining = stmt;
      let foundEdge = false;
      let prevId: string | undefined;

      while (true) {
        const edgeMatch = remaining.match(edgeRegex);
        if (!edgeMatch) break;
        foundEdge = true;
        const [, rawSrc, arrow, edgeLabel, rest] = edgeMatch;
        const srcDef = parseNodeDef(rawSrc.trim());
        const srcId = prevId ?? ensureNode(srcDef.name, srcDef.label, srcDef.shape);

        // The rest might be another chain: try to split off just the target node
        // Check if rest contains another arrow
        const nextArrowMatch = rest.match(/^([A-Za-z0-9_-]+(?:\[.*?\]|\{.*?\}|\(\(.*?\)\)|\[\(.*?\)\]|\(.*?\))?)\s*(==>|-->|---|-\.->)/);
        let rawTgt: string;
        if (nextArrowMatch) {
          rawTgt = nextArrowMatch[1];
          remaining = rest; // Continue parsing from rest
        } else {
          rawTgt = rest.trim();
          remaining = ""; // No more edges
        }

        const tgtDef = parseNodeDef(rawTgt.trim());
        const tgtId = ensureNode(tgtDef.name, tgtDef.label, tgtDef.shape);
        d.connect(srcId, tgtId, edgeLabel?.trim() || undefined, parseEdgeOpts(arrow));
        prevId = tgtId;

        if (!nextArrowMatch) break;
      }

      if (foundEdge) continue;

      // Standalone node definition (no edge)
      const nodeDef = parseNodeDef(stmt);
      if (nodeDef.name && /^[A-Za-z0-9_-]+/.test(nodeDef.name)) {
        ensureNode(nodeDef.name, nodeDef.label, nodeDef.shape);
      }
    }

    // Assign rows/cols based on direction and topological depth
    const depthMap = new Map<string, number>();
    const allNodeNames = [...nodeMap.keys()];

    // Build adjacency from edges
    const adj = new Map<string, string[]>();
    const idToName = new Map<string, string>();
    for (const [name, id] of nodeMap) idToName.set(id, name);
    for (const name of allNodeNames) adj.set(name, []);
    for (const edge of d.getEdges()) {
      const srcName = idToName.get(edge.from);
      const tgtName = idToName.get(edge.to);
      if (srcName && tgtName) {
        adj.get(srcName)!.push(tgtName);
      }
    }

    // BFS to compute depth
    const visited = new Set<string>();
    // Find roots (nodes with no incoming edges)
    const hasIncoming = new Set<string>();
    for (const [, targets] of adj) {
      for (const t of targets) hasIncoming.add(t);
    }
    const roots = allNodeNames.filter(n => !hasIncoming.has(n));
    if (roots.length === 0 && allNodeNames.length > 0) roots.push(allNodeNames[0]);

    const queue: { name: string; depth: number }[] = roots.map(n => ({ name: n, depth: 0 }));
    for (const r of roots) { visited.add(r); depthMap.set(r, 0); }

    while (queue.length > 0) {
      const { name, depth } = queue.shift()!;
      for (const neighbor of (adj.get(name) ?? [])) {
        const existingDepth = depthMap.get(neighbor) ?? -1;
        if (depth + 1 > existingDepth) {
          depthMap.set(neighbor, depth + 1);
        }
        if (!visited.has(neighbor)) {
          visited.add(neighbor);
          queue.push({ name: neighbor, depth: depth + 1 });
        }
      }
    }
    // Assign unvisited nodes depth 0
    for (const name of allNodeNames) {
      if (!depthMap.has(name)) depthMap.set(name, 0);
    }

    // Group by depth and assign row/col
    const byDepth = new Map<number, string[]>();
    for (const [name, depth] of depthMap) {
      if (!byDepth.has(depth)) byDepth.set(depth, []);
      byDepth.get(depth)!.push(name);
    }

    for (const [depth, names] of byDepth) {
      names.forEach((name, idx) => {
        const nodeId = nodeMap.get(name);
        if (!nodeId) return;
        const node = d._getNodes().get(nodeId);
        if (!node) return;
        if (direction === "LR" || direction === "RL") {
          node.col = depth;
          node.row = idx;
        } else {
          node.row = depth;
          node.col = idx;
        }
      });
    }

    d.setDirection(direction);
    return d;
  }

  /** Find node IDs by label match. Substring by default, exact with opts. */
  findByLabel(label: string, opts?: { exact?: boolean }): string[] {
    const lower = label.toLowerCase();
    const results: string[] = [];
    for (const node of this.nodes.values()) {
      const match = opts?.exact
        ? node.label.toLowerCase() === lower
        : node.label.toLowerCase().includes(lower);
      if (match) results.push(node.id);
    }
    return results;
  }

  /** Get all node IDs. */
  getNodes(): string[] {
    return Array.from(this.nodes.keys());
  }

  /** Get all edges. */
  getEdges(): Array<{ from: string; to: string; label?: string }> {
    return this.edges.map(e => ({ from: e.from, to: e.to, label: e.label }));
  }

  /** Get a node's properties by ID. Returns undefined if not found. */
  getNode(id: string): { label: string; type: string; width: number; height: number;
    backgroundColor?: string; strokeColor?: string; row?: number; col?: number } | undefined {
    const node = this.nodes.get(id);
    if (!node) return undefined;
    return {
      label: node.label,
      type: node.type,
      width: node.width,
      height: node.height,
      backgroundColor: node.opts?.backgroundColor ?? node.color.background,
      strokeColor: node.opts?.strokeColor ?? node.color.stroke,
      row: node.row,
      col: node.col,
    };
  }

  /** Update a node's properties. */
  updateNode(id: string, update: Partial<ShapeOpts> & { label?: string }): void {
    const node = this.nodes.get(id);
    if (!node) throw new Error(`Node not found: ${id}`);

    if (update.label !== undefined) node.label = update.label;
    if (update.width !== undefined) node.width = update.width;
    if (update.height !== undefined) node.height = update.height;
    if (update.x !== undefined) node.absX = update.x;
    if (update.y !== undefined) node.absY = update.y;
    if (update.row !== undefined) node.row = update.row;
    if (update.col !== undefined) node.col = update.col;

    // Update color
    if (update.color) {
      node.color = COLOR_PALETTE[update.color];
    }
    if (update.strokeColor) {
      node.color = { ...node.color, stroke: update.strokeColor };
    }
    if (update.backgroundColor) {
      node.color = { ...node.color, background: update.backgroundColor };
    }

    // Merge remaining opts (exclude non-ShapeOpts fields)
    const { label: _, ...shapeUpdates } = update;
    node.opts = { ...node.opts, ...shapeUpdates };
  }

  /** Remove a node and all its connected edges. */
  removeNode(id: string): void {
    this.nodes.delete(id);
    this.edges = this.edges.filter(e => e.from !== id && e.to !== id);
    // Also remove from groups and frames
    for (const group of this.groups.values()) {
      group.children = group.children.filter(c => c !== id);
    }
    for (const frame of this.frames.values()) {
      frame.children = frame.children.filter(c => c !== id);
    }
  }

  /** Remove an edge between two nodes. Optional label disambiguates multi-edges. */
  removeEdge(from: string, to: string, label?: string): void {
    let removed = false;
    this.edges = this.edges.filter(e => {
      if (e.from !== from || e.to !== to) return true;
      if (label !== undefined && e.label !== label) return true;
      if (removed) return true; // only remove first match
      removed = true;
      return false;
    });
  }

  /** Add an actor to a sequence diagram. Returns the actor ID. */
  addActor(label: string, opts?: ShapeOpts): string {
    const id = this.nextId("actor");
    this.sequenceActors.push({ id, label, index: this.sequenceActors.length, opts });
    return id;
  }

  /** Add a message between actors in a sequence diagram. */
  message(from: string, to: string, label?: string, opts?: ConnectOpts): void {
    const actorIds = new Set(this.sequenceActors.map(a => a.id));
    if (!actorIds.has(from)) throw new Error(`Actor not found: "${from}". Add actors with addActor() before sending messages.`);
    if (!actorIds.has(to)) throw new Error(`Actor not found: "${to}". Add actors with addActor() before sending messages.`);
    this.sequenceMessages.push({ from, to, label, index: this.sequenceMessages.length, opts });
  }

  /** Update an existing edge's properties. Optional matchLabel disambiguates multi-edges. */
  updateEdge(from: string, to: string, update: Partial<ConnectOpts> & { label?: string }, matchLabel?: string): void {
    const edge = this.edges.find(e =>
      e.from === from && e.to === to &&
      (matchLabel === undefined || e.label === matchLabel),
    );
    if (!edge) throw new Error(`Edge not found: ${from} -> ${to}`);
    if (update.label !== undefined) edge.label = update.label;
    if (update.style !== undefined) edge.style = update.style;
    const { label: _, style: __, ...connectUpdates } = update;
    edge.opts = { ...edge.opts, ...connectUpdates };
  }

  /** Render the diagram to the specified format. */
  async render(opts?: RenderOpts): Promise<RenderResult> {
    const warnings: string[] = [];
    const elements = await this.buildElements(warnings);

    // WASM validation: log warnings to stderr if available
    if (isWasmLoaded()) {
      const errorsJson = await validateElements(JSON.stringify(elements));
      if (errorsJson) {
        try {
          const errors = JSON.parse(errorsJson);
          if (Array.isArray(errors) && errors.length > 0) {
            for (const err of errors) {
              const msg = `validation: ${err.msg} (${err.id})`;
              warnings.push(msg);
              if (typeof process !== "undefined") {
                process.stderr.write(`[drawmode] ${msg}\n`);
              }
            }
          }
        } catch (e) {
          if (typeof process !== "undefined") process.stderr.write(`[drawmode] validation parse error: ${e}\n`);
        }
      }
    }

    const excalidrawJson: ExcalidrawFile = {
      type: "excalidraw" as const,
      version: EXCALIDRAW_VERSION,
      source: "drawmode",
      elements,
      appState: { gridSize: null, viewBackgroundColor: "#ffffff" },
      files: {},
    };

    const result: RenderResult = { json: excalidrawJson };
    if (warnings.length > 0) result.warnings = warnings;

    // Compute diagram stats
    result.stats = {
      nodes: this.nodes.size,
      edges: this.edges.length,
      groups: this.groups.size,
    };

    const formatInput = opts?.format ?? "excalidraw";
    const formats: OutputFormat[] = Array.isArray(formatInput) ? formatInput : [formatInput];
    const filePaths: string[] = [];

    // Import fs once if any format writes to disk
    const needsFs = formats.includes("excalidraw") || formats.includes("png") || formats.includes("svg");
    const fs = needsFs ? await import("node:fs/promises") : null;

    for (const format of formats) {
      if (format === "excalidraw") {
        const path = opts?.path ?? "diagram.excalidraw";

        // Compute diff against existing file (if any)
        try {
          await fs!.access(path);
          const oldContent = await fs!.readFile(path, "utf-8");
          const oldFile = ExcalidrawFileSchema.parse(JSON.parse(oldContent));
          const summary = computeChangeSummary(oldFile.elements, elements);
          if (summary) result.changeSummary = summary;
        } catch { /* file doesn't exist or is unparseable — skip diff */ }

        await fs!.writeFile(path, JSON.stringify(excalidrawJson, null, 2));
        if (!result.filePath) result.filePath = path;
        filePaths.push(path);
      }

      if (format === "url") {
        const { uploadToExcalidraw } = await import("./upload.js");
        result.url = await uploadToExcalidraw(JSON.stringify(excalidrawJson));
      }

      if (format === "png") {
        const pngPath = (opts?.path ?? "diagram").replace(/\.(excalidraw|png|svg)$/, "") + ".png";
        let pngData: string | null = null;

        const { renderPngWasm } = await import("./png.js");
        const wasmResult = await renderPngWasm(elements);
        if (wasmResult) {
          pngData = wasmResult.pngBase64;
          if (fs) await fs.writeFile(pngPath, wasmResult.pngBytes);
          result.pngBase64 = pngData;
          if (!result.filePath) result.filePath = pngPath;
          filePaths.push(pngPath);
        } else {
          throw new Error("PNG export failed. Ensure WASM and linkedom are available.");
        }
      }

      if (format === "svg") {
        const svgPath = (opts?.path ?? "diagram").replace(/\.(excalidraw|png|svg)$/, "") + ".svg";
        let svgData: string | null = null;

        const { renderSvgWasm } = await import("./png.js");
        svgData = await renderSvgWasm(elements);
        if (svgData) {
          if (fs) await fs.writeFile(svgPath, svgData, "utf-8");
          result.svgString = svgData;
          if (!result.filePath) result.filePath = svgPath;
          filePaths.push(svgPath);
        } else {
          throw new Error("SVG export failed. Ensure linkedom is available.");
        }
      }
    }

    // Write sidecar .drawmode.ts for any format that writes to disk
    // Wrapped in try/catch — sidecar is a convenience, should never fail the render
    if (fs && filePaths.length > 0) {
      try {
        const basePath = (opts?.path ?? "diagram").replace(/\.(excalidraw|png|svg)$/, "");
        const sidecarPath = basePath + ".drawmode.ts";
        const sidecarCode = this.toCode({ path: opts?.path });
        if (sidecarCode) {
          await fs.writeFile(sidecarPath, sidecarCode);
        }
      } catch { /* sidecar write is non-critical */ }
    }

    if (filePaths.length > 1) result.filePaths = filePaths;

    return result;
  }

  /** Convert the graph to Excalidraw elements with layout. */
  private async buildElements(warnings?: string[]): Promise<ExcalidrawElement[]> {
    if (this.diagramType === "sequence") {
      return this.buildSequenceElements();
    }

    const elements: ExcalidrawElement[] = [];
    const { positioned, edgeRoutes, groupBounds } = await this.layoutNodes(warnings);

    // Nudge text nodes that overlap with box nodes
    resolveTextBoxOverlaps(positioned);

    // Nudge non-member nodes out of group bounding boxes when Graphviz clusters
    // are skipped (e.g. due to row constraints). Graphviz clusters handle
    // containment properly, but without clusters nodes may overlap group boxes.
    if (!groupBounds && this.groups.size > 0) {
      for (const [_groupId, group] of this.groups) {
        const childNodes = group.children
          .map(c => positioned.get(c))
          .filter((n): n is PositionedNode => n !== undefined);
        if (childNodes.length === 0) continue;

        const padding = group.opts?.padding ?? 30;
        const allBounds = childNodes.map(n => ({ x: n.x ?? 0, y: n.y ?? 0, width: n.width, height: n.height }));
        const { minX, minY, maxX, maxY } = computeNodeBounds(allBounds);
        const gx = minX - padding;
        const gy = minY - padding - 20;
        const gw = (maxX + padding) - gx;
        const gh = (maxY + padding) - gy;

        // Check every non-member node
        const childSet = new Set(group.children);
        for (const [nodeId, node] of positioned) {
          if (childSet.has(nodeId)) continue;
          const nx = node.x ?? 0, ny = node.y ?? 0;
          if (!rectsOverlap(gx, gy, gw, gh, nx, ny, node.width, node.height, 5)) continue;

          // Find smallest displacement to push node outside group bbox
          const pushLeft = (gx - 5) - (nx + node.width);
          const pushRight = (gx + gw + 5) - nx;
          const pushUp = (gy - 5) - (ny + node.height);
          const pushDown = (gy + gh + 5) - ny;

          // Pick smallest absolute displacement
          const candidates = [
            { dx: pushLeft, dy: 0 },
            { dx: pushRight, dy: 0 },
            { dx: 0, dy: pushUp },
            { dx: 0, dy: pushDown },
          ];
          candidates.sort((a, b) => (Math.abs(a.dx) + Math.abs(a.dy)) - (Math.abs(b.dx) + Math.abs(b.dy)));
          node.x = nx + candidates[0].dx;
          node.y = ny + candidates[0].dy;
        }
      }
    }

    // Pre-compute arrow IDs (one per edge, in order)
    const arrowIds: string[] = [];
    for (let i = 0; i < this.edges.length; i++) {
      arrowIds.push(this.nextId("arr"));
    }

    // Create shape + bound text for each node
    for (const node of positioned.values()) {
      const o = node.opts;

      // Standalone text node
      if (node.type === "text") {
        const ff: FontFamily = o?.fontFamily ?? 1;
        elements.push({
          id: node.id,
          type: "text",
          x: node.x!, y: node.y!,
          width: node.width,
          height: node.height,
          text: node.label,
          fontSize: o?.fontSize ?? 16,
          fontFamily: ff,
          lineHeight: getLineHeight(ff),
          textAlign: o?.textAlign ?? "left",
          verticalAlign: o?.verticalAlign ?? "top",
          containerId: null,
          originalText: node.label,
          autoResize: true,
          strokeColor: node.color.stroke,
          backgroundColor: "transparent",
          fillStyle: "solid",
          strokeWidth: 1,
          roughness: 0,
          opacity: o?.opacity ?? 100,
          angle: 0,
          groupIds: [],
          frameId: null,
          isDeleted: false,
          boundElements: null,
          updated: Date.now(),
          locked: false,
          link: null,
          seed: randSeed(),
          version: 1,
          versionNonce: randSeed(),
        });
        continue;
      }

      // Line element
      if (node.type === "line") {
        elements.push({
          id: node.id,
          type: "line",
          x: node.x!, y: node.y!,
          width: node.width,
          height: node.height,
          points: node.linePoints ?? [[0, 0], [node.width, 0]],
          strokeColor: node.color.stroke,
          backgroundColor: "transparent",
          fillStyle: "solid",
          strokeWidth: o?.strokeWidth ?? 2,
          strokeStyle: o?.strokeStyle ?? "solid",
          roughness: o?.roughness ?? 1,
          opacity: o?.opacity ?? 100,
          angle: 0,
          roundness: null,
          startBinding: null,
          endBinding: null,
          startArrowhead: null,
          endArrowhead: null,
          groupIds: [],
          frameId: null,
          isDeleted: false,
          boundElements: null,
          updated: Date.now(),
          locked: false,
          link: null,
          seed: randSeed(),
          version: 1,
          versionNonce: randSeed(),
        });
        continue;
      }

      // Rectangle / Ellipse
      const textId = `${node.id}-text`;

      elements.push({
        id: node.id,
        type: node.type,
        x: node.x!, y: node.y!,
        width: node.width, height: node.height,
        backgroundColor: o?.backgroundColor ?? node.color.background,
        strokeColor: o?.strokeColor ?? node.color.stroke,
        fillStyle: o?.fillStyle ?? "solid",
        strokeWidth: o?.strokeWidth ?? 2,
        roughness: o?.roughness ?? 1,
        opacity: o?.opacity ?? 100,
        angle: 0,
        strokeStyle: o?.strokeStyle ?? "solid",
        roundness: o?.roundness !== undefined ? o.roundness : { type: 3 },
        boundElements: [
          { type: "text", id: textId },
        ],
        groupIds: [],
        frameId: null,
        isDeleted: false,
        updated: Date.now(),
        locked: false,
        link: o?.link ?? null,
        seed: randSeed(),
        version: 1,
        versionNonce: randSeed(),
        ...(o?.customData !== undefined ? { customData: o.customData } : {}),
      });

      const textWidth = node.width - 20;
      const boundFf: FontFamily = o?.fontFamily ?? 1;
      const textMeasured = measureText(node.label, o?.fontSize ?? 16, boundFf);
      const textHeight = Math.max(20, textMeasured.height);

      elements.push({
        id: textId,
        type: "text",
        x: node.x! + (node.width - textWidth) / 2,
        y: node.y! + (node.height - textHeight) / 2,
        width: textWidth,
        height: textHeight,
        text: node.label,
        fontSize: o?.fontSize ?? 16,
        fontFamily: boundFf,
        lineHeight: getLineHeight(boundFf),
        textAlign: o?.textAlign ?? "center",
        verticalAlign: o?.verticalAlign ?? "middle",
        containerId: node.id,
        originalText: node.label,
        autoResize: true,
        strokeColor: o?.strokeColor ?? node.color.stroke,
        backgroundColor: "transparent",
        fillStyle: "solid",
        strokeWidth: 1,
        roughness: 0,
        opacity: o?.opacity ?? 100,
        angle: 0,
        groupIds: [],
        frameId: null,
        isDeleted: false,
        boundElements: null,
        updated: Date.now(),
        locked: false,
        link: null,
        seed: randSeed(),
        version: 1,
        versionNonce: randSeed(),
      });
    }

    // Create arrows for edges
    const edgePairCounts = new Map<string, number>(); // track multi-edges between same nodes

    // Count outgoing edges per source node for staggering start points
    const outgoingCount = new Map<string, number>();
    const outgoingIdx = new Map<string, number>();
    for (const edge of this.edges) {
      outgoingCount.set(edge.from, (outgoingCount.get(edge.from) ?? 0) + 1);
    }

    for (let ei = 0; ei < this.edges.length; ei++) {
      const edge = this.edges[ei];
      const fromNode = positioned.get(edge.from);
      const toNode = positioned.get(edge.to);
      if (!fromNode || !toNode) continue;

      const co = edge.opts;
      const arrowId = arrowIds[ei];

      // Check for WASM-computed edge route (with index for multi-edges)
      const baseRouteKey = `${edge.from}->${edge.to}`;
      const edgePairIdx = edgePairCounts.get(baseRouteKey) ?? 0;
      edgePairCounts.set(baseRouteKey, edgePairIdx + 1);
      const routeKey = edgePairIdx === 0 ? baseRouteKey : `${baseRouteKey}#${edgePairIdx}`;
      const edgeRoute = edgeRoutes?.get(routeKey);
      const hasEdgeRoute = edgeRoute && edgeRoute.points.length >= 2;

      // Stagger start points when multiple arrows leave the same source node edge
      const srcOutCount = outgoingCount.get(edge.from) ?? 1;
      const srcOutIdx = outgoingIdx.get(edge.from) ?? 0;
      outgoingIdx.set(edge.from, srcOutIdx + 1);

      // Always elbowed=false — arrows are static polylines (no bindings).
      // Excalidraw renders our exact points without recalculating on interaction.
      const isElbowed = false;

      let arrowX: number, arrowY: number;
      let points: number[][];

      // Compute stagger offset for multiple arrows leaving same source node edge.
      // Distributes start points at 20%, 35%, 50%, 65%, 80% along the edge.
      const staggerPositions = [0.5, 0.2, 0.8, 0.35, 0.65];
      let staggerOffsetX = 0;
      let staggerOffsetY = 0;
      if (srcOutCount > 1) {
        const dirIsHoriz = this.direction === "LR" || this.direction === "RL";
        const pos = staggerPositions[srcOutIdx % staggerPositions.length];
        if (dirIsHoriz) {
          staggerOffsetY = Math.round((pos - 0.5) * fromNode.height * 0.6);
        } else {
          staggerOffsetX = Math.round((pos - 0.5) * fromNode.width * 0.6);
        }
      }

      if (hasEdgeRoute) {
        // Use Graphviz-computed route (orthogonal spline points)
        // Stagger: shift start along source edge to avoid overlapping arrows
        const baseX = edgeRoute.points[0][0];
        const baseY = edgeRoute.points[0][1];
        arrowX = baseX + staggerOffsetX;
        arrowY = baseY + staggerOffsetY;
        // points[0] must be [0,0]; all others relative to staggered origin
        points = [[0, 0]];
        for (let i = 1; i < edgeRoute.points.length; i++) {
          points.push([edgeRoute.points[i][0] - arrowX, edgeRoute.points[i][1] - arrowY]);
        }
      } else {
        // Fallback: orthogonal route from source edge to target edge
        const dirIsHorizontal = this.direction === "LR" || this.direction === "RL";
        if (dirIsHorizontal) {
          // LR: right edge of source → left edge of target
          const fx = (fromNode.x ?? 0) + fromNode.width;
          const fy = (fromNode.y ?? 0) + fromNode.height / 2 + staggerOffsetY;
          const tx = (toNode.x ?? 0);
          const ty = (toNode.y ?? 0) + toNode.height / 2;
          arrowX = fx;
          arrowY = fy;
          if (Math.abs(fy - ty) < 1) {
            points = [[0, 0], [tx - fx, 0]];
          } else {
            const midX = Math.round((fx + tx) / 2);
            points = [[0, 0], [midX - fx, 0], [midX - fx, ty - fy], [tx - fx, ty - fy]];
          }
        } else {
          // TB: bottom edge of source → top edge of target
          const fx = (fromNode.x ?? 0) + fromNode.width / 2 + staggerOffsetX;
          const fy = (fromNode.y ?? 0) + fromNode.height;
          const tx = (toNode.x ?? 0) + toNode.width / 2;
          const ty = (toNode.y ?? 0);
          arrowX = fx;
          arrowY = fy;
          if (Math.abs(fx - tx) < 1) {
            points = [[0, 0], [0, ty - fy]];
          } else {
            const midY = Math.round((fy + ty) / 2);
            points = [[0, 0], [0, midY - fy], [tx - fx, midY - fy], [tx - fx, ty - fy]];
          }
        }
      }

      const { minX: bMinX, minY: bMinY, maxX: bMaxX, maxY: bMaxY } = computeBounds(points);
      const boundsWidth = bMaxX - bMinX;
      const boundsHeight = bMaxY - bMinY;

      const labelTextId = edge.label ? this.nextId("arrlbl") : undefined;

      elements.push({
        id: arrowId,
        type: "arrow",
        x: arrowX,
        y: arrowY,
        width: boundsWidth || 1,
        height: boundsHeight || 1,
        points,
        strokeColor: co?.strokeColor ?? toNode.color.stroke,
        backgroundColor: "transparent",
        fillStyle: "solid",
        strokeWidth: co?.strokeWidth ?? 2,
        strokeStyle: edge.style,
        roughness: co?.roughness ?? 0,
        roundness: null,
        elbowed: isElbowed,
        opacity: co?.opacity ?? 100,
        angle: 0,
        startBinding: null,
        endBinding: null,
        startArrowhead: co?.startArrowhead ?? null,
        endArrowhead: co?.endArrowhead !== undefined ? co.endArrowhead : "arrow",
        groupIds: [],
        frameId: null,
        isDeleted: false,
        boundElements: null,
        updated: Date.now(),
        locked: false,
        link: null,
        seed: randSeed(),
        version: 1,
        versionNonce: randSeed(),
        customData: { ...(co?.customData ?? {}), _from: edge.from, _to: edge.to, ...(labelTextId ? { _labelId: labelTextId } : {}) },
      });

      // Arrow label placement
      if (edge.label && labelTextId) {
        const labelWidth = measureText(edge.label, co?.labelFontSize ?? 14, 1).width + 16;
        let labelX: number, labelY: number;
        const labelPos = co?.labelPosition ?? "middle";

        if (edgeRoute?.labelPos && labelPos === "middle") {
          // Use Zig-computed label position (center-based, with collision avoidance)
          labelX = edgeRoute.labelPos.x - labelWidth / 2;
          labelY = edgeRoute.labelPos.y - 12;
        } else if (points.length >= 2) {
          // Pick segment based on labelPosition
          let targetSeg: number;
          if (labelPos === "start") {
            targetSeg = 0;
          } else if (labelPos === "end") {
            targetSeg = points.length - 2;
          } else {
            // "middle": midpoint of longest segment
            let bestLen = 0;
            targetSeg = 0;
            for (let s = 0; s < points.length - 1; s++) {
              const segDx = points[s + 1][0] - points[s][0];
              const segDy = points[s + 1][1] - points[s][1];
              const segLen = Math.abs(segDx) + Math.abs(segDy);
              if (segLen > bestLen) { bestLen = segLen; targetSeg = s; }
            }
          }
          const p1 = points[targetSeg], p2 = points[targetSeg + 1];
          labelX = arrowX + (p1[0] + p2[0]) / 2;
          labelY = arrowY + (p1[1] + p2[1]) / 2 - 12;
        } else {
          labelX = arrowX;
          labelY = arrowY - 12;
        }

        const labelFf: FontFamily = 1;
        elements.push({
          id: labelTextId,
          type: "text",
          x: labelX,
          y: labelY,
          width: labelWidth,
          height: 20,
          text: edge.label,
          fontSize: co?.labelFontSize ?? 14,
          fontFamily: labelFf,
          lineHeight: getLineHeight(labelFf),
          textAlign: "center",
          verticalAlign: "middle",
          containerId: null,
          originalText: edge.label,
          autoResize: true,
          strokeColor: "#1e1e1e",
          backgroundColor: "transparent",
          fillStyle: "solid",
          strokeWidth: 1,
          roughness: 0,
          opacity: co?.opacity ?? 100,
          angle: 0,
          groupIds: [],
          frameId: null,
          isDeleted: false,
          boundElements: null,
          updated: Date.now(),
          locked: false,
          link: null,
          seed: randSeed(),
          version: 1,
          versionNonce: randSeed(),
        });
      }
    }

    // ── Unified overlap resolver ──
    // Iterates until stable or budget exhausted. Arrow labels are moveable;
    // they check against nodes, groups, other labels, and arrow segments.
    resolveOverlaps(elements);

    // Groups: dashed rectangle around children + label
    // Use Graphviz cluster bounding boxes when available (guaranteed non-overlapping)
    const groupBoundsMap = new Map<string, GroupBounds>();
    if (groupBounds) {
      for (const gb of groupBounds) groupBoundsMap.set(gb.id, gb);
    }

    // Render groups in dependency order: child groups before parents so parent bounds include children
    const groupOrder = this.topologicalGroupOrder();

    for (const groupId of groupOrder) {
      const group = this.groups.get(groupId)!;
      let gx: number, gy: number, gw: number, gh: number;

      const gb = groupBoundsMap.get(groupId);
      if (gb) {
        // Use Graphviz-computed cluster bounding box (non-overlapping)
        gx = gb.x; gy = gb.y; gw = gb.width; gh = gb.height;
      } else {
        // Fallback: compute from child node positions + child group bounds
        const childNodes = group.children
          .map(c => positioned.get(c))
          .filter((n): n is PositionedNode => n !== undefined);
        // Include child groups that have already been computed
        const childGroupRects = group.children
          .filter(c => groupBoundsMap.has(c))
          .map(c => {
            const cgb = groupBoundsMap.get(c)!;
            return { x: cgb.x, y: cgb.y, width: cgb.width, height: cgb.height };
          });
        const allBounds = [
          ...childNodes.map(n => ({ x: n.x ?? 0, y: n.y ?? 0, width: n.width, height: n.height })),
          ...childGroupRects,
        ];
        if (allBounds.length === 0) continue;

        const padding = group.opts?.padding ?? 30;
        const { minX, minY, maxX, maxY } = computeNodeBounds(allBounds);
        gx = minX - padding;
        gy = minY - padding - 20;
        gw = (maxX + padding) - gx;
        gh = (maxY + padding) - gy;
      }

      // Store computed bounds so parent groups can reference them
      groupBoundsMap.set(groupId, { id: groupId, x: gx, y: gy, width: gw, height: gh });

      const grpStrokeColor = group.opts?.strokeColor ?? "#868e96";
      const grpStrokeStyle = group.opts?.strokeStyle ?? "dashed";
      const grpOpacity = group.opts?.opacity ?? 60;

      elements.push({
        id: groupId,
        type: "rectangle",
        x: gx, y: gy,
        width: gw, height: gh,
        backgroundColor: "transparent",
        strokeColor: grpStrokeColor,
        fillStyle: "solid",
        strokeWidth: 1,
        strokeStyle: grpStrokeStyle,
        roughness: 0,
        opacity: grpOpacity,
        angle: 0,
        roundness: { type: 3 },
        boundElements: null,
        groupIds: [],
        frameId: null,
        isDeleted: false,
        updated: Date.now(),
        locked: false,
        link: null,
        seed: randSeed(),
        version: 1,
        versionNonce: randSeed(),
      });

      elements.push({
        id: `${groupId}-label`,
        type: "text",
        x: gx + 10, y: gy + 5,
        width: measureText(group.label, 14, 1).width + 16, height: 20,
        text: group.label,
        fontSize: 14,
        fontFamily: 1,
        lineHeight: 1.25,
        textAlign: "left",
        verticalAlign: "top",
        containerId: null,
        originalText: group.label,
        autoResize: true,
        strokeColor: grpStrokeColor,
        backgroundColor: "transparent",
        fillStyle: "solid",
        strokeWidth: 1,
        roughness: 0,
        opacity: grpOpacity,
        angle: 0,
        groupIds: [],
        frameId: null,
        isDeleted: false,
        boundElements: null,
        updated: Date.now(),
        locked: false,
        link: null,
        seed: randSeed(),
        version: 1,
        versionNonce: randSeed(),
      });
    }

    // Frames: native Excalidraw frame containers
    for (const [frameId, frame] of this.frames) {
      const childNodes = frame.children
        .map(c => positioned.get(c))
        .filter((n): n is PositionedNode => n !== undefined);
      if (childNodes.length === 0) continue;

      const padding = 30;
      let { minX, minY, maxX, maxY } = computeNodeBounds(childNodes);
      minX -= padding;
      minY -= padding + 20;
      maxX += padding;
      maxY += padding;

      elements.push({
        id: frameId,
        type: "frame",
        x: minX, y: minY,
        width: maxX - minX, height: maxY - minY,
        name: frame.name,
        backgroundColor: "transparent",
        strokeColor: "#bbb",
        fillStyle: "solid",
        strokeWidth: 1,
        strokeStyle: "solid",
        roughness: 0,
        opacity: 100,
        angle: 0,
        roundness: null,
        boundElements: null,
        groupIds: [],
        frameId: null,
        isDeleted: false,
        updated: Date.now(),
        locked: false,
        link: null,
        seed: randSeed(),
        version: 1,
        versionNonce: randSeed(),
      });

      // Set frameId on child elements
      const frameChildSet = new Set(frame.children);
      for (const el of elements) {
        if (frameChildSet.has(el.id)) {
          el.frameId = frameId;
        }
        // Also tag bound text elements
        if (el.containerId && frameChildSet.has(el.containerId)) {
          el.frameId = frameId;
        }
      }
    }

    // Append passthrough elements
    elements.push(...this.passthrough);

    return elements;
  }

  /** Build elements for a sequence diagram (formula layout, no Graphviz). */
  private buildSequenceElements(): ExcalidrawElement[] {
    const SEQ_ACTOR_SPACING = 220;
    const SEQ_ACTOR_WIDTH = 160;
    const SEQ_ACTOR_HEIGHT = 60;
    const SEQ_MSG_START_Y = 200;
    const SEQ_MSG_SPACING = 80;
    const LIFELINE_EXTRA = 10; // extra space below last message

    const elements: ExcalidrawElement[] = [];

    // Build actor position map
    const actorX = new Map<string, number>();
    for (const actor of this.sequenceActors) {
      const x = BASE_X + actor.index * SEQ_ACTOR_SPACING;
      actorX.set(actor.id, x);

      // Actor box
      const textId = `${actor.id}-text`;
      elements.push({
        id: actor.id,
        type: "rectangle",
        x, y: BASE_Y,
        width: SEQ_ACTOR_WIDTH, height: SEQ_ACTOR_HEIGHT,
        backgroundColor: actor.opts?.backgroundColor ?? COLOR_PALETTE[actor.opts?.color ?? "backend"].background,
        strokeColor: actor.opts?.strokeColor ?? COLOR_PALETTE[actor.opts?.color ?? "backend"].stroke,
        fillStyle: "solid", strokeWidth: 2, roughness: 1, opacity: 100,
        angle: 0, strokeStyle: "solid",
        roundness: { type: 3 },
        boundElements: [{ type: "text", id: textId }],
        groupIds: [], frameId: null, isDeleted: false,
        updated: Date.now(), locked: false, link: null,
        seed: randSeed(), version: 1, versionNonce: randSeed(),
      });

      // Actor label
      const ff: FontFamily = 1;
      const measured = measureText(actor.label, 16, ff);
      elements.push({
        id: textId,
        type: "text",
        x: x + (SEQ_ACTOR_WIDTH - measured.width) / 2,
        y: BASE_Y + (SEQ_ACTOR_HEIGHT - measured.height) / 2,
        width: measured.width, height: measured.height,
        text: actor.label, fontSize: 16, fontFamily: ff,
        lineHeight: getLineHeight(ff), textAlign: "center" as TextAlign,
        verticalAlign: "middle" as VerticalAlign,
        containerId: actor.id, originalText: actor.label, autoResize: true,
        strokeColor: COLOR_PALETTE[actor.opts?.color ?? "backend"].stroke,
        backgroundColor: "transparent", fillStyle: "solid",
        strokeWidth: 1, roughness: 0, opacity: 100, angle: 0,
        groupIds: [], frameId: null, isDeleted: false,
        boundElements: null, updated: Date.now(), locked: false, link: null,
        seed: randSeed(), version: 1, versionNonce: randSeed(),
      });
    }

    // Lifeline length depends on message count
    const lifelineEndY = SEQ_MSG_START_Y + (this.sequenceMessages.length - 1) * SEQ_MSG_SPACING + SEQ_MSG_SPACING / 2 + LIFELINE_EXTRA;

    // Dashed lifelines
    for (const actor of this.sequenceActors) {
      const x = actorX.get(actor.id)!;
      const centerX = x + SEQ_ACTOR_WIDTH / 2;
      const startY = BASE_Y + SEQ_ACTOR_HEIGHT;
      const lineId = `${actor.id}-lifeline`;
      elements.push({
        id: lineId,
        type: "line",
        x: centerX, y: startY,
        width: 0, height: lifelineEndY - startY,
        points: [[0, 0], [0, lifelineEndY - startY]],
        strokeColor: "#868e96",
        backgroundColor: "transparent", fillStyle: "solid",
        strokeWidth: 2, strokeStyle: "dashed", roughness: 0, opacity: 100,
        angle: 0, roundness: null,
        startBinding: null, endBinding: null,
        startArrowhead: null, endArrowhead: null,
        groupIds: [], frameId: null, isDeleted: false,
        boundElements: null, updated: Date.now(), locked: false, link: null,
        seed: randSeed(), version: 1, versionNonce: randSeed(),
      });
    }

    // Message arrows
    for (const msg of this.sequenceMessages) {
      const y = SEQ_MSG_START_Y + msg.index * SEQ_MSG_SPACING;
      const fromX = actorX.get(msg.from);
      const toX = actorX.get(msg.to);
      if (fromX === undefined || toX === undefined) continue;

      const fromCenterX = fromX + SEQ_ACTOR_WIDTH / 2;
      const toCenterX = toX + SEQ_ACTOR_WIDTH / 2;
      const arrowId = this.nextId("seq_arr");

      const isSelf = msg.from === msg.to;
      const style: StrokeStyle = msg.opts?.style ?? "solid";
      const endArrowhead: Arrowhead = msg.opts?.endArrowhead ?? "arrow";
      const startArrowhead: Arrowhead = msg.opts?.startArrowhead ?? null;

      if (isSelf) {
        // Self-message: rectangular loop on the right side
        const loopWidth = 40;
        const loopHeight = 30;
        elements.push({
          id: arrowId,
          type: "arrow",
          x: fromCenterX, y,
          width: loopWidth, height: loopHeight,
          points: [[0, 0], [loopWidth, 0], [loopWidth, loopHeight], [0, loopHeight]],
          strokeColor: msg.opts?.strokeColor ?? "#1e1e1e",
          backgroundColor: "transparent", fillStyle: "solid",
          strokeWidth: msg.opts?.strokeWidth ?? 2,
          strokeStyle: style, roughness: 0, opacity: msg.opts?.opacity ?? 100,
          angle: 0, roundness: null,
          startBinding: null, endBinding: null,
          startArrowhead, endArrowhead,
          elbowed: false,
          groupIds: [], frameId: null, isDeleted: false,
          boundElements: null, updated: Date.now(), locked: false, link: null,
          seed: randSeed(), version: 1, versionNonce: randSeed(),
        });
      } else {
        const dx = toCenterX - fromCenterX;
        elements.push({
          id: arrowId,
          type: "arrow",
          x: fromCenterX, y,
          width: Math.abs(dx), height: 0,
          points: [[0, 0], [dx, 0]],
          strokeColor: msg.opts?.strokeColor ?? "#1e1e1e",
          backgroundColor: "transparent", fillStyle: "solid",
          strokeWidth: msg.opts?.strokeWidth ?? 2,
          strokeStyle: style, roughness: 0, opacity: msg.opts?.opacity ?? 100,
          angle: 0, roundness: null,
          startBinding: null, endBinding: null,
          startArrowhead, endArrowhead,
          elbowed: false,
          groupIds: [], frameId: null, isDeleted: false,
          boundElements: null, updated: Date.now(), locked: false, link: null,
          seed: randSeed(), version: 1, versionNonce: randSeed(),
        });
      }

      // Message label as free-standing text
      if (msg.label) {
        const labelId = `${arrowId}-label`;
        const ff: FontFamily = 1;
        const labelSize = measureText(msg.label, 14, ff);
        let labelX: number;
        let labelY: number;

        if (isSelf) {
          labelX = fromCenterX + 45;
          labelY = y + 2;
        } else {
          labelX = (fromCenterX + toCenterX) / 2 - labelSize.width / 2;
          labelY = y - labelSize.height - 4;
        }

        elements.push({
          id: labelId,
          type: "text",
          x: labelX, y: labelY,
          width: labelSize.width, height: labelSize.height,
          text: msg.label, fontSize: 14, fontFamily: ff,
          lineHeight: getLineHeight(ff), textAlign: "center" as TextAlign,
          verticalAlign: "top" as VerticalAlign,
          containerId: null, originalText: msg.label, autoResize: true,
          strokeColor: "#1e1e1e", backgroundColor: "transparent",
          fillStyle: "solid", strokeWidth: 1, roughness: 0, opacity: 100,
          angle: 0, groupIds: [], frameId: null, isDeleted: false,
          boundElements: null, updated: Date.now(), locked: false, link: null,
          seed: randSeed(), version: 1, versionNonce: randSeed(),
        });
      }
    }

    return elements;
  }

  /** Assign x,y positions to all nodes via WASM Graphviz layout. */
  private async layoutNodes(_warnings?: string[]): Promise<{ positioned: Map<string, PositionedNode>; edgeRoutes?: Map<string, EdgeRoute>; groupBounds?: GroupBounds[] }> {
    const wasmResult = await this.layoutNodesWasm();
    if (wasmResult) return wasmResult;
    // No shape nodes to lay out (text-only / line-only diagrams) — return empty positions
    const hasShapes = Array.from(this.nodes.values()).some(
      n => n.type === "rectangle" || n.type === "ellipse" || n.type === "diamond",
    );
    if (!hasShapes) return { positioned: this.applyPositions([]) };
    throw new Error("Graphviz WASM layout failed. Ensure WASM module is built and loaded.");
  }

  private async layoutNodesWasm(): Promise<{ positioned: Map<string, PositionedNode>; edgeRoutes: Map<string, EdgeRoute>; groupBounds?: GroupBounds[] } | null> {
    // Auto-load WASM on first use (ensures Graphviz layout works for direct SDK imports)
    if (!isWasmLoaded()) await loadWasm();
    const graphNodes = Array.from(this.nodes.values()).filter(
      n => n.type === "rectangle" || n.type === "ellipse" || n.type === "diamond",
    );
    if (graphNodes.length === 0) return null;

    // When ALL shape nodes have absolute x/y, use neato nop2 (edge routing only,
    // no position computation). dot hangs on dense pre-positioned graphs.
    const allAbsolute = graphNodes.every(n => n.absX !== undefined && n.absY !== undefined);

    // For LR/RL directions, swap row/col so Graphviz interprets rank constraints correctly.
    // Graphviz Sugiyama always treats "rank" as the primary axis (TB=vertical, LR=horizontal).
    // User-facing row/col are in TB space, so we swap them for LR/RL before passing to Graphviz.
    const isHorizontal = this.direction === "LR" || this.direction === "RL";
    const wasmNodes = graphNodes.map(n => ({
      id: n.id, width: n.width, height: n.height,
      row: isHorizontal ? n.col : n.row,
      col: isHorizontal ? n.row : n.col,
      absX: n.absX, absY: n.absY,
      type: n.type,
    }));
    const wasmEdges = this.edges.map(e => ({ from: e.from, to: e.to, label: e.label }));
    // Resolve nested groups: detect which children are group IDs and set parent relationships
    const groupIds = new Set(this.groups.keys());
    // Pre-compute child→parent map for O(1) parent lookup
    const childToParent = new Map<string, string>();
    for (const [pid, pg] of this.groups) {
      for (const child of pg.children) {
        if (groupIds.has(child)) childToParent.set(child, pid);
      }
    }
    const wasmGroups = Array.from(this.groups.entries()).map(([id, g]) => {
      // Separate node children from group children
      const nodeChildren = g.children.filter(c => !groupIds.has(c));
      return { id, label: g.label, children: nodeChildren, parent: childToParent.get(id) };
    });

    // When row/col constraints are present, skip Graphviz clusters (groups).
    // Graphviz cluster processing overrides rank=same constraints, breaking row alignment.
    // The SDK computes group bounding boxes from positioned nodes instead.
    const hasRowConstraints = wasmNodes.some(n => n.row !== undefined);
    const effectiveGroups = hasRowConstraints ? undefined : (wasmGroups.length > 0 ? wasmGroups : undefined);

    const result = await layoutGraphWasm(wasmNodes, wasmEdges, effectiveGroups, { rankdir: this.direction, engine: allAbsolute ? "nop2" : "dot" });
    if (!result) return null;

    const positioned = this.applyPositions(result.nodes);

    // Sanitize edge routes: detect and fix arrows that loop backwards or have teardrop artifacts
    for (const [key, route] of result.edgeRoutes) {
      if (route.points.length < 2) continue;
      const parts = key.split("->");
      const fromId = parts[0];
      const toId = (parts[1] ?? "").replace(/#\d+$/, "");
      const fromNode = positioned.get(fromId);
      const toNode = positioned.get(toId);
      if (!fromNode || !toNode) continue;

      if (isHorizontal) {
        // For LR, arrow X should be monotonically increasing
        let looping = false;
        for (let i = 1; i < route.points.length; i++) {
          if (route.points[i][0] < route.points[0][0] - 5) { looping = true; break; }
        }
        if (looping) {
          const fx = (fromNode.x ?? 0) + fromNode.width;
          const fy = (fromNode.y ?? 0) + fromNode.height / 2;
          const tx = (toNode.x ?? 0);
          const ty = (toNode.y ?? 0) + toNode.height / 2;
          const midX = Math.round((fx + tx) / 2);
          route.points = Math.abs(fy - ty) < 1
            ? [[fx, fy], [tx, ty]]
            : [[fx, fy], [midX, fy], [midX, ty], [tx, ty]];
          route.labelPos = { x: Math.round((fx + tx) / 2), y: Math.round((fy + ty) / 2) - 15 };
        }
      }
    }

    return { positioned, edgeRoutes: result.edgeRoutes, groupBounds: result.groupBounds };
  }

  /** Apply layout positions to nodes, with absX/absY overrides and fallback for unpositioned nodes.
   *  WASM now returns correct positions for pinned nodes; the absX ?? pos.x override is a safety net. */
  private applyPositions(positions: { id: string; x: number; y: number }[]): Map<string, PositionedNode> {
    const result = new Map<string, PositionedNode>();
    for (const pos of positions) {
      const node = this.nodes.get(pos.id);
      if (node) {
        result.set(pos.id, { ...node, x: node.absX ?? pos.x, y: node.absY ?? pos.y });
      }
    }
    // Place unpositioned nodes (text, line) below the positioned graph
    let maxBottom = BASE_Y;
    for (const n of result.values()) {
      const bottom = (n.y ?? 0) + n.height;
      if (bottom > maxBottom) maxBottom = bottom;
    }
    let offsetX = BASE_X;
    for (const node of this.nodes.values()) {
      if (!result.has(node.id)) {
        const x = node.absX ?? offsetX;
        const y = node.absY ?? (maxBottom + ROW_SPACING);
        result.set(node.id, { ...node, x, y });
        if (node.absX === undefined) offsetX += node.width + 40;
      }
    }
    return result;
  }

  /** Topological order for groups: child groups before parents (so parent bounds include children). */
  private topologicalGroupOrder(): string[] {
    const groupIds = new Set(this.groups.keys());
    const result: string[] = [];
    const visited = new Set<string>();

    const visit = (id: string) => {
      if (visited.has(id)) return;
      visited.add(id);
      const group = this.groups.get(id);
      if (!group) return;
      // Visit child groups first
      for (const child of group.children) {
        if (groupIds.has(child)) visit(child);
      }
      result.push(id);
    };

    for (const id of groupIds) visit(id);
    return result;
  }
}

/** Resolve color from opts: hex overrides > preset > default */
function resolveColor(opts?: ShapeOpts, defaultPreset: ColorPreset = "backend"): ColorPair {
  const preset = opts?.color ?? defaultPreset;
  const palette = COLOR_PALETTE[preset];
  return {
    background: opts?.backgroundColor ?? palette.background,
    stroke: opts?.strokeColor ?? palette.stroke,
  };
}

