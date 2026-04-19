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
  TableColumn, ClassMember, Visibility, Relation,
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
  /** Swimlanes: ordered horizontal strips that lock their children to a
   *  specific row in the layout grid. The lane index becomes the row, the
   *  user supplies the col (the sequence position within the lane). At
   *  render time each lane gets a header bar + a coloured background
   *  rectangle drawn behind its members. */
  private lanes = new Map<string, { name: string; children: string[]; index: number; opts?: GroupOpts }>();
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

  /** Switch the diagram type — "architecture" (default), "sequence", etc.
   *  Needed because the LLM driver creates the Diagram before seeing the user
   *  prompt, so it can't pass `type` to the constructor. */
  setType(type: DiagramType): void {
    this.diagramType = type;
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

  /** Add an ER table — a compound multi-row primitive that graphviz still
   *  treats as one node. Renders as one outer rectangle with a header row
   *  showing the table name and N data rows showing column entries.
   *
   *  @param name     Table name shown in the header row.
   *  @param columns  Ordered list of column rows. Each row gets one line
   *                  inside the table rectangle.
   *  @param opts     Optional ShapeOpts (color, row, col). Icons are not
   *                  rendered on tables — there's no room for them.
   */
  addTable(name: string, columns: TableColumn[], opts?: ShapeOpts): string {
    if (!Array.isArray(columns) || columns.length === 0) {
      throw new Error(`addTable("${name}"): columns must be a non-empty array`);
    }
    const id = this.nextId("tbl");
    const mergedOpts: ShapeOpts | undefined = Object.keys(this.themeDefaults).length > 0
      ? { ...this.themeDefaults, ...opts } as ShapeOpts
      : opts;

    // Compute the table's intrinsic width: max of the header line width and
    // every formatted column line width, plus padding. The render path uses
    // the same formatter so what we measure here is what gets drawn.
    const fontSize = mergedOpts?.fontSize ?? 16;
    const ff: FontFamily = mergedOpts?.fontFamily ?? 1;
    const headerLine = name;
    const columnLines = columns.map(c => Diagram.formatTableRow(c));
    const allLines = [headerLine, ...columnLines];
    let maxLineWidth = 0;
    for (const line of allLines) {
      const m = measureText(line, fontSize, ff);
      if (m.width > maxLineWidth) maxLineWidth = m.width;
    }
    const horizontalPadding = 24; // 12px on each side
    const computedWidth = mergedOpts?.width ?? Math.max(DEFAULT_WIDTH, maxLineWidth + horizontalPadding);

    // Height: header row + one row per column. Each row is a fixed
    // multiple of the font size so dividers line up cleanly regardless of
    // measured text height variation between rows.
    const rowHeight = Math.max(28, Math.round(fontSize * 1.6));
    const computedHeight = mergedOpts?.height ?? rowHeight * (1 + columns.length);

    this.nodes.set(id, {
      id, label: name, type: "table",
      tableColumns: columns,
      row: mergedOpts?.row, col: mergedOpts?.col,
      width: computedWidth,
      height: computedHeight,
      // Use a slightly lighter "database" preset by default since tables
      // are most often database entities.
      color: resolveColor(mergedOpts, "database"),
      opts: mergedOpts,
      absX: mergedOpts?.x,
      absY: mergedOpts?.y,
    });
    return id;
  }

  /** Format one column row for measurement and render. Centralised so the
   *  measure pass and emit pass cannot drift apart. */
  private static formatTableRow(c: TableColumn): string {
    const sigil = c.key === "PK" ? "🔑 " : c.key === "FK" ? "🔗 " : "";
    const typePart = c.type ? `: ${c.type}` : "";
    return `${sigil}${c.name}${typePart}`;
  }

  /** Format one class member row (attribute or method). UML visibility
   *  shows as a leading sigil: + public, - private, # protected, ~ package. */
  private static formatClassMember(m: ClassMember): string {
    const sigil = Diagram.visibilitySigil(m.visibility);
    const typePart = m.type ? `: ${m.type}` : "";
    return `${sigil} ${m.name}${typePart}`;
  }

  private static visibilitySigil(v?: Visibility): string {
    switch (v) {
      case "private":   return "-";
      case "protected": return "#";
      case "package":   return "~";
      case "public":
      default:          return "+";
    }
  }

  /** Render a multi-row compound node (table / class) into the elements list.
   *  Layout: row 0 is the centered header, rows 1..N are body rows. A divider
   *  line is drawn under every row except the last; entries in
   *  `thickDividerAfter` get a heavy solid divider (section break), all
   *  other dividers are light dotted lines (per-row separation).
   *
   *  All rows share the same height so divider geometry stays aligned with
   *  the height computed by addTable / addClass. */
  private static emitCompoundNode(
    elements: ExcalidrawElement[],
    node: PositionedNode,
    rows: { text: string; align: "left" | "center" }[],
    thickDividerAfter: Set<number>,
    o: ShapeOpts | undefined,
  ): void {
    const ff: FontFamily = o?.fontFamily ?? 1;
    const fontSize = o?.fontSize ?? 16;
    const rowHeight = node.height / rows.length;
    const baseX = node.x!;
    const baseY = node.y!;

    // 1. Outer rectangle (the compound border).
    elements.push({
      id: node.id,
      type: "rectangle",
      x: baseX, y: baseY,
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
      // No bound text — header lives in its own standalone element so it
      // sits at the top of the compound rather than centered in the whole rect.
      boundElements: null,
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

    // 2. Each row: a text element + (unless it's the last row) a divider below.
    for (let rowIdx = 0; rowIdx < rows.length; rowIdx++) {
      const row = rows[rowIdx];
      const rowY = baseY + rowHeight * rowIdx;
      const measured = measureText(row.text, fontSize, ff);
      const isHeader = rowIdx === 0;
      const textHeight = Math.max(20, measured.height);

      // Position: header centered, body rows left-aligned with 12px padding.
      const textX = row.align === "center"
        ? baseX + (node.width - measured.width) / 2
        : baseX + 12;

      elements.push({
        id: isHeader ? `${node.id}-header` : `${node.id}-row${rowIdx - 1}`,
        type: "text",
        x: textX,
        y: rowY + (rowHeight - textHeight) / 2,
        width: measured.width || 1,
        height: textHeight,
        text: row.text,
        fontSize,
        fontFamily: ff,
        lineHeight: getLineHeight(ff),
        textAlign: row.align,
        verticalAlign: "middle",
        containerId: null,
        originalText: row.text,
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

      // Skip the divider after the last row — the bottom of the compound is
      // the outer rect's bottom edge.
      if (rowIdx === rows.length - 1) continue;

      const isThick = thickDividerAfter.has(rowIdx);
      elements.push({
        id: `${node.id}-div${rowIdx}`,
        type: "line",
        x: baseX, y: rowY + rowHeight,
        width: node.width, height: 0,
        points: [[0, 0], [node.width, 0]],
        strokeColor: o?.strokeColor ?? node.color.stroke,
        backgroundColor: "transparent",
        fillStyle: "solid",
        strokeWidth: isThick ? 2 : 1,
        strokeStyle: isThick ? "solid" : "dotted",
        roughness: 0,
        opacity: isThick ? (o?.opacity ?? 100) : 60,
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
    }
  }

  /** Add a UML class — three-compartment compound primitive (header /
   *  attributes / methods). Sized as one layout node like addTable.
   *
   *  @param name        Class name shown in the header compartment.
   *  @param attributes  Optional attribute rows (fields / properties).
   *  @param methods     Optional method rows (operations).
   *  @param opts        Optional ShapeOpts. Icons are not rendered on classes.
   */
  addClass(
    name: string,
    members: { attributes?: ClassMember[]; methods?: ClassMember[] },
    opts?: ShapeOpts,
  ): string {
    const attributes = members.attributes ?? [];
    const methods = members.methods ?? [];
    if (attributes.length === 0 && methods.length === 0) {
      throw new Error(`addClass("${name}"): must have at least one attribute or method`);
    }
    const id = this.nextId("cls");
    const mergedOpts: ShapeOpts | undefined = Object.keys(this.themeDefaults).length > 0
      ? { ...this.themeDefaults, ...opts } as ShapeOpts
      : opts;

    // Compute width from the widest formatted line (header, attributes, methods).
    const fontSize = mergedOpts?.fontSize ?? 16;
    const ff: FontFamily = mergedOpts?.fontFamily ?? 1;
    const attrLines = attributes.map(a => Diagram.formatClassMember(a));
    const methodLines = methods.map(m => Diagram.formatClassMember(m));
    const allLines = [name, ...attrLines, ...methodLines];
    let maxLineWidth = 0;
    for (const line of allLines) {
      const m = measureText(line, fontSize, ff);
      if (m.width > maxLineWidth) maxLineWidth = m.width;
    }
    const horizontalPadding = 24;
    const computedWidth = mergedOpts?.width ?? Math.max(DEFAULT_WIDTH, maxLineWidth + horizontalPadding);

    // Height: header + N attributes + N methods, all at fixed row height so
    // the divider lines stay aligned regardless of measured text variation.
    const rowHeight = Math.max(28, Math.round(fontSize * 1.6));
    const computedHeight = mergedOpts?.height ?? rowHeight * (1 + attributes.length + methods.length);

    this.nodes.set(id, {
      id, label: name, type: "class",
      classAttributes: attributes,
      classMethods: methods,
      row: mergedOpts?.row, col: mergedOpts?.col,
      width: computedWidth,
      height: computedHeight,
      color: resolveColor(mergedOpts, "backend"),
      opts: mergedOpts,
      absX: mergedOpts?.x,
      absY: mergedOpts?.y,
    });
    return id;
  }

  /** Map a UML relation to start/end arrowheads + stroke style. Centralised
   *  so the connect path and tests use the same mapping. */
  private static relationArrowheads(rel: Relation): { startArrowhead: Arrowhead; endArrowhead: Arrowhead; style: StrokeStyle } {
    switch (rel) {
      // Inheritance (is-a): plain line, hollow triangle at the parent (target) end.
      // Excalidraw doesn't have a "triangle outline" — "triangle" is filled,
      // which is the closest available marker for the UML inheritance arrow.
      case "inheritance":
        return { startArrowhead: null, endArrowhead: "triangle", style: "solid" };
      // Composition (filled diamond at the WHOLE end / source).
      case "composition":
        return { startArrowhead: "diamond", endArrowhead: "arrow", style: "solid" };
      // Aggregation (open diamond at the WHOLE end / source).
      case "aggregation":
        return { startArrowhead: "diamond_outline", endArrowhead: "arrow", style: "solid" };
      // Dependency: dashed line, plain arrow.
      case "dependency":
        return { startArrowhead: null, endArrowhead: "arrow", style: "dashed" };
      // Association: plain undirected line.
      case "association":
        return { startArrowhead: null, endArrowhead: null, style: "solid" };
    }
  }

  /** Build the rendered label by prepending the icon emoji, if any.
   *  Returns the same string when no icon is set so callers can compare
   *  identity to detect "did the icon prefix actually do anything". */
  private static applyIconPrefix(label: string, opts?: ShapeOpts): string {
    if (!opts?.icon) return label;
    const emoji = ICON_PRESETS[opts.icon] ?? opts.icon;
    return `${emoji}\n${label}`;
  }

  private addShape(prefix: string, type: GraphNode["type"], label: string, opts?: ShapeOpts, defaultPreset: ColorPreset = "backend"): string {
    const id = this.nextId(prefix);
    this._registerNode(id, type, label, opts, defaultPreset);
    return id;
  }

  /** Write a GraphNode entry into `this.nodes` with theme/icon/measurement
   *  handling. Shared by addShape and addActor so both paths produce nodes
   *  with identical layout metadata — before this was extracted addActor
   *  only pushed to sequenceActors, leaving connect() unable to resolve
   *  actor identifiers in non-sequence diagrams. */
  private _registerNode(id: string, type: GraphNode["type"], label: string, opts: ShapeOpts | undefined, defaultPreset: ColorPreset): void {
    // Merge theme defaults under per-node opts (per-node wins)
    const mergedOpts: ShapeOpts | undefined = Object.keys(this.themeDefaults).length > 0
      ? { ...this.themeDefaults, ...opts } as ShapeOpts
      : opts;
    // Canonical label vs. rendered label: store the user's original `label`
    // verbatim so findByLabel/getNode/_resolveNodeRef can match it, and only
    // set displayLabel when the icon prefix actually changes the text.
    const displayLabel = Diagram.applyIconPrefix(label, mergedOpts);
    const hasIconPrefix = displayLabel !== label;
    // Measurements always use the rendered text (that's what gets drawn).
    const extraLines = displayLabel.split("\n").length - 1;
    const measured = measureText(displayLabel, mergedOpts?.fontSize ?? 16, mergedOpts?.fontFamily ?? 1);
    const autoWidth = Math.max(DEFAULT_WIDTH, measured.width + 40);
    // Minimum height must fit the text content + padding (20px top/bottom)
    const minTextHeight = measured.height + 20;
    const computedHeight = mergedOpts?.height ?? (DEFAULT_HEIGHT + extraLines * EXTRA_LINE_PX);
    this.nodes.set(id, {
      id, label, type,
      ...(hasIconPrefix ? { displayLabel } : {}),
      row: mergedOpts?.row, col: mergedOpts?.col,
      width: mergedOpts?.width ?? autoWidth,
      height: Math.max(computedHeight, minTextHeight),
      color: resolveColor(mergedOpts, defaultPreset),
      opts: mergedOpts,
      absX: mergedOpts?.x,
      absY: mergedOpts?.y,
    });
  }

  /** Add standalone text (no container shape). Returns the element ID. */
  addText(text: string, opts?: {
    x?: number; y?: number;
    fontSize?: number; fontFamily?: FontFamily;
    color?: ColorPreset; strokeColor?: string;
  }): string {
    const id = this.nextId("txt");
    const preset = opts?.color ?? "backend";
    const paletteColor = COLOR_PALETTE[preset] ?? COLOR_PALETTE["backend"];
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
   *  Children can be node IDs, node labels, or other group IDs (for nesting).
   *
   *  Children are stored as raw refs; the real linkup to node ids runs at
   *  render time via `resolveDeferred()`. Unresolvable refs are dropped
   *  silently there, not thrown here — this lets callers assemble diagrams
   *  in any order and gracefully ignore typos / stale refs. */
  addGroup(label: string, children: string[], opts?: GroupOpts): string {
    const id = this.nextId("grp");
    this.groups.set(id, { label, children: [...children], opts });
    return id;
  }

  /** Add a native Excalidraw frame container. Returns the frame ID.
   *  Children can be node IDs, node labels, or group IDs.
   *
   *  Children stored raw and resolved at render time (see addGroup). */
  addFrame(name: string, children: string[]): string {
    const id = this.nextId("frm");
    this.frames.set(id, { name, children: [...children] });
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

  /**
   * Resolve a node reference. Strictly requires the ID returned by addXxx —
   * label-string fallback was removed so the model can't accidentally pass a
   * human-readable label that collides across diagrams. The model must bind
   * addXxx to a const and pass that const.
   */
  private _resolveNodeRef(ref: string): string | undefined {
    if (typeof ref !== "string") return undefined;
    return this.nodes.has(ref) ? ref : undefined;
  }

  /**
   * Deferred-linkup resolver. Every SDK call that takes node refs (connect,
   * message, addLane, addGroup, addFrame) stores the refs raw; this method
   * walks those collections at render time, converts raw refs to canonical
   * node ids, and silently drops anything unresolvable.
   *
   * Order matters:
   *   1. Lanes — filter children to valid boxes, row-lock each to lane's
   *      index + 1 so graphviz aligns the lane horizontally.
   *   2. Groups & frames — filter children; groups may contain group refs
   *      (nesting), frames may too. Anything else is dropped.
   *   3. Edges & sequence messages — resolve endpoints. A lane ref resolves
   *      to the lane's first box (for `from`) or last box (for `to`); an
   *      empty lane means nothing to anchor to, drop the edge.
   *
   * Mutates collections in place; idempotent on already-resolved data
   * because `_resolveNodeRef(X)` returns X when X is already a node id.
   */
  private resolveDeferred(): void {
    // 0. Auto-infer lane membership from connect() edges. Small LLMs don't
    //    reliably call `addLane(name, [boxes])` — they declare empty lanes
    //    up front, then scatter `connect(lane, box)` or `connect(box, lane)`
    //    across the code. Without inference the lanes stay empty and the
    //    diagram renders as a flowchart with ghost lanes floating behind
    //    it. Walk every edge: if exactly one endpoint is a lane and the
    //    other is a box, take the membership. First lane wins per box so
    //    later conflicting edges don't yank a box between lanes.
    if (this.lanes.size > 0) {
      const boxToLane = new Map<string, string>();
      for (const [laneId, lane] of this.lanes) {
        for (const c of lane.children) {
          const nid = this._resolveNodeRef(c);
          if (nid && !boxToLane.has(nid)) boxToLane.set(nid, laneId);
        }
      }
      const assignToLane = (laneId: string, boxRaw: string) => {
        const nid = this._resolveNodeRef(boxRaw);
        if (!nid || boxToLane.has(nid)) return;
        const lane = this.lanes.get(laneId);
        if (!lane) return;
        lane.children.push(nid);
        boxToLane.set(nid, laneId);
      };
      for (const edge of this.edges) {
        if (this.lanes.has(edge.from)) assignToLane(edge.from, edge.to);
        if (this.lanes.has(edge.to)) assignToLane(edge.to, edge.from);
      }

      // Propagate membership along box→box edges until fixed-point. A flow
      // like `reserveInventory → capturePayment → sendReceipt` should carry
      // the starting lane through the chain so sequential steps end up in
      // the same lane. Propagates forward (from lane'd box to its target)
      // AND backward (from lane'd box to its source) — picks up both
      // "upstream orphan feeds into lane'd step" and "lane'd step feeds
      // into downstream orphan". Fixed-point loop bounded by edge count.
      let changed = true;
      let iter = 0;
      const MAX_ITER = this.edges.length + 1;
      while (changed && iter++ < MAX_ITER) {
        changed = false;
        for (const edge of this.edges) {
          if (this.lanes.has(edge.from) || this.lanes.has(edge.to)) continue;
          const fromNid = this._resolveNodeRef(edge.from);
          const toNid = this._resolveNodeRef(edge.to);
          if (!fromNid || !toNid) continue;
          const fromLane = boxToLane.get(fromNid);
          const toLane = boxToLane.get(toNid);
          if (fromLane && !toLane) {
            this.lanes.get(fromLane)!.children.push(toNid);
            boxToLane.set(toNid, fromLane);
            changed = true;
          } else if (toLane && !fromLane) {
            this.lanes.get(toLane)!.children.push(fromNid);
            boxToLane.set(fromNid, toLane);
            changed = true;
          }
        }
      }
    }

    // 1. Lanes — children must resolve to real nodes. Lane refs, group refs,
    //    typos etc. are dropped. Dedupe so the inference pass can't double-
    //    assign a box via both explicit addLane() + connect() inference.
    //    Valid children get their row locked.
    for (const [, lane] of this.lanes) {
      const kept: string[] = [];
      const seen = new Set<string>();
      for (const raw of lane.children) {
        const nid = this._resolveNodeRef(raw);
        if (!nid || seen.has(nid)) continue;
        const node = this.nodes.get(nid);
        if (!node) continue;
        // Row = lane index + 1 (lane 0 → row 1, so row 0 is reserved for
        // whatever sits above all lanes). If the same box appears in
        // multiple lanes, last lane wins — matching the previous behaviour
        // where eager addLane calls overwrote row on each call.
        node.row = lane.index + 1;
        seen.add(nid);
        kept.push(nid);
      }
      lane.children = kept;
    }

    // 2. Groups — children may be nodes OR other groups (nesting). Drop
    //    refs that are neither. Frames are not allowed inside groups.
    for (const [, group] of this.groups) {
      const kept: string[] = [];
      for (const raw of group.children) {
        const nid = this._resolveNodeRef(raw);
        if (nid) { kept.push(nid); continue; }
        if (this.groups.has(raw)) { kept.push(raw); continue; }
        // lane / frame / unknown → silently dropped
      }
      group.children = kept;
    }

    // 3. Frames — similar to groups but no nested frames.
    for (const [, frame] of this.frames) {
      const kept: string[] = [];
      for (const raw of frame.children) {
        const nid = this._resolveNodeRef(raw);
        if (nid) { kept.push(nid); continue; }
        if (this.groups.has(raw)) { kept.push(raw); continue; }
      }
      frame.children = kept;
    }

    // Lane-endpoint auto-resolve: a lane ref in an edge endpoint becomes
    // its first (from) or last (to) box. Must run AFTER step 1 so
    // lane.children already holds valid node ids.
    const resolveEndpoint = (raw: string, side: "from" | "to"): string | undefined => {
      const lane = this.lanes.get(raw);
      if (lane) {
        if (lane.children.length === 0) return undefined;
        return side === "from" ? lane.children[0] : lane.children[lane.children.length - 1];
      }
      if (this.groups.has(raw)) return undefined;  // groups not connectable
      if (this.frames.has(raw)) return undefined;
      return this._resolveNodeRef(raw);
    };

    // 4. Edges — resolve endpoints; drop the edge if either side fails.
    this.edges = this.edges.filter(e => {
      const f = resolveEndpoint(e.from, "from");
      const t = resolveEndpoint(e.to, "to");
      if (!f || !t) return false;
      e.from = f;
      e.to = t;
      return true;
    });

    // 5. Sequence messages — same treatment but resolve against
    //    sequenceActors (they ARE nodes too since addActor registers them,
    //    so _resolveNodeRef works). Messages between non-actor ids render
    //    fine on the sequence canvas as long as the id is in the actor
    //    position map; the sequence renderer itself skips missing entries.
    this.sequenceMessages = this.sequenceMessages.filter(m => {
      const f = resolveEndpoint(m.from, "from");
      const t = resolveEndpoint(m.to, "to");
      if (!f || !t) return false;
      m.from = f;
      m.to = t;
      return true;
    });
  }

  /** Connect two elements with an arrow.
   *
   *  `from` and `to` are stored as raw strings; the real wiring to node
   *  ids happens at render time via `resolveDeferred()`. If either ref
   *  fails to resolve there, the edge is dropped silently — this lets
   *  the caller (usually a small LLM) build diagrams out-of-order or
   *  misuse lane refs without triggering a retry cascade. Lane refs
   *  auto-resolve to the lane's first (from) or last (to) box. */
  connect(from: string, to: string, label?: string, opts?: ConnectOpts): void {
    // Type errors (undefined, non-string) are programmer bugs — fail loudly
    // so the caller sees them immediately. Ref-resolution errors (string
    // that doesn't match any node) are deferred: stored as-is and dropped
    // silently at render if they never resolve. That's the "defer the
    // linkup" part — no throws for out-of-order references.
    if (typeof from !== "string") {
      throw new Error(`connect: 'from' must be a string node ref, got ${from === undefined ? "undefined" : JSON.stringify(from)}. Assign addXxx to a const and pass that const.`);
    }
    if (typeof to !== "string") {
      throw new Error(`connect: 'to' must be a string node ref, got ${to === undefined ? "undefined" : JSON.stringify(to)}. Assign addXxx to a const and pass that const.`);
    }
    this.edges.push({
      from, to, label,
      style: opts?.style ?? "solid",
      opts,
    });
  }

  // ── Editing / Query Methods ──

  // fromFile removed — browser build has no filesystem

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
      if (eo?.cardinality) o.cardinality = eo.cardinality;
      if (eo?.relation) o.relation = eo.relation;
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

      // Pre-compute which node IDs were created via `addActor` so toCode
      // emits `d.addActor(label)` for them instead of `d.addBox(...)`.
      // Architecture-mode diagrams that mix addActor + addBox (produced
      // when the model decides a human participant should still be an
      // actor) would otherwise lose the actor intent on round-trip.
      const actorNodeIds = new Set(this.sequenceActors.map(a => a.id));

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

        // Tables get their own emit path: addTable(name, columns, opts).
        if (node.type === "table" && node.tableColumns) {
          const colsLiteral = node.tableColumns.map(c => {
            const parts = [`name: ${JSON.stringify(c.name)}`];
            if (c.type) parts.push(`type: ${JSON.stringify(c.type)}`);
            if (c.key) parts.push(`key: ${JSON.stringify(c.key)}`);
            return `{ ${parts.join(", ")} }`;
          }).join(", ");
          const tableOpts = buildShapeOpts(node);
          // Drop width/height — addTable computes them from the columns. Keep
          // row/col/color so layout placement and styling round-trip.
          delete tableOpts.width;
          delete tableOpts.height;
          delete tableOpts.icon; // icons aren't rendered on tables
          const optsStr = Object.keys(tableOpts).length > 0 ? `, ${stringify(tableOpts)}` : "";
          lines.push(`const ${varName} = d.addTable(${JSON.stringify(node.label)}, [${colsLiteral}]${optsStr});`);
          continue;
        }

        // UML class: addClass(name, { attributes, methods }, opts).
        if (node.type === "class") {
          const formatMember = (m: ClassMember): string => {
            const parts = [`name: ${JSON.stringify(m.name)}`];
            if (m.type) parts.push(`type: ${JSON.stringify(m.type)}`);
            if (m.visibility && m.visibility !== "public") parts.push(`visibility: ${JSON.stringify(m.visibility)}`);
            return `{ ${parts.join(", ")} }`;
          };
          const memberSections: string[] = [];
          if (node.classAttributes && node.classAttributes.length > 0) {
            memberSections.push(`attributes: [${node.classAttributes.map(formatMember).join(", ")}]`);
          }
          if (node.classMethods && node.classMethods.length > 0) {
            memberSections.push(`methods: [${node.classMethods.map(formatMember).join(", ")}]`);
          }
          const membersLiteral = `{ ${memberSections.join(", ")} }`;
          const classOpts = buildShapeOpts(node);
          delete classOpts.width;
          delete classOpts.height;
          delete classOpts.icon;
          const optsStr = Object.keys(classOpts).length > 0 ? `, ${stringify(classOpts)}` : "";
          lines.push(`const ${varName} = d.addClass(${JSON.stringify(node.label)}, ${membersLiteral}${optsStr});`);
          continue;
        }

        const method = actorNodeIds.has(id)
          ? "addActor"
          : node.type === "ellipse" ? "addEllipse" : node.type === "diamond" ? "addDiamond" : "addBox";
        // node.label is the canonical label (no icon prefix), so toCode can
        // emit it directly. The icon is round-tripped via the opts object.
        const codeLabel = node.label;

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

      // Emit lanes (swimlane / activity diagram strips). Lanes have the same
      // optional GroupOpts shape as groups, so the round-trip emit logic
      // mirrors the addGroup branch above.
      if (this.lanes.size > 0) {
        lines.push("");
        const sortedLanes = [...this.lanes.entries()].sort((a, b) => a[1].index - b[1].index);
        for (const [id, lane] of sortedLanes) {
          const childVars = lane.children.map(c => varNames.get(c) ?? `"${c}"`);
          const laneOpts: Record<string, unknown> = {};
          if (lane.opts?.strokeColor) laneOpts.strokeColor = lane.opts.strokeColor;
          if (lane.opts?.strokeStyle && lane.opts.strokeStyle !== "solid") laneOpts.strokeStyle = lane.opts.strokeStyle;
          if (lane.opts?.opacity !== undefined && lane.opts.opacity !== 30) laneOpts.opacity = lane.opts.opacity;
          const optsStr = Object.keys(laneOpts).length > 0 ? `, ${stringify(laneOpts)}` : "";
          const varName = toVarName(lane.name, `lane${lane.index}`);
          varNames.set(id, varName);
          lines.push(`const ${varName} = d.addLane(${JSON.stringify(lane.name)}, [${childVars.join(", ")}]${optsStr});`);
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

  /**
   * Resolve a node ID by either its real ID or its label. Returns the ID
   * string (not the node object) so it can be passed straight back into
   * `connect`, `addGroup`, etc. Returns undefined if not found.
   *
   * Models like Gemma 4 E2B tend to write `d.connect(d.getNode("X"), d.getNode("Y"))`
   * instead of binding addBox to a const — this lookup makes that pattern work.
   */
  getNode(idOrLabel: string): string | undefined {
    return this._resolveNodeRef(idOrLabel);
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
      node.color = COLOR_PALETTE[update.color] ?? COLOR_PALETTE["backend"];
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

    // Recompute displayLabel if either label or icon changed — without this,
    // updating the label of an icon-bearing node would leave the rendered
    // text pointing at the old text.
    if (update.label !== undefined || update.icon !== undefined) {
      const newDisplay = Diagram.applyIconPrefix(node.label, node.opts);
      if (newDisplay !== node.label) node.displayLabel = newDisplay;
      else delete node.displayLabel;
    }
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

  /** Add a swimlane (activity diagram lane). Lanes are horizontal strips
   *  that lock their children to a specific row in the layout grid: the
   *  first registered lane goes to row 1, the second to row 2, etc. The
   *  user controls the column for each child via its addBox row/col opts,
   *  so the natural pattern is:
   *
   *    const userLane = d.addLane("User", []);
   *    const sysLane  = d.addLane("System", []);
   *    const enter    = d.addBox("Enter creds", { row: 1, col: 1 });
   *    const validate = d.addBox("Validate",    { row: 2, col: 2 });
   *    d.addLane("User",   [enter]);    // overwrites the empty version
   *    d.addLane("System", [validate]);
   *
   *  Or more concisely, addLane the children up front and let the lane
   *  enforce their row constraints:
   *
   *    const enter    = d.addBox("Enter creds", { col: 1 });
   *    const validate = d.addBox("Validate",    { col: 2 });
   *    d.addLane("User",   [enter]);
   *    d.addLane("System", [validate]);
   *
   *  Either way, addLane mutates each child's `row` to the lane index so
   *  Graphviz aligns them on the same horizontal track. Connections that
   *  cross lane boundaries draw as vertical-then-horizontal arrows.
   *
   *  Returns the lane ID.
   */
  addLane(name: string, children: string[], opts?: GroupOpts): string {
    // Deferred linkup: children are stored as raw refs and resolved at
    // render time. Row-locking moves to render time so addBox / addLane
    // order doesn't matter.
    //
    // Dedup by name: `addLane("Customer", [])` followed later by
    // `addLane("Customer", [validateCart])` used to create two separate
    // lanes named "Customer" which broke rendering. Now the second call
    // finds the existing entry and APPENDS children / replaces opts.
    // This matches the long-standing doc comment ("overwrites the empty
    // version") and lets LLMs declare lanes upfront then fill them later
    // — the natural "outline first, detail second" pattern.
    const raw = Array.isArray(children) ? [...children] : [];
    for (const [existingId, existingLane] of this.lanes) {
      if (existingLane.name === name) {
        existingLane.children.push(...raw);
        if (opts) existingLane.opts = { ...existingLane.opts, ...opts };
        return existingId;
      }
    }
    const id = this.nextId("lane");
    const index = this.lanes.size;
    this.lanes.set(id, { name, children: raw, index, opts });
    return id;
  }

  /** Add an actor to a sequence diagram. Returns the actor ID. */
  addActor(label: string, opts?: ShapeOpts): string {
    const id = this.nextId("actor");
    this.sequenceActors.push({ id, label, index: this.sequenceActors.length, opts });
    // Also register as a regular graph node so `_resolveNodeRef`, `connect`,
    // and the architecture-mode layout engine see actors. Without this,
    // `d.addActor("User")` + `d.connect(user, browser, ...)` on a
    // non-sequence diagram threw "Source node not found: actor_1_..." at
    // runtime because the SDK kept actors in a separate collection the
    // resolver never checked. `buildSequenceElements` reads only from
    // `this.sequenceActors`, so the extra nodes entry is invisible to
    // sequence rendering.
    //
    // Default to the "users" colour + users icon so actors render as
    // human-avatar boxes in architecture mode, matching the SYSTEM_PROMPT's
    // expectation that User/Browser entities look visually distinct from
    // backend components.
    const actorOpts: ShapeOpts = { icon: "users", ...opts };
    this._registerNode(id, "rectangle", label, actorOpts, "users");
    return id;
  }

  /** Add a message between two nodes. In sequence diagrams this renders
   *  as a horizontal arrow between lifelines; in architecture diagrams
   *  it renders as an edge just like `connect(...)`. Pre-fix, `message`
   *  validated that both endpoints were in `this.sequenceActors` and
   *  threw "Actor not found: ..." otherwise, which broke the LLM's
   *  frequent pattern of mixing `addActor`/`addBox` with `connect`/
   *  `message` in a single architecture diagram. Routing through
   *  `connect()` first means the resolver handles all node kinds
   *  uniformly and the architecture edge list picks the message up.
   *  The `sequenceMessages` entry is kept in sync afterwards so
   *  `buildSequenceElements` still renders messages in sequence mode. */
  message(from: string, to: string, label?: string, opts?: ConnectOpts): void {
    // Deferred linkup — same pattern as connect(). Record raw refs in
    // both this.edges (so architecture-mode renders still see the edge)
    // and this.sequenceMessages (so sequence-mode renders see the
    // message). Both get resolved at render time; an unresolvable ref
    // drops the entry from whichever path is rendering.
    if (typeof from !== "string" || typeof to !== "string") return;
    this.connect(from, to, label, opts);
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
  async render(_opts?: RenderOpts): Promise<RenderResult> {
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

    // Browser build: JSON is always available, no file I/O or upload API needed
    return result;
  }

  /** Convert the graph to Excalidraw elements with layout. */
  private async buildElements(warnings?: string[]): Promise<ExcalidrawElement[]> {
    // Deferred linkup: connect / addLane / addGroup etc. just record raw
    // refs. This pass resolves them all against the final node set, drops
    // anything unresolvable, and row-locks lane members. After this call
    // every edge.from/to / lane.children / group.children is a real node id.
    this.resolveDeferred();

    if (this.diagramType === "sequence") {
      return this.buildSequenceElements();
    }

    const elements: ExcalidrawElement[] = [];
    const { positioned, edgeRoutes, groupBounds } = await this.layoutNodes(warnings);

    // Nudge text nodes that overlap with box nodes
    resolveTextBoxOverlaps(positioned);

    // NOTE: there used to be a "nudge non-member nodes out of group bounding
    // boxes" pass here. It was the root cause of every "addGroup breaks arrow
    // alignment" bug — it mutated node.x/y after graphviz had already computed
    // the edge routes against the pre-nudge positions, so every arrow touching
    // a nudged node ended up pointing at coordinates that no longer existed.
    //
    // The nudge tried to solve a purely visual problem (non-member nodes sitting
    // inside the bounding box we draw around a group's members) and created a
    // structural one (arrows and boxes drawn from different coordinate frames).
    // Dropping it keeps node positions consistent with graphviz's edge routes.
    // If a non-member does end up inside a group rect, that's a visual artifact
    // we can address with a real fix (either patch the Zig layout to route
    // edges against pinned positions, or compute a group rect that excludes
    // non-members) instead of mutating the layout after the fact.

    // Pre-compute arrow IDs (one per edge, in order)
    const arrowIds: string[] = [];
    for (let i = 0; i < this.edges.length; i++) {
      arrowIds.push(this.nextId("arr"));
    }

    // Render swimlane backgrounds BEFORE shape elements so the lane band
    // sits behind its members. Each lane spans the full diagram width
    // (computed from the union of all positioned nodes) and the vertical
    // band of its own children.
    if (this.lanes.size > 0) {
      // Total horizontal extent of the diagram = leftmost node X to
      // rightmost node X+width. Lanes get a 60px header zone on the left
      // for the lane label.
      let diagramMinX = Infinity, diagramMaxX = -Infinity;
      for (const node of positioned.values()) {
        if (node.x === undefined) continue;
        if (node.x < diagramMinX) diagramMinX = node.x;
        if (node.x + node.width > diagramMaxX) diagramMaxX = node.x + node.width;
      }
      const LANE_HEADER_WIDTH = 80;
      const LANE_HORIZONTAL_PADDING = 30;
      const LANE_VERTICAL_PADDING = 20;
      const laneStartX = diagramMinX - LANE_HEADER_WIDTH - LANE_HORIZONTAL_PADDING;
      const laneEndX = diagramMaxX + LANE_HORIZONTAL_PADDING;
      const laneWidth = laneEndX - laneStartX;

      // Sort lanes by index so they render in declared order.
      const sortedLanes = [...this.lanes.entries()].sort((a, b) => a[1].index - b[1].index);

      // Two-pass layout: first resolve each lane's Y band. Populated lanes
      // use their children's bbox; empty lanes are slotted in at a Y
      // derived from their neighbours so the diagram still shows them as
      // horizontal bands (previously they were skipped entirely, which
      // compressed 5-lane swimlanes down to however-many-got-populated
      // and didn't look like a swimlane at all).
      const populatedRanges: Array<{ index: number; minY: number; maxY: number }> = [];
      for (const [laneId, lane] of sortedLanes) {
        let laneMinY = Infinity, laneMaxY = -Infinity;
        for (const childId of lane.children) {
          const child = positioned.get(childId);
          if (!child || child.y === undefined) continue;
          if (child.y < laneMinY) laneMinY = child.y;
          if (child.y + child.height > laneMaxY) laneMaxY = child.y + child.height;
        }
        if (laneMinY !== Infinity) {
          populatedRanges.push({ index: lane.index, minY: laneMinY, maxY: laneMaxY });
        }
      }
      // Average height of the populated lanes — used as the default slot
      // size for empty lanes so visual density stays consistent.
      const avgPopulatedHeight = populatedRanges.length > 0
        ? populatedRanges.reduce((s, r) => s + (r.maxY - r.minY), 0) / populatedRanges.length
        : 80;
      const DEFAULT_EMPTY_HEIGHT = Math.max(60, avgPopulatedHeight);

      for (const [laneId, lane] of sortedLanes) {
        let laneMinY = Infinity, laneMaxY = -Infinity;
        for (const childId of lane.children) {
          const child = positioned.get(childId);
          if (!child || child.y === undefined) continue;
          if (child.y < laneMinY) laneMinY = child.y;
          if (child.y + child.height > laneMaxY) laneMaxY = child.y + child.height;
        }
        if (laneMinY === Infinity) {
          // Empty lane — position from neighbour bands. Find the nearest
          // populated lane below (higher index) and above (lower index);
          // slot this one between them. If we're at the bottom with no
          // lower-index neighbour, stack beneath the previous band.
          let topY: number;
          const below = populatedRanges.find(r => r.index > lane.index);
          const aboveCandidate = [...populatedRanges].reverse().find(r => r.index < lane.index);
          if (aboveCandidate) {
            topY = aboveCandidate.maxY + LANE_VERTICAL_PADDING * 2;
          } else if (below) {
            topY = below.minY - DEFAULT_EMPTY_HEIGHT - LANE_VERTICAL_PADDING * 2;
          } else {
            // Entirely empty diagram — nothing anchored. Skip; there's
            // no meaningful Y to place this at.
            continue;
          }
          laneMinY = topY;
          laneMaxY = topY + DEFAULT_EMPTY_HEIGHT;
          populatedRanges.push({ index: lane.index, minY: laneMinY, maxY: laneMaxY });
          populatedRanges.sort((a, b) => a.index - b.index);
        }

        const laneY = laneMinY - LANE_VERTICAL_PADDING;
        const laneHeight = (laneMaxY - laneMinY) + LANE_VERTICAL_PADDING * 2;

        // 1. Background band — a low-opacity rectangle spanning the row.
        elements.push({
          id: laneId,
          type: "rectangle",
          x: laneStartX, y: laneY,
          width: laneWidth, height: laneHeight,
          backgroundColor: lane.opts?.strokeColor ?? "#e7f5ff",
          strokeColor: lane.opts?.strokeColor ?? "#74c0fc",
          fillStyle: "solid",
          strokeWidth: 1,
          strokeStyle: lane.opts?.strokeStyle ?? "solid",
          roughness: 0,
          opacity: lane.opts?.opacity ?? 30,
          angle: 0,
          roundness: { type: 3 },
          boundElements: null,
          groupIds: [],
          frameId: null,
          isDeleted: false,
          updated: Date.now(),
          locked: false,
          link: null,
          // Same rationale as the addGroup rectangle — lanes are structural,
          // not addressable edge endpoints, so orphan detection must skip
          // them. Re-use the `_group` flag rather than invent a second one.
          customData: { _group: true },
          seed: randSeed(),
          version: 1,
          versionNonce: randSeed(),
        });

        // 2. Header zone — a slightly darker rectangle on the left edge
        // that holds the lane label vertically centered. Drawn AFTER the
        // background so it sits on top.
        elements.push({
          id: `${laneId}-header-bg`,
          type: "rectangle",
          x: laneStartX, y: laneY,
          width: LANE_HEADER_WIDTH, height: laneHeight,
          backgroundColor: lane.opts?.strokeColor ?? "#74c0fc",
          strokeColor: lane.opts?.strokeColor ?? "#74c0fc",
          fillStyle: "solid",
          strokeWidth: 1,
          strokeStyle: "solid",
          roughness: 0,
          opacity: lane.opts?.opacity ?? 60,
          angle: 0,
          roundness: { type: 3 },
          boundElements: null,
          groupIds: [],
          frameId: null,
          isDeleted: false,
          updated: Date.now(),
          locked: false,
          link: null,
          // Mark structural so orphan detection skips it. Without this flag
          // the header-bg rectangle looks like a regular node to
          // detectOrphanNodes — it has no customData._from/_to references
          // pointing at its id, so it gets reported as "orphan" every time,
          // triggering a retry loop on every swimlane diagram.
          customData: { _group: true },
          seed: randSeed(),
          version: 1,
          versionNonce: randSeed(),
        });

        // 3. Lane label text — centered in the header zone.
        const labelMeasured = measureText(lane.name, 16, 1);
        elements.push({
          id: `${laneId}-label`,
          type: "text",
          x: laneStartX + (LANE_HEADER_WIDTH - labelMeasured.width) / 2,
          y: laneY + (laneHeight - labelMeasured.height) / 2,
          width: labelMeasured.width,
          height: labelMeasured.height,
          text: lane.name,
          fontSize: 16,
          fontFamily: 1,
          lineHeight: getLineHeight(1),
          textAlign: "center",
          verticalAlign: "middle",
          containerId: null,
          originalText: lane.name,
          autoResize: true,
          strokeColor: "#1e1e1e",
          backgroundColor: "transparent",
          fillStyle: "solid",
          strokeWidth: 1,
          roughness: 0,
          opacity: 100,
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

      // Table (ER) and class (UML) — compound shapes that share the same
      // multi-row layout: outer rectangle, header text, one text element per
      // body row, and dividers between sections. Graphviz laid each one out
      // as one node so x/y/width/height refer to the bounding rect.
      if ((node.type === "table" && node.tableColumns) || node.type === "class") {
        const compoundRows: { text: string; align: "left" | "center" }[] = [];
        const thickDividerAfter = new Set<number>();

        if (node.type === "table") {
          // [0] header (centered) — thick divider after — [1..N] columns (left)
          compoundRows.push({ text: node.label, align: "center" });
          thickDividerAfter.add(0);
          for (const col of node.tableColumns!) {
            compoundRows.push({ text: Diagram.formatTableRow(col), align: "left" });
          }
        } else {
          // class: header (centered) → thick divider → attributes (left) →
          // thick divider → methods (left). When only one of attributes or
          // methods is set, the second thick divider is omitted because
          // there's no boundary to draw — the single section sits directly
          // beneath the header.
          const attrs = node.classAttributes ?? [];
          const methods = node.classMethods ?? [];
          compoundRows.push({ text: node.label, align: "center" });
          thickDividerAfter.add(0);
          for (const a of attrs) compoundRows.push({ text: Diagram.formatClassMember(a), align: "left" });
          if (attrs.length > 0 && methods.length > 0) thickDividerAfter.add(attrs.length);
          for (const m of methods) compoundRows.push({ text: Diagram.formatClassMember(m), align: "left" });
        }

        Diagram.emitCompoundNode(elements, node, compoundRows, thickDividerAfter, o);
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
      // Rendered text is displayLabel when an icon was set, otherwise the
      // canonical label. Measurement, text, and originalText must all match
      // what actually gets drawn or Excalidraw will misalign the bound text.
      const renderedText = node.displayLabel ?? node.label;
      const textMeasured = measureText(renderedText, o?.fontSize ?? 16, boundFf);
      const textHeight = Math.max(20, textMeasured.height);

      elements.push({
        id: textId,
        type: "text",
        x: node.x! + (node.width - textWidth) / 2,
        y: node.y! + (node.height - textHeight) / 2,
        width: textWidth,
        height: textHeight,
        text: renderedText,
        fontSize: o?.fontSize ?? 16,
        fontFamily: boundFf,
        lineHeight: getLineHeight(boundFf),
        textAlign: o?.textAlign ?? "center",
        verticalAlign: o?.verticalAlign ?? "middle",
        containerId: node.id,
        originalText: renderedText,
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

      // Compose the displayed edge label. ER cardinality (if set) is prefixed
      // to whatever the user passed as the label so "1:N" sits next to the
      // verb describing the relationship.
      const cardinality = co?.cardinality;
      const effectiveLabel = cardinality
        ? (edge.label ? `${cardinality}  ${edge.label}` : cardinality)
        : edge.label;
      const labelTextId = effectiveLabel ? this.nextId("arrlbl") : undefined;

      // UML relation overrides the explicit start/end arrowheads + style
      // when set. This keeps the model from having to remember which
      // arrowhead means inheritance vs composition vs aggregation.
      const relationArrows = co?.relation ? Diagram.relationArrowheads(co.relation) : null;
      const effectiveStartArrow = relationArrows ? relationArrows.startArrowhead : (co?.startArrowhead ?? null);
      const effectiveEndArrow = relationArrows ? relationArrows.endArrowhead : (co?.endArrowhead !== undefined ? co.endArrowhead : "arrow");
      const effectiveStrokeStyle = relationArrows ? relationArrows.style : edge.style;

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
        strokeStyle: effectiveStrokeStyle,
        roughness: co?.roughness ?? 0,
        roundness: null,
        elbowed: isElbowed,
        opacity: co?.opacity ?? 100,
        angle: 0,
        startBinding: null,
        endBinding: null,
        startArrowhead: effectiveStartArrow,
        endArrowhead: effectiveEndArrow,
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
      if (effectiveLabel && labelTextId) {
        const labelWidth = measureText(effectiveLabel, co?.labelFontSize ?? 14, 1).width + 16;
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
          text: effectiveLabel,
          fontSize: co?.labelFontSize ?? 14,
          fontFamily: labelFf,
          lineHeight: getLineHeight(labelFf),
          textAlign: "center",
          verticalAlign: "middle",
          containerId: null,
          originalText: effectiveLabel,
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
        // Marker so downstream consumers (notably the orphan detector in
        // demo/src/draw/main.ts) can skip group containers — they're
        // decorative frames, not edge endpoints, so it's always invalid to
        // flag them as "unconnected nodes".
        customData: { _group: true },
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
        backgroundColor: actor.opts?.backgroundColor ?? (COLOR_PALETTE[actor.opts?.color ?? "backend"] ?? COLOR_PALETTE["backend"]).background,
        strokeColor: actor.opts?.strokeColor ?? (COLOR_PALETTE[actor.opts?.color ?? "backend"] ?? COLOR_PALETTE["backend"]).stroke,
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
        strokeColor: (COLOR_PALETTE[actor.opts?.color ?? "backend"] ?? COLOR_PALETTE["backend"]).stroke,
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

      // Stamp the endpoint actor ids in customData so downstream consumers
      // (orphan detection, graph introspection) can see the message → actor
      // reference without re-deriving it from pixel positions. Matches the
      // customData._from / _to convention used by architecture edges at
      // buildElements (~line 2520).
      const seqEdgeCustom = { _from: msg.from, _to: msg.to };
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
          customData: seqEdgeCustom,
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
          customData: seqEdgeCustom,
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
      n => n.type === "rectangle" || n.type === "ellipse" || n.type === "diamond" || n.type === "table" || n.type === "class",
    );
    if (!hasShapes) return { positioned: this.applyPositions([]) };
    throw new Error("Graphviz WASM layout failed. Ensure WASM module is built and loaded.");
  }

  private async layoutNodesWasm(): Promise<{ positioned: Map<string, PositionedNode>; edgeRoutes: Map<string, EdgeRoute>; groupBounds?: GroupBounds[] } | null> {
    // Auto-load WASM on first use (ensures Graphviz layout works for direct SDK imports)
    if (!isWasmLoaded()) await loadWasm();
    // Tables and classes are compound primitives but graphviz still treats
    // them as one node sized by the outer rectangle, so include them
    // alongside the simple shapes in the layout filter.
    const graphNodes = Array.from(this.nodes.values()).filter(
      n => n.type === "rectangle" || n.type === "ellipse" || n.type === "diamond" || n.type === "table" || n.type === "class",
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

    // Pick layout engine. `dot` is a hierarchical Sugiyama layout — ideal for
    // DAGs (architecture, flowchart, ER) but its routing of cycle-closing
    // back-edges in LR mode runs long curves through the middle of the chart
    // and overlaps node labels. `circo` is a circular layout — every edge is
    // short and roughly tangent to the cycle, no back-edges by construction.
    // Heuristic: if every node is an ellipse and there are no row/col grid
    // constraints, treat it as a state machine and use circo.
    let engine = allAbsolute ? "nop2" : "dot";
    const allEllipses = wasmNodes.length > 0 && wasmNodes.every(n => n.type === "ellipse");
    if (allEllipses && !hasRowConstraints) engine = "circo";

    const result = await layoutGraphWasm(wasmNodes, wasmEdges, effectiveGroups, { rankdir: this.direction, engine });
    // Remember which engine ran so the back-edge sanitizer below knows to
    // skip itself for circo (whose radial layout legitimately produces
    // edges going in all four directions).
    const engineUsed = engine;
    if (!result) return null;

    const positioned = this.applyPositions(result.nodes);

    // Sanitize edge routes: detect dot's back-edge teardrop artifacts (arrows
    // whose X goes backward in LR layout) and replace them with a clean
    // orthogonal detour. SKIPPED for circo — circular layouts produce edges
    // legitimately going in every direction, and collapsing them to
    // orthogonal would destroy the whole point of the radial routing.
    if (engineUsed !== "circo") {
      for (const [key, route] of result.edgeRoutes) {
        if (route.points.length < 2) continue;
        const parts = key.split("->");
        const fromId = parts[0];
        const toId = (parts[1] ?? "").replace(/#\d+$/, "");
        const fromNode = positioned.get(fromId);
        const toNode = positioned.get(toId);
        if (!fromNode || !toNode) continue;

        if (isHorizontal) {
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
  const palette = COLOR_PALETTE[preset] ?? COLOR_PALETTE["backend"];
  return {
    background: opts?.backgroundColor ?? palette.background,
    stroke: opts?.strokeColor ?? palette.stroke,
  };
}

