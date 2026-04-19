import { z } from "zod";

/** Excalidraw fill styles */
export type FillStyle = "solid" | "hachure" | "cross-hatch" | "zigzag";

/** Excalidraw stroke styles */
export type StrokeStyle = "solid" | "dashed" | "dotted";

/** Excalidraw font families: 1=Virgil, 2=Helvetica, 3=Cascadia */
export type FontFamily = 1 | 2 | 3;

/** Excalidraw arrowhead types */
export type Arrowhead = null | "arrow" | "bar" | "dot" | "triangle" | "diamond" | "diamond_outline";

/** Excalidraw text alignment */
export type TextAlign = "left" | "center" | "right";

/** Excalidraw vertical alignment */
export type VerticalAlign = "top" | "middle";

/** Color presets for diagram components */
export type ColorPreset =
  | "frontend" | "backend" | "database" | "storage"
  | "ai" | "external" | "orchestration" | "queue"
  | "cache" | "users"
  // AWS
  | "aws-compute" | "aws-storage" | "aws-database" | "aws-network" | "aws-security" | "aws-ml"
  // Azure
  | "azure-compute" | "azure-data" | "azure-network" | "azure-ai"
  // GCP
  | "gcp-compute" | "gcp-data" | "gcp-network" | "gcp-ai"
  // K8s
  | "k8s-pod" | "k8s-service" | "k8s-ingress" | "k8s-volume";

export interface ColorPair {
  background: string;
  stroke: string;
}

export const COLOR_PALETTE: Record<ColorPreset, ColorPair> = {
  frontend:      { background: "#a5d8ff", stroke: "#1971c2" },
  backend:       { background: "#d0bfff", stroke: "#7048e8" },
  database:      { background: "#b2f2bb", stroke: "#2f9e44" },
  storage:       { background: "#ffec99", stroke: "#f08c00" },
  ai:            { background: "#e599f7", stroke: "#9c36b5" },
  external:      { background: "#ffc9c9", stroke: "#e03131" },
  orchestration: { background: "#ffa8a8", stroke: "#c92a2a" },
  queue:         { background: "#fff3bf", stroke: "#fab005" },
  cache:         { background: "#ffe8cc", stroke: "#fd7e14" },
  users:         { background: "#e7f5ff", stroke: "#1971c2" },
  // AWS — light pastel backgrounds with brand-color strokes for readability
  "aws-compute":  { background: "#ffe8cc", stroke: "#FF9900" },
  "aws-storage":  { background: "#d4ecd0", stroke: "#3F8624" },
  "aws-database": { background: "#d6ddf0", stroke: "#3B48CC" },
  "aws-network":  { background: "#e4d6ff", stroke: "#8C4FFF" },
  "aws-security": { background: "#ffd6dc", stroke: "#DD344C" },
  "aws-ml":       { background: "#ccf0e8", stroke: "#01A88D" },
  // Azure — light pastel backgrounds with brand-color strokes
  "azure-compute": { background: "#cce4f7", stroke: "#0078D4" },
  "azure-data":    { background: "#d6f5ff", stroke: "#0097C7" },
  "azure-network": { background: "#e2d4f5", stroke: "#773ADC" },
  "azure-ai":      { background: "#cce4f7", stroke: "#0078D4" },
  // GCP — light pastel backgrounds with brand-color strokes
  "gcp-compute": { background: "#d6e4fc", stroke: "#4285F4" },
  "gcp-data":    { background: "#d4ecd8", stroke: "#34A853" },
  "gcp-network": { background: "#fef3cc", stroke: "#F9AB00" },
  "gcp-ai":      { background: "#fcd6d4", stroke: "#EA4335" },
  // K8s — light pastel backgrounds with brand-color strokes
  "k8s-pod":     { background: "#d4def5", stroke: "#326CE5" },
  "k8s-service": { background: "#dde9f5", stroke: "#4577A0" },
  "k8s-ingress": { background: "#ccf2f5", stroke: "#0097A7" },
  "k8s-volume":  { background: "#fff3cc", stroke: "#C99605" },
};

/** Options for adding a box/ellipse to the diagram */
export interface ShapeOpts {
  row?: number;
  col?: number;
  color?: ColorPreset;
  width?: number;
  height?: number;
  /** Absolute x position (bypasses grid layout) */
  x?: number;
  /** Absolute y position (bypasses grid layout) */
  y?: number;
  /** Hex stroke color override (takes precedence over ColorPreset) */
  strokeColor?: string;
  /** Hex background color override (takes precedence over ColorPreset) */
  backgroundColor?: string;
  fillStyle?: FillStyle;
  strokeWidth?: number;
  strokeStyle?: StrokeStyle;
  /** 0=architect, 1=artist, 2=cartoonist */
  roughness?: number;
  /** 0-100 */
  opacity?: number;
  roundness?: { type: number } | null;
  fontSize?: number;
  fontFamily?: FontFamily;
  textAlign?: TextAlign;
  verticalAlign?: VerticalAlign;
  /** Hyperlink URL attached to the element */
  link?: string | null;
  /** Arbitrary custom metadata stored on the element */
  customData?: Record<string, unknown> | null;
  /** Icon preset name or raw emoji — prepended above the label */
  icon?: string;
}

/** ER cardinality marker rendered alongside the edge label. */
export type Cardinality = "1:1" | "1:N" | "N:1" | "N:M";

/** UML relationship style. Maps to a specific arrowhead pair on the edge. */
export type Relation = "inheritance" | "composition" | "aggregation" | "dependency" | "association";

/** Options for connecting two elements */
export interface ConnectOpts {
  style?: StrokeStyle;
  strokeColor?: string;
  strokeWidth?: number;
  roughness?: number;
  /** 0-100 */
  opacity?: number;
  startArrowhead?: Arrowhead;
  endArrowhead?: Arrowhead;
  /** Whether to use elbow routing (default true) */
  elbowed?: boolean;
  labelFontSize?: number;
  /** Where to place the edge label along the arrow path */
  labelPosition?: "start" | "middle" | "end";
  /** Arbitrary custom metadata stored on the arrow element */
  customData?: Record<string, unknown> | null;
  /** ER cardinality (e.g. "1:N") prepended to the edge label. */
  cardinality?: Cardinality;
  /** UML relation style — picks specific arrowheads automatically:
   *    inheritance → triangle (open) at target end (UML "is-a")
   *    composition → diamond (filled) at source end (UML "owns")
   *    aggregation → diamond (outline) at source end (UML "has-a")
   *    dependency  → dashed line, plain arrow at target end
   *    association → plain solid line, no arrowheads
   *  Setting `relation` overrides startArrowhead/endArrowhead/style. */
  relation?: Relation;
}

// ── Zod Schemas for Excalidraw elements ──

const BindingSchema = z.object({
  elementId: z.string(),
  focus: z.number().optional(),
  gap: z.number().optional(),
}).passthrough();

export const ExcalidrawElementSchema = z.object({
  id: z.string(),
  type: z.string(),
  x: z.number(),
  y: z.number(),
  width: z.number(),
  height: z.number(),
  angle: z.number().optional(),
  strokeColor: z.string().optional(),
  backgroundColor: z.string().optional(),
  fillStyle: z.string().optional(),
  strokeWidth: z.number().optional(),
  strokeStyle: z.string().optional(),
  roughness: z.number().optional(),
  opacity: z.number().optional(),
  roundness: z.object({ type: z.number() }).nullable().optional(),
  seed: z.number().optional(),
  version: z.number().optional(),
  versionNonce: z.number().optional(),
  isDeleted: z.boolean().optional(),
  groupIds: z.array(z.string()).optional(),
  frameId: z.string().nullable().optional(),
  boundElements: z.array(z.object({ type: z.string(), id: z.string() })).nullable().optional(),
  updated: z.number().optional(),
  locked: z.boolean().optional(),
  link: z.string().nullable().optional(),
  customData: z.record(z.string(), z.unknown()).nullable().optional(),
  // Text fields
  text: z.string().optional(),
  fontSize: z.number().optional(),
  fontFamily: z.number().optional(),
  lineHeight: z.number().optional(),
  textAlign: z.string().optional(),
  verticalAlign: z.string().optional(),
  containerId: z.string().nullable().optional(),
  originalText: z.string().optional(),
  autoResize: z.boolean().optional(),
  // Arrow/Line fields
  points: z.array(z.array(z.number())).optional(),
  startBinding: BindingSchema.nullable().optional(),
  endBinding: BindingSchema.nullable().optional(),
  startArrowhead: z.string().nullable().optional(),
  endArrowhead: z.string().nullable().optional(),
  elbowed: z.boolean().optional(),
  // Frame fields
  name: z.string().optional(),
}).passthrough();

export type ExcalidrawElement = z.infer<typeof ExcalidrawElementSchema>;

export const ExcalidrawFileSchema = z.object({
  type: z.literal("excalidraw"),
  version: z.number(),
  source: z.string().optional(),
  elements: z.array(ExcalidrawElementSchema),
  appState: z.record(z.string(), z.unknown()).optional(),
  files: z.record(z.string(), z.unknown()).optional(),
}).passthrough();

export type ExcalidrawFile = z.infer<typeof ExcalidrawFileSchema>;

/** Excalidraw file format version — used in ExcalidrawFile JSON */
export const EXCALIDRAW_VERSION = 2;

/** Layout direction for Graphviz rankdir */
export type LayoutDirection = "TB" | "LR" | "RL" | "BT";

/** Diagram type.
 *
 *  Two categories at the LAYOUT level:
 *    - "sequence" triggers the vertical actors + horizontal-messages
 *      chronological renderer.
 *    - Everything else falls through to the generic Graphviz layout
 *      (boxes/ellipses/diamonds/tables/classes connected via edges).
 *
 *  The sub-types within the generic category ("architecture",
 *  "flowchart", "state", "orgchart", "er", "class", "swimlane") are
 *  presentational — they tell the model which shapes to prefer and
 *  which rules to follow, but share the underlying renderer. Listed
 *  exhaustively here so the type system matches every string that the
 *  SDK prompts can produce.
 */
export type DiagramType =
  | "architecture"
  | "sequence"
  | "flowchart"
  | "state"
  | "orgchart"
  | "er"
  | "class"
  | "swimlane";

/** Theme preset names */
export type ThemePreset = "default" | "sketch" | "blueprint" | "minimal";

/** Theme definition — defaults applied to all shapes unless overridden per-node */
export interface ThemeOpts {
  fillStyle?: FillStyle;
  roughness?: number;
  strokeWidth?: number;
  fontFamily?: FontFamily;
  opacity?: number;
}

/** Output format options */
export type OutputFormat = "excalidraw" | "url" | "png" | "svg";
export type OutputFormatInput = OutputFormat | OutputFormat[];

/** Diagram statistics computed from the element graph */
export interface DiagramStats {
  nodes: number;
  edges: number;
  groups: number;
}

/** Result from rendering a diagram */
export interface RenderResult {
  json: ExcalidrawFile;
  url?: string;
  filePath?: string;
  /** All file paths written (when using multi-format output) */
  filePaths?: string[];
  pngBase64?: string;
  svgString?: string;
  warnings?: string[];
  changeSummary?: string;
  stats?: DiagramStats;
}

/** Render options */
export interface RenderOpts {
  format?: OutputFormatInput;
  path?: string;
}

/** Options for group styling and layout */
export interface GroupOpts {
  /** Padding in pixels around group children (default 30) */
  padding?: number;
  /** Stroke color for the group boundary */
  strokeColor?: string;
  /** Stroke style for the group boundary (default "dashed") */
  strokeStyle?: StrokeStyle;
  /** Opacity 0-100 (default 60) */
  opacity?: number;
}

/** A single column row inside an ER table or UML class compound node. */
export interface TableColumn {
  /** Column / attribute name. */
  name: string;
  /** Optional type or value annotation rendered to the right of the name
   *  (e.g. "INT", "VARCHAR(100)" for ER, or "string" / "void" for UML). */
  type?: string;
  /** Key marker rendered as a prefix sigil. PK = primary key, FK = foreign key. */
  key?: "PK" | "FK";
}

/** UML visibility for class members. Rendered as the leading sigil on each row:
 *  + public, - private, # protected, ~ package. Defaults to + when omitted. */
export type Visibility = "public" | "private" | "protected" | "package";

/** A single attribute or method row inside an addClass compound node. */
export interface ClassMember {
  /** Member name (attribute name or method signature like "save()"). */
  name: string;
  /** Optional type annotation. For attributes: the type. For methods: return type. */
  type?: string;
  /** UML visibility marker. Defaults to "public". */
  visibility?: Visibility;
}

/** Internal node representation before layout */
export interface GraphNode {
  id: string;
  /** Canonical label as the user wrote it. Used for findByLabel / getNode
   *  lookup and for code regeneration in toCode(). Never includes the icon
   *  emoji prefix — that belongs on displayLabel and is a render concern. */
  label: string;
  /** Rendered label with icon emoji prepended (e.g. "🔍\nSearch"). Set only
   *  when an icon was supplied; otherwise undefined and renderers fall back
   *  to `label`. Splitting this lets _resolveNodeRef match by canonical
   *  label even when an icon mutates the rendered text. */
  displayLabel?: string;
  /** Node shape class. "table" (ER) and "class" (UML) are compound multi-row
   *  primitives that graphviz still treats as a single layout node. */
  type: "rectangle" | "ellipse" | "diamond" | "text" | "line" | "table" | "class";
  row?: number;
  col?: number;
  width: number;
  height: number;
  color: ColorPair;
  /** Stored ShapeOpts for property pass-through */
  opts?: ShapeOpts;
  /** Absolute position override */
  absX?: number;
  absY?: number;
  /** For line elements: array of [x,y] point pairs */
  linePoints?: [number, number][];
  /** For "table" type: ordered column rows rendered inside the outer rect. */
  tableColumns?: TableColumn[];
  /** For "class" type: attributes section (rendered between header and methods). */
  classAttributes?: ClassMember[];
  /** For "class" type: methods section (rendered below attributes). */
  classMethods?: ClassMember[];
}

/** Internal edge representation before layout */
export interface GraphEdge {
  from: string;
  to: string;
  label?: string;
  style: StrokeStyle;
  /** Full connect opts for property pass-through */
  opts?: ConnectOpts;
}
