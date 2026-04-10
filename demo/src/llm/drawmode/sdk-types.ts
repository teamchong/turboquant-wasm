/**
 * Shared SDK type declarations string — used in tool descriptions for both
 * the local MCP server (src/index.ts) and the Cloudflare Worker (worker/index.ts).
 */
export const SDK_TYPES = `
type FillStyle = "solid" | "hachure" | "cross-hatch" | "zigzag";
type StrokeStyle = "solid" | "dashed" | "dotted";
type FontFamily = 1 | 2 | 3;  // Virgil / Helvetica / Cascadia
type Arrowhead = null | "arrow" | "bar" | "dot" | "triangle" | "diamond" | "diamond_outline";
type ColorPreset =
  | "frontend" | "backend" | "database" | "storage" | "ai" | "external" | "orchestration" | "queue" | "cache" | "users"
  | "aws-compute" | "aws-storage" | "aws-database" | "aws-network" | "aws-security" | "aws-ml"
  | "azure-compute" | "azure-data" | "azure-network" | "azure-ai"
  | "gcp-compute" | "gcp-data" | "gcp-network" | "gcp-ai"
  | "k8s-pod" | "k8s-service" | "k8s-ingress" | "k8s-volume";

interface ShapeOpts {
  row?: number; col?: number;          // grid position (280px cols, 220px rows)
  color?: ColorPreset;
  width?: number; height?: number;
  x?: number; y?: number;              // absolute position (bypasses grid)
  strokeColor?: string; backgroundColor?: string;  // hex overrides
  fillStyle?: FillStyle; strokeWidth?: number; strokeStyle?: StrokeStyle;
  roughness?: number; opacity?: number; roundness?: { type: number } | null;
  fontSize?: number; fontFamily?: FontFamily;
  textAlign?: "left" | "center" | "right"; verticalAlign?: "top" | "middle";
  link?: string | null; customData?: Record<string, unknown> | null;
  icon?: string;  // "lambda","docker","database","db","cloud","lock","globe","server","api","queue","cache","storage","user","users","warning","check","fire","key","mail","search","kubernetes","k8s" or emoji
}

interface ConnectOpts {
  style?: StrokeStyle; strokeColor?: string; strokeWidth?: number;
  roughness?: number; opacity?: number;
  startArrowhead?: Arrowhead; endArrowhead?: Arrowhead;  // default: null / "arrow"
  elbowed?: boolean; labelFontSize?: number;
  labelPosition?: "start" | "middle" | "end";
  customData?: Record<string, unknown> | null;
}

declare class Diagram {
  constructor(opts?: { theme?: "default" | "sketch" | "blueprint" | "minimal"; direction?: "TB" | "LR" | "RL" | "BT"; type?: "architecture" | "sequence" });
  setTheme(theme: "default" | "sketch" | "blueprint" | "minimal"): void;
  setDirection(direction: "TB" | "LR" | "RL" | "BT"): void;

  addBox(label: string, opts?: ShapeOpts): string;
  addEllipse(label: string, opts?: ShapeOpts): string;
  addDiamond(label: string, opts?: ShapeOpts): string;
  addText(text: string, opts?: { x?: number; y?: number; fontSize?: number; fontFamily?: FontFamily; color?: ColorPreset; strokeColor?: string }): string;
  addLine(points: [number, number][], opts?: { strokeColor?: string; strokeWidth?: number; strokeStyle?: StrokeStyle }): string;
  addGroup(label: string, children: string[], opts?: { padding?: number; strokeColor?: string; strokeStyle?: StrokeStyle; opacity?: number }): string;
  addFrame(name: string, children: string[]): string;
  removeGroup(id: string): void;
  removeFrame(id: string): void;

  connect(from: string, to: string, label?: string, opts?: ConnectOpts): void;

  static fromFile(path: string): Promise<Diagram>;
  static fromElements(elements: object[]): Diagram;
  static fromMermaid(syntax: string): Diagram;
  toCode(opts?: { path?: string }): string;  // convert diagram state to TypeScript SDK code
  findByLabel(label: string, opts?: { exact?: boolean }): string[];
  getNodes(): string[];
  getEdges(): Array<{ from: string; to: string; label?: string }>;
  getNode(id: string): { label: string; type: string; width: number; height: number; backgroundColor?: string; strokeColor?: string; row?: number; col?: number } | undefined;

  addActor(label: string, opts?: ShapeOpts): string;
  message(from: string, to: string, label?: string, opts?: ConnectOpts): void;

  updateNode(id: string, opts: Partial<ShapeOpts> & { label?: string }): void;
  updateEdge(from: string, to: string, update: Partial<ConnectOpts> & { label?: string }, matchLabel?: string): void;
  removeNode(id: string): void;
  removeEdge(from: string, to: string, label?: string): void;

  /** Always return this. Pass format as array for multi-output e.g. ["excalidraw", "svg"]. */
  render(opts?: { format?: "excalidraw" | "url" | "png" | "svg" | ("excalidraw" | "url" | "png" | "svg")[]; path?: string }): Promise<{ json: object; url?: string; filePath?: string; filePaths?: string[]; pngBase64?: string; svgString?: string; warnings?: string[]; changeSummary?: string; stats?: { nodes: number; edges: number; groups: number } }>;
}
`;
