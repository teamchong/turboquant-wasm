/**
 * Shared SDK type declarations string — used in tool descriptions for both
 * the local MCP server (src/index.ts) and the Cloudflare Worker (worker/index.ts).
 */
export const SDK_TYPES = `
type FillStyle = "solid" | "hachure" | "cross-hatch" | "zigzag";
type StrokeStyle = "solid" | "dashed" | "dotted";
type Arrowhead = null | "arrow" | "bar" | "dot" | "triangle" | "diamond" | "diamond_outline";
type ColorPreset =
  | "frontend" | "backend" | "database" | "storage" | "ai" | "external" | "orchestration" | "queue" | "cache" | "users"
  | "aws-compute" | "aws-storage" | "aws-database" | "aws-network" | "aws-security" | "aws-ml"
  | "gcp-compute" | "gcp-data" | "gcp-network" | "gcp-ai"
  | "k8s-pod" | "k8s-service" | "k8s-ingress" | "k8s-volume";
type Cardinality = "1:1" | "1:N" | "N:1" | "N:M";
type Relation = "inheritance" | "composition" | "aggregation" | "dependency" | "association";
type Visibility = "public" | "private" | "protected" | "package";

interface ShapeOpts {
  row?: number; col?: number;
  color?: ColorPreset;
  icon?: string;  // "server","database","lock","globe","users","api","queue","cache","cloud","code","shield","search","mail" or emoji
}
interface TableColumn { name: string; type?: string; key?: "PK" | "FK" }
interface ClassMember { name: string; type?: string; visibility?: Visibility }
interface ConnectOpts {
  style?: StrokeStyle; strokeColor?: string;
  startArrowhead?: Arrowhead; endArrowhead?: Arrowhead;
  labelPosition?: "start" | "middle" | "end";
  cardinality?: Cardinality;  // ER
  relation?: Relation;        // UML
}

declare class Diagram {
  setDirection(direction: "TB" | "LR" | "RL" | "BT"): void;
  setType(type: "architecture" | "sequence" | "swimlane" | "class"): void;

  addBox(label: string, opts?: ShapeOpts): string;
  addEllipse(label: string, opts?: ShapeOpts): string;
  addDiamond(label: string, opts?: ShapeOpts): string;
  addTable(name: string, columns: TableColumn[], opts?: ShapeOpts): string;
  addClass(name: string, members: { attributes?: ClassMember[]; methods?: ClassMember[] }, opts?: ShapeOpts): string;
  addActor(label: string, opts?: ShapeOpts): string;

  addGroup(label: string, children: string[]): string;
  addLane(name: string, children: string[]): string;

  connect(from: string, to: string, label?: string, opts?: ConnectOpts): void;
  message(from: string, to: string, label?: string, opts?: ConnectOpts): void;
}
`;
