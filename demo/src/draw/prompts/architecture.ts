/**
 * Architecture-mode branch. Mounted after the grammar recognises
 * `setType("architecture")`.
 *
 * Generic infrastructure diagrams: services, deployments, data planes,
 * control planes. Every node is an addBox with row/col/color/icon;
 * connections are labelled edges; related nodes are grouped via
 * addGroup. For any diagram that isn't a flowchart/state/orgchart/er/
 * class/swimlane — those have their own modes and branches.
 *
 * No "think briefly" tail here: infrastructure diagrams (15+ nodes
 * across columns with row/col/color/icon assignments) need thorough
 * planning or the model hallucinates undeclared vars.
 */

import { PREAMBLE } from "./preamble.js";
import { SDK_TYPES } from "../drawmode/sdk-types.js";

export const ARCHITECTURE_PROMPT = `${PREAMBLE}

You are in ARCHITECTURE mode (setType("architecture") has been called).
Use:
  addBox(label, { row, col, color, icon })  for every service / component
  connect(from, to, "label")                for every edge, with a short verb as label
  addGroup("name", [boxes])                 to cluster related components

Do NOT use addActor, message, addEllipse, addDiamond, addTable, addClass, addLane — those belong to other modes.

ARCHITECTURE example:
setType("architecture");
const browser = addBox("Browser", { row: 0, col: 0, color: "frontend", icon: "globe" });
const cdn = addBox("CDN", { row: 0, col: 2, color: "external", icon: "cloud" });
const lb = addBox("Load Balancer", { row: 1, col: 1, color: "orchestration", icon: "server" });
const api = addBox("API Server", { row: 2, col: 1, color: "backend", icon: "api" });
const cache = addBox("Redis", { row: 2, col: 0, color: "cache", icon: "cache" });
const db = addBox("Postgres", { row: 2, col: 2, color: "database", icon: "database" });
const queue = addBox("Job Queue", { row: 3, col: 1, color: "queue", icon: "queue" });
const worker = addBox("Worker", { row: 3, col: 2, color: "backend", icon: "code" });
connect(browser, cdn, "GET /static");
connect(browser, lb, "HTTPS");
connect(lb, api, "round-robin");
connect(api, cache, "GET session");
connect(api, db, "SQL");
connect(api, queue, "enqueue job");
connect(queue, worker, "dispatch");
connect(worker, db, "write result");
addGroup("Edge", [browser, cdn]);
addGroup("App", [lb, api]);
addGroup("Data", [cache, db]);
addGroup("Async", [queue, worker]);

Pattern — plan BEFORE emitting code (thinking channel allowed to be thorough here): list every component, pick its row/col/color/icon, then write the addBox lines. Every variable you reference in connect/addGroup MUST be one you declared on a prior line — NEVER a bare word like \`request\` or \`user\` that wasn't returned by a prior addBox call.

Rules:
- Each node represents a distinct real entity. Do NOT duplicate.
- EVERY addBox node MUST be referenced by at least one connect OR addGroup. Orphan nodes are invalid — either wire them up or don't add them.
- EVERY node needs row, col, color, icon. Spread components across columns 0-4 so the layout reads naturally.
- Aim for 15+ nodes for a real system. Use \\n in labels for multi-line text.
- color is one of: "frontend" | "backend" | "database" | "cache" | "queue" | "external" | "orchestration" | "users" | "storage" | "ai"
- icon is one of: "server" | "database" | "lock" | "globe" | "users" | "api" | "cache" | "queue" | "cloud" | "code" | "shield" | "search"
- connect(from, to, "label") — short verb / protocol label only, no options object, no hex colors.
- 4+ groups. Group names are short ("Backend", "Data", "Edge") — nouns, not sentences.

${SDK_TYPES}`;
