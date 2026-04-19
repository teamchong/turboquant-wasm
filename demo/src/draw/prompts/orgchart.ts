/**
 * Org-chart-mode branch. Mounted after the grammar recognises
 * `setType("orgchart")`.
 *
 * Hierarchical tree of roles / people. Each node is an addBox; edges
 * go from parent to child (no reverse). Direction is "TB"
 * (top-to-bottom) — the root goes at the top.
 */

import { PREAMBLE } from "./preamble.js";
import { SDK_TYPES } from "../drawmode/sdk-types.js";

export const ORGCHART_PROMPT = `${PREAMBLE}

You are in ORG CHART mode (setType("orgchart") has been called).
Use:
  addBox(label, opts?)    for every person / role — label is the name and optionally "name\\ntitle"
  connect(manager, report) edges go from manager to direct report (one direction only)

Do NOT use addEllipse, addDiamond, addActor, message, addTable, addClass, addGroup, or addLane.

ORG CHART example:
setType("orgchart");
setDirection("TB");
const ceo = addBox("Alice\\nCEO");
const cto = addBox("Bob\\nCTO");
const cfo = addBox("Carol\\nCFO");
const eng1 = addBox("Dave\\nEng Lead");
const eng2 = addBox("Eve\\nEng Lead");
const fin1 = addBox("Frank\\nController");
connect(ceo, cto);
connect(ceo, cfo);
connect(cto, eng1);
connect(cto, eng2);
connect(cfo, fin1);

Rules:
- One root node (the top of the hierarchy). Every other node must be reachable from the root via parent→child edges.
- Edges are directional: connect(manager, report). Never connect(report, manager).
- No cycles. Each report has exactly one manager (no dotted lines / matrix reporting in this mode).
- Use \\n in labels to separate name and title. No row/col/color/icon — layout is automatic.
- 6+ people total, 2+ levels of hierarchy.
- Direction "TB" is standard for org charts.

${SDK_TYPES}`;
