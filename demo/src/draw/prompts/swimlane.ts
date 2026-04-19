/**
 * Swimlane-mode branch. Mounted after the grammar recognises
 * `setType("swimlane")`.
 *
 * Process flow grouped by responsibility. Each lane is a horizontal
 * band containing the activity steps performed by that actor/system.
 * Steps are addBox nodes; lanes group them via addLane. Arrows
 * between boxes automatically route across lanes.
 */

import { PREAMBLE } from "./preamble.js";
import { SDK_TYPES } from "../drawmode/sdk-types.js";

export const SWIMLANE_PROMPT = `${PREAMBLE}

You are in SWIMLANE mode (setType("swimlane") has been called).
Use:
  addBox(label, { col, color?, icon? })    for every activity step — col only, NOT row (lane assigns the row)
  addLane(name, [stepConst, ...])          groups activity boxes into a named lane (actor/system/responsibility)
  connect(from, to, "label")               arrows between activity boxes; cross-lane routing is automatic

Do NOT use addActor, message, addEllipse, addDiamond, addTable, addClass, addGroup.

SWIMLANE example:
setType("swimlane");
const placeOrder = addBox("Place order", { col: 0, color: "frontend" });
const validateCart = addBox("Validate cart", { col: 1, color: "backend" });
const charge = addBox("Charge card", { col: 2, color: "backend" });
const ship = addBox("Ship package", { col: 3, color: "storage" });
const notify = addBox("Send receipt", { col: 4, color: "queue" });
addLane("Customer", [placeOrder]);
addLane("Web App", [validateCart]);
addLane("Payment", [charge]);
addLane("Warehouse", [ship]);
addLane("Email", [notify]);
connect(placeOrder, validateCart, "submit");
connect(validateCart, charge, "authorize");
connect(charge, ship, "paid");
connect(ship, notify, "shipped");

Rules:
- Lane names are the FIRST STRING ARGUMENT to addLane — they are NOT separate nodes. Never addBox a lane.
- NEVER assign addLane(...) to a const. Lanes are containers, not nodes; their return value is NOT a valid arg to connect() or to another addLane([...]).
- connect(from, to, "label") — both from and to MUST be addBox-returned consts. NEVER pass a lane name (string) or a lane variable to connect.
- Activity steps use col only (NOT row) — each lane assigns a fixed row automatically. col sequences the boxes within a lane and across lanes temporally.
- After all boxes are declared, group them into lanes. Lanes stack in declaration order (first addLane = top). Each box goes in EXACTLY ONE lane.
- 3+ lanes, 6+ activity boxes total. Lane names should be actor / system / responsibility labels ("User", "Server", "Database", "Warehouse").
- Every addBox must appear in some addLane's children array. Orphan boxes are invalid.

${SDK_TYPES}`;
