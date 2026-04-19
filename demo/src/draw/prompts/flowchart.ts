/**
 * Flowchart-mode branch. Mounted after the grammar recognises
 * `setType("flowchart")`.
 *
 * Decision process with branching: addBox for action steps, addEllipse
 * for start/end, addDiamond for decisions, connect with labels on
 * BOTH outgoing arrows of a diamond ("yes"/"no" or the actual
 * condition). Direction is typically "TB" (top-to-bottom) — set with
 * setDirection("TB") if the default doesn't match the user's wording.
 *
 * No addActor, message, addTable, addClass, addGroup, or addLane —
 * those belong to other modes. Use addBox/addEllipse/addDiamond only.
 */

import { PREAMBLE } from "./preamble.js";
import { SDK_TYPES } from "../drawmode/sdk-types.js";

export const FLOWCHART_PROMPT = `${PREAMBLE}

You are in FLOWCHART mode (setType("flowchart") has been called).
Flowcharts show a decision process with branching logic. Use:
  addEllipse(label)   for start / end nodes
  addBox(label)       for action / process steps
  addDiamond(label)   for decisions (branch points)
  connect(from, to, "label")  for edges — ALWAYS label the two outgoing arrows of a diamond

Do NOT use addActor, message, addTable, addClass, addGroup, or addLane.

FLOWCHART example:
setType("flowchart");
setDirection("TB");
const start = addEllipse("Start");
const login = addBox("Enter credentials");
const validate = addDiamond("Valid?");
const dashboard = addBox("Show dashboard");
const error = addBox("Show error");
const end = addEllipse("End");
connect(start, login);
connect(login, validate);
connect(validate, dashboard, "yes");
connect(validate, error, "no");
connect(dashboard, end);
connect(error, login, "retry");

Rules:
- Exactly one start ellipse and at least one end ellipse. Every path from start must reach an end.
- EVERY diamond's outgoing arrows MUST be labelled ("yes"/"no", "true"/"false", or a specific condition). Unlabelled decision branches are invalid.
- Action steps use addBox; do NOT decorate with row/col/color/icon (those are architecture-mode concerns).
- Every addBox/addEllipse/addDiamond result must be a const referenced by at least one connect.
- 6+ nodes total. Use \\n in labels for multi-line text.
- Direction "TB" is most natural for flowcharts; "LR" is acceptable for short horizontal flows.

${SDK_TYPES}`;
