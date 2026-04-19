/**
 * State-machine-mode branch. Mounted after the grammar recognises
 * `setType("state")`.
 *
 * State machines have named states connected by transitions. States
 * are drawn as ellipses; transitions are edges labelled with the event
 * that triggers the state change. Direction is usually "LR" (reads
 * left-to-right like a timeline) but "TB" is fine for vertical flows.
 */

import { PREAMBLE } from "./preamble.js";
import { SDK_TYPES } from "../drawmode/sdk-types.js";

export const STATE_PROMPT = `${PREAMBLE}

You are in STATE MACHINE mode (setType("state") has been called).
Use:
  addEllipse(label)   for every state
  connect(from, to, "transition label")  for every transition — each edge is an event that causes the state change

Do NOT use addBox, addDiamond, addActor, message, addTable, addClass, addGroup, or addLane.

STATE example:
setType("state");
setDirection("LR");
const idle = addEllipse("Idle");
const loading = addEllipse("Loading");
const ready = addEllipse("Ready");
const error = addEllipse("Error");
connect(idle, loading, "fetch()");
connect(loading, ready, "success");
connect(loading, error, "failure");
connect(ready, idle, "reset()");
connect(error, idle, "reset()");
connect(error, loading, "retry()");

Rules:
- Every state is an addEllipse. Do NOT use addBox for states.
- Every transition (connect) MUST have a label describing the event or condition that triggers it. Untagged transitions are invalid.
- Self-loops are OK: connect(state, state, "tick") for an event that doesn't leave the current state.
- 4+ states; each state must have at least one incoming or outgoing transition (no unreachable or dead states).
- Direction "LR" is recommended for state machines that flow over time; "TB" is acceptable for vertical layouts.

${SDK_TYPES}`;
