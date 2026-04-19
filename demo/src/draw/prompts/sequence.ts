/**
 * Sequence-mode branch.
 *
 * Mounted after the grammar recognises `setType("sequence")`. Contains:
 *   - restated preamble (so the branch is self-contained on mount)
 *   - sequence-specific example
 *   - sequence-specific rules
 *   - full SDK type declarations (the model still needs the full typedef)
 *
 * Does NOT mention architecture / swimlane / UML / ER — those are in the
 * architecture branch, out of scope here.
 */

import { PREAMBLE, BRIEF_THINKING_TAIL } from "./preamble.js";
import { SDK_TYPES } from "../drawmode/sdk-types.js";

export const SEQUENCE_PROMPT = `${PREAMBLE}${BRIEF_THINKING_TAIL}

You are in SEQUENCE mode (setType("sequence") has been called). The SDK for
this mode is addActor(...) + message(from, to, label). Do NOT use addBox,
addEllipse, addDiamond, addTable, addClass, addGroup, or addLane in this
mode — they are invalid here.

SEQUENCE example:
setType("sequence");
const browser = addActor("Browser");
const dns = addActor("DNS");
const api = addActor("API");
const db = addActor("Database");
message(browser, dns, "Resolve example.com");
message(dns, browser, "93.184.216.34");
message(browser, api, "GET /users");
message(api, db, "SELECT * FROM users");
message(db, api, "rows");
message(api, browser, "200 OK");

Sequence-specific rules:
- Use addActor for every participant (one per column). NEVER addBox in a sequence diagram.
- message(from, to, "label") in chronological order — order in code = order in time, top to bottom.
- Each message must have a label. Include request AND response as separate messages, plus error paths.
- 8+ messages, 3+ actors. NO row/col/color/icon on actors — layout is automatic.
- Every addActor result must be assigned to a const and referenced by at least one message.

${SDK_TYPES}`;
