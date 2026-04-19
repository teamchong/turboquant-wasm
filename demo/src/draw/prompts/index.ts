/**
 * Prompt module exports.
 *
 * Each branch is a complete, self-contained system prompt: preamble +
 * its type's docs + full SDK_TYPES. Exactly ONE branch is active as
 * the live system cache at any moment. The router is the boot branch;
 * the model's setType(...) call triggers a mid-generation swap to the
 * chosen type's branch via engine.mountKV.
 *
 * BRANCHES is the canonical ordered map — the build pipeline iterates
 * it in this order, main.ts tokenizes every entry, and the worker
 * registerBranch's each one so later mounts don't reparse any blob.
 */

export { ROUTER_PROMPT } from "./router.js";
export { SEQUENCE_PROMPT } from "./sequence.js";
export { ARCHITECTURE_PROMPT } from "./architecture.js";
export { FLOWCHART_PROMPT } from "./flowchart.js";
export { STATE_PROMPT } from "./state.js";
export { ORGCHART_PROMPT } from "./orgchart.js";
export { ER_PROMPT } from "./er.js";
export { CLASS_PROMPT } from "./class.js";
export { SWIMLANE_PROMPT } from "./swimlane.js";
export { PREAMBLE, BRIEF_THINKING_TAIL } from "./preamble.js";

import { ROUTER_PROMPT } from "./router.js";
import { SEQUENCE_PROMPT } from "./sequence.js";
import { ARCHITECTURE_PROMPT } from "./architecture.js";
import { FLOWCHART_PROMPT } from "./flowchart.js";
import { STATE_PROMPT } from "./state.js";
import { ORGCHART_PROMPT } from "./orgchart.js";
import { ER_PROMPT } from "./er.js";
import { CLASS_PROMPT } from "./class.js";
import { SWIMLANE_PROMPT } from "./swimlane.js";

export const BRANCHES = {
  router: ROUTER_PROMPT,
  sequence: SEQUENCE_PROMPT,
  architecture: ARCHITECTURE_PROMPT,
  flowchart: FLOWCHART_PROMPT,
  state: STATE_PROMPT,
  orgchart: ORGCHART_PROMPT,
  er: ER_PROMPT,
  class: CLASS_PROMPT,
  swimlane: SWIMLANE_PROMPT,
} as const;

export type BranchName = keyof typeof BRANCHES;
