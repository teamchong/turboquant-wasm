/**
 * Prompt module exports.
 *
 *   ROUTER_PROMPT         — mode picker, mounted at startup
 *   SEQUENCE_PROMPT       — sequence-mode branch
 *   ARCHITECTURE_PROMPT   — architecture-mode branch (covers flowchart,
 *                           class, ER, swimlane, state machine, org chart)
 *   BRANCHES              — map of branch name → prompt string, used by
 *                           the build pipeline (tests/build-cache.spec.ts)
 *
 * Legacy combined prompt (the current main.ts SYSTEM_PROMPT) is not
 * re-exported here — main.ts keeps its own definition until the mount
 * infrastructure lands and swaps it out.
 */

export { ROUTER_PROMPT } from "./router.js";
export { SEQUENCE_PROMPT } from "./sequence.js";
export { ARCHITECTURE_PROMPT } from "./architecture.js";
export { PREAMBLE } from "./preamble.js";

import { ROUTER_PROMPT } from "./router.js";
import { SEQUENCE_PROMPT } from "./sequence.js";
import { ARCHITECTURE_PROMPT } from "./architecture.js";

/** Named branches for the build pipeline. Order is stable — used as the
 *  canonical iteration order when writing the multi-branch cache file. */
export const BRANCHES = {
  router: ROUTER_PROMPT,
  sequence: SEQUENCE_PROMPT,
  architecture: ARCHITECTURE_PROMPT,
} as const;

export type BranchName = keyof typeof BRANCHES;
