/**
 * Router prompt — the "mode picker" branch.
 *
 * Small on purpose. Gives the model just enough to choose between the two
 * diagram modes and emit the correct `setType(...)` call. Once the grammar
 * sees that call, it swaps in the full mode-specific branch (sequence.ts
 * or architecture.ts) via engine.mountKV(), so the model continues
 * generation with rich mode-specific context.
 *
 * Model MUST emit setType(...) as the first call — grammar enforces this
 * by masking anything else until setType(...) is complete.
 */

import { PREAMBLE } from "./preamble.js";

export const ROUTER_PROMPT = `${PREAMBLE}

FIRST line MUST be setType("..."). Pick one:

  setType("sequence")     — chronological message flow between participants.
                            Use this ONLY when the user asks for a sequence /
                            message / call-flow / swimlane-of-time diagram.
                            Uses addActor(...) + message(from, to, label).

  setType("architecture") — everything else. Boxes connected by lines with
                            rows/columns/colors/icons. Covers: infrastructure,
                            flowcharts, state machines, ER, UML class diagrams,
                            swimlanes, org charts.

After setType(...), the full SDK for that mode becomes available. Keep every
node real and distinct; connect every node to at least one other node.`;
