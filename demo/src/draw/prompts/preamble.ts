/**
 * Shared preamble used by every mode branch (and the router).
 *
 * Kept as a separate string so all branches open identically — model sees
 * the same "this is how you talk to the SDK" boilerplate regardless of
 * which branch is mounted.
 */
export const PREAMBLE = `Reply with JavaScript SDK calls only. Start with the first call, end with the last call.

Call SDK methods as globals: addBox(...), connect(...), addActor(...), etc. One statement per line. Assign every addXxx result to a const and pass that const to connect/message/addGroup/addLane. Every from/to argument of connect and message MUST be a const from addXxx — never a raw string.`;

/**
 * Optional "keep thinking short" tail. Router + sequence mount this;
 * architecture deliberately does NOT — architecture diagrams need
 * thorough planning before emitting 15+ nodes with row/col/color/icon
 * assignments, and truncating the thinking phase to ~3 sentences
 * produced incoherent code (hallucinated undeclared vars, mis-placed
 * option keys). Documented in 2026-04-18 session log.
 */
export const BRIEF_THINKING_TAIL = `\n\nThink briefly: one short paragraph, ≤3 sentences, plain text only. No markdown (no headers, no bullets, no numbered lists, no code fences, no bold/italic). No step-by-step breakdowns in thinking — save the structure for the actual SDK calls.`;
