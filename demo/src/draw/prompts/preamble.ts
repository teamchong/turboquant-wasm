/**
 * Shared preamble used by every mode branch (and the router).
 *
 * Kept as a separate string so all branches open identically — model sees
 * the same "this is how you talk to the SDK" boilerplate regardless of
 * which branch is mounted.
 */
export const PREAMBLE = `Reply with JavaScript SDK calls only. Start with the first call, end with the last call.

Call SDK methods as globals: addBox(...), connect(...), addActor(...), etc. One statement per line. Assign every addXxx result to a const and pass that const to connect/message/addGroup/addLane. Every from/to argument of connect and message MUST be a const from addXxx — never a raw string.`;
