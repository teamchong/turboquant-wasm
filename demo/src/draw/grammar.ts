/**
 * Constrained decoding grammar for the draw DSL.
 *
 * The model tends to emit broken code like:
 *   connect(User Input, DNS Lookup, "DNS Query");
 *   connect(HTTP Request", "Server", "Send Request");
 *
 * Two failure modes we catch:
 *
 *   (a) Identifier-with-space inside a call arg: `User Input` — the model
 *       tries to use a multi-word label as a variable reference, which is
 *       invalid JS. Fix: once an identifier starts inside `(...)`, mask
 *       any `letter` that follows a ` ` (without a separator in between).
 *
 *   (b) Unclosed string literal: a `"` that never closes before `\n` or a
 *       structural char. Fix: inside `"..."`, mask tokens containing
 *       `\n`, `,`, `;`, `)`, etc. before a closing `"`.
 *
 * The state machine tracks three contexts (statement / paren-call / object)
 * plus in-string sub-states. Object-literal contexts deliberately relax
 * identifier rules — `{ row: 1, col: 2 }` has spaces and colons that would
 * false-positive the identifier-space rule.
 *
 * State bytes (fits in u8):
 *   0 FREE              — outside parens/strings/objects (statement level)
 *   1 STRING_FREE       — "..." opened from FREE
 *   2 PAREN_NEUTRAL     — inside (...), arg-slot empty or just after `,`
 *   3 PAREN_IDENT       — inside (...), mid-identifier
 *   4 PAREN_IDENT_SPACE — inside (...), ident emitted then ` ` seen
 *   5 STRING_PAREN      — "..." opened from inside (...)
 *   6 OBJECT            — inside {...}, relaxed rules
 *   7 STRING_OBJECT     — "..." opened from inside {...}
 */

export const NUM_STATES = 9;
export const S_FREE = 0;
export const S_STRING_FREE = 1;
export const S_PAREN_NEUTRAL = 2;
export const S_PAREN_IDENT = 3;
export const S_PAREN_IDENT_SPACE = 4;
export const S_STRING_PAREN = 5;
export const S_OBJECT = 6;
export const S_STRING_OBJECT = 7;
// Thinking-mode pass-through. Gemma 4 E2B's thinking channel emits free
// natural-language reasoning before the actual code. The JS-syntax grammar
// would mask all of it — so during the thinking phase we pass every token
// and only transition to S_FREE when the close-channel marker appears in
// the decoded text.
export const S_IN_THINK = 8;

// Alias kept so existing imports (engine-worker) resolve.
export const STATE_FREE = S_FREE;

// Close-of-thinking marker. When a token's decoded text contains this
// substring, we leave S_IN_THINK and enter S_FREE (code-writing mode).
const CHANNEL_CLOSE = "<channel|>";

// Chars that would break an in-flight JS string literal. Tokens inside a
// string must NOT contain any of these before the closing `"`.
const BREAK_CHARS_IN_STRING = new Set(["\n", "\r", "\t", ",", ";", "(", ")", "{", "}"]);

function isIdentStart(c: string): boolean {
  return (c >= "A" && c <= "Z") || (c >= "a" && c <= "z") || c === "_" || c === "$";
}

function isIdentCont(c: string): boolean {
  return isIdentStart(c) || (c >= "0" && c <= "9");
}

function isBlank(c: string): boolean {
  return c === " " || c === "\t";
}

function simulateToken(startState: number, text: string): { allowed: boolean; endState: number } {
  let state = startState;
  if (text.length === 0) return { allowed: true, endState: state };

  // Thinking-mode pass-through: accept every token without JS-syntax checks.
  // Flip to S_FREE (code mode) as soon as a token decodes to text containing
  // the close-channel marker — the subsequent tokens are real diagram code
  // and should be validated normally.
  if (state === S_IN_THINK) {
    if (text.includes(CHANNEL_CLOSE)) {
      return { allowed: true, endState: S_FREE };
    }
    return { allowed: true, endState: S_IN_THINK };
  }

  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    const escaped = i > 0 && text[i - 1] === "\\";

    switch (state) {
      case S_FREE:
        if (c === '"' && !escaped) state = S_STRING_FREE;
        else if (c === "(") state = S_PAREN_NEUTRAL;
        // Everything else stays in FREE — `const foo = bar` should pass.
        break;

      case S_STRING_FREE:
        if (c === '"' && !escaped) state = S_FREE;
        else if (BREAK_CHARS_IN_STRING.has(c)) return { allowed: false, endState: startState };
        break;

      case S_PAREN_NEUTRAL:
        if (c === ")") state = S_FREE;
        else if (c === ",") state = S_PAREN_NEUTRAL;
        else if (c === '"' && !escaped) state = S_STRING_PAREN;
        else if (c === "{") state = S_OBJECT;
        else if (c === "(") state = S_PAREN_NEUTRAL;
        else if (isIdentStart(c)) state = S_PAREN_IDENT;
        // Whitespace / digits / operators: stay neutral.
        break;

      case S_PAREN_IDENT:
        if (c === ")") state = S_FREE;
        else if (c === ",") state = S_PAREN_NEUTRAL;
        else if (isBlank(c)) state = S_PAREN_IDENT_SPACE;
        else if (isIdentCont(c)) state = S_PAREN_IDENT;
        else if (c === '"' && !escaped) state = S_STRING_PAREN;
        else if (c === "(") state = S_PAREN_NEUTRAL;
        else if (c === "{") state = S_OBJECT;
        else state = S_PAREN_NEUTRAL;
        break;

      case S_PAREN_IDENT_SPACE:
        if (c === ")") state = S_FREE;
        else if (c === ",") state = S_PAREN_NEUTRAL;
        else if (isBlank(c)) state = S_PAREN_IDENT_SPACE;
        // Identifier continuation after space means the model started a
        // second word in the same arg — this is the `User Input` bug.
        else if (isIdentCont(c)) return { allowed: false, endState: startState };
        else if (c === '"' && !escaped) state = S_STRING_PAREN;
        else if (c === "(") state = S_PAREN_NEUTRAL;
        else if (c === "{") state = S_OBJECT;
        else state = S_PAREN_NEUTRAL;
        break;

      case S_STRING_PAREN:
        if (c === '"' && !escaped) state = S_PAREN_NEUTRAL;
        else if (BREAK_CHARS_IN_STRING.has(c)) return { allowed: false, endState: startState };
        break;

      case S_OBJECT:
        if (c === "}") state = S_PAREN_NEUTRAL;
        else if (c === '"' && !escaped) state = S_STRING_OBJECT;
        // Inside objects we deliberately relax identifier tracking —
        // `{ row: 1, col: 2 }` has spaces and colons that would otherwise
        // false-positive the ident-space rule.
        break;

      case S_STRING_OBJECT:
        if (c === '"' && !escaped) state = S_OBJECT;
        else if (BREAK_CHARS_IN_STRING.has(c)) return { allowed: false, endState: startState };
        break;
    }
  }
  return { allowed: true, endState: state };
}

/**
 * Precompute grammar bitmaps + per-(state, token) transition table.
 *
 * Yields to the browser every `chunkSize` tokens so the vocab-scale decode
 * doesn't freeze the UI. For Gemma 4 E2B (vocab=262144, NUM_STATES=8) the
 * build produces ~256 KB of bitmaps + 2 MB of transitions.
 */
export async function buildGrammar(
  decode: (id: number) => string,
  vocabSize: number,
  chunkSize = 4096,
): Promise<{ masks: Uint32Array; transitions: Uint8Array }> {
  const words = Math.ceil(vocabSize / 32);
  const masks = new Uint32Array(NUM_STATES * words);
  const transitions = new Uint8Array(NUM_STATES * vocabSize);

  for (let base = 0; base < vocabSize; base += chunkSize) {
    const end = Math.min(base + chunkSize, vocabSize);
    for (let id = base; id < end; id++) {
      let text = "";
      try {
        text = decode(id);
      } catch {
        // Some ids (reserved/unused) fail to decode — treat as empty.
      }

      for (let s = 0; s < NUM_STATES; s++) {
        const { allowed, endState } = simulateToken(s, text);
        if (allowed) {
          masks[s * words + (id >>> 5)] |= (1 << (id & 31));
        }
        transitions[s * vocabSize + id] = endState;
      }
    }
    await new Promise((r) => setTimeout(r, 0));
  }
  return { masks, transitions };
}

/** Runtime state holder that advances via the precomputed transition table. */
export class GrammarState {
  state: number = S_FREE;

  constructor(private transitions: Uint8Array, private vocabSize: number) {}

  reset(): void { this.state = S_FREE; }

  advance(tokenId: number): void {
    this.state = this.transitions[this.state * this.vocabSize + tokenId];
  }
}

// =============================================================================
// SDK-level state (orthogonal to char-level grammar states above)
//
// The char-level grammar validates JS syntax (quotes balanced, identifiers
// don't contain spaces, etc.) but does not know which SDK functions are
// available in which diagram mode. We track that separately via ModeTracker:
// a side-channel that watches decoded text as the model emits it and flips
// SDK state when it recognises a completed `setType("...")` call.
//
// Why orthogonal, not nested into the char-level state table: nesting would
// 3× the mask + transition tables (9 char states × 3 SDK states) for a
// feature that only needs to fire once per generation. A side-channel tracks
// O(emitted_text_len) characters for the single `setType(...)` pattern match
// and is trivially cheap.
// =============================================================================

export const SDK_MODE_UNSET = 0;
export const SDK_MODE_SEQUENCE = 1;
export const SDK_MODE_ARCHITECTURE = 2;

export type SdkMode =
  | typeof SDK_MODE_UNSET
  | typeof SDK_MODE_SEQUENCE
  | typeof SDK_MODE_ARCHITECTURE;

/**
 * Tracks SDK-level state by scanning decoded text for completed
 * `setType("...")` calls. One instance per generation.
 *
 * v1 only fires one transition (UNSET → SEQUENCE | ARCHITECTURE). Further
 * setType calls are ignored — our grammar will mask them, but even if a
 * second one slipped through we don't act on it.
 *
 * Pattern matches `setType("sequence")` or `setType("architecture")` with
 * flexible whitespace. The value must be exactly one of those two strings
 * — anything else is ignored and SDK state stays UNSET.
 */
const SET_TYPE_RE = /setType\s*\(\s*"(sequence|architecture)"\s*\)/;

export type ModeAction = (mode: SdkMode) => void;

export class ModeTracker {
  private buffer = "";
  private _mode: SdkMode = SDK_MODE_UNSET;
  private onEnterHandlers = new Map<SdkMode, ModeAction[]>();

  /** Register a callback that fires when SDK state transitions into `mode`.
   *  Multiple handlers per mode are allowed; they run in registration order.
   *  Handlers run synchronously — async work must be scheduled by the handler
   *  itself (e.g. via queue.submit on a GPU device).
   *
   *  v1 only fires onEnter for SEQUENCE / ARCHITECTURE. UNSET is never
   *  "entered" (it's the initial state + the post-reset state). */
  onEnter(mode: SdkMode, fn: ModeAction): void {
    if (mode === SDK_MODE_UNSET) {
      throw new Error("onEnter(UNSET) never fires — use reset() for that");
    }
    const list = this.onEnterHandlers.get(mode);
    if (list) list.push(fn);
    else this.onEnterHandlers.set(mode, [fn]);
  }

  /** Feed the latest decoded token text. Returns true iff SDK state
   *  just transitioned into a non-UNSET mode on this call. If a transition
   *  happens, registered onEnter handlers for the new mode fire before
   *  this method returns. */
  observe(text: string): boolean {
    if (this._mode !== SDK_MODE_UNSET) return false;
    if (text.length === 0) return false;
    this.buffer += text;
    const m = SET_TYPE_RE.exec(this.buffer);
    if (!m) return false;
    this._mode = m[1] === "sequence" ? SDK_MODE_SEQUENCE : SDK_MODE_ARCHITECTURE;
    this.buffer = "";
    const handlers = this.onEnterHandlers.get(this._mode);
    if (handlers) for (const fn of handlers) fn(this._mode);
    return true;
  }

  /** Reset SDK state to UNSET and clear the scan buffer. Does NOT
   *  unregister handlers — they persist across generations. */
  reset(): void {
    this.buffer = "";
    this._mode = SDK_MODE_UNSET;
  }

  /** Unregister all handlers. Useful for tests; production code typically
   *  registers once at construction and relies on reset() to re-arm. */
  clearHandlers(): void {
    this.onEnterHandlers.clear();
  }

  get mode(): SdkMode { return this._mode; }
}
