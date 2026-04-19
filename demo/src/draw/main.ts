/** Prompt to Diagram: Gemma 4 E2B with custom WebGPU engine + TQ KV cache.
 *  UI thread — tokenizer + rendering. All GPU work runs in engine-worker.ts. */

import { AutoTokenizer } from "@huggingface/transformers";
import { executeCode } from "./drawmode/executor.js";
import { SDK_TYPES } from "./drawmode/sdk-types.js";
import { BRANCHES, type BranchName, ROUTER_PROMPT } from "./prompts/index.js";
import { parseSystemCache } from "./system-cache-container.js";
import { loadWasm } from "./drawmode/layout.js";
import drawmodeWasm from "./drawmode/drawmode-wasm.js";
import { mountExcalidraw, updateDiagram, resetDiagram, fitToScreen, enterEditMode, exitEditMode, getMode, showThinkingCloud, clearThinkingCloud } from "./excalidraw-viewer.js";
import { createEditor, setCode, appendCode, getCode } from "./code-editor.js";
import EngineWorker from "./engine-worker.ts?worker";
import {
  buildGrammar,
  NUM_STATES,
  ModeTracker,
  SDK_MODE_ARCHITECTURE,
  SDK_MODE_SEQUENCE,
  SDK_MODE_FLOWCHART,
  SDK_MODE_STATE,
  SDK_MODE_ORGCHART,
  SDK_MODE_ER,
  SDK_MODE_CLASS,
  SDK_MODE_SWIMLANE,
} from "./grammar.js";

const TOKENIZER_ID = "onnx-community/gemma-4-E2B-it-ONNX";
const GGUF_URL = "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf";

// Minimal Gemma 4 E2B (gemma3n) chat template. IMPORTANT: uses `<|turn>` and
// `<turn|>` — the tokens this model was actually trained with. Using Gemma 2's
// `<start_of_turn>` / `<end_of_turn>` makes the model see unfamiliar tokens
// and it emits garbage or end-of-turn immediately.
//
// Thinking mode — force the model to start every response inside the
// thought channel by seeding the generation prompt with
// '<|turn>model\n<|channel>thought\n'. The `<|think|>` marker alone at the
// system turn wasn't enough: the model would emit a bare `<|channel>` and
// immediately EOS, producing no content (diagnostic confirmed 2026-04-17
// with attempt 1 tokCount=1, thinkingBuffer='<|channel>'). Seeding the
// full channel opener leaves the model in "reasoning" territory at the
// first predicted token, so it streams real thought text until it emits
// '<channel|>' and transitions to code.
//
// The streaming filter in generate() hides reasoning from the code editor
// and grammar.ts's S_IN_THINK state suppresses JS-syntax masking until the
// channel closes.
const CHAT_TEMPLATE = `{{ bos_token }}{%- if messages[0]['role'] in ['system', 'developer'] -%}{{- '<|turn>system\\n' -}}{{- messages[0]['content'] | trim -}}{{- '<turn|>\\n' -}}{%- set loop_messages = messages[1:] -%}{%- else -%}{%- set loop_messages = messages -%}{%- endif -%}{%- for message in loop_messages -%}{%- set role = message['role'] -%}{%- if role == 'assistant' -%}{%- set role = 'model' -%}{%- endif -%}{{- '<|turn>' + role + '\\n' -}}{{- message['content'] | trim -}}{{- '<turn|>\\n' -}}{%- endfor -%}{%- if add_generation_prompt -%}{{- '<|turn>model\\n<|channel>thought\\n' -}}{%- endif -%}`;

// Gemma 4 E2B (gemma3n) stop tokens:
//   1   = <eos> (end of sequence)
//   106 = <turn|> (end of model turn)
// NOTE: 107 is plain `\n`, not any turn marker — including it here used to
// stop generation at the very first newline in the output, cutting every
// response off after one line of code.
const EOS_TOKENS = new Set([1, 106]);

// Vocab size for Gemma 4 E2B. Matches VOCAB_SIZE in engine.ts.
const VOCAB_SIZE = 262144;

const $ = (s: string) => document.querySelector(s)!;
const statusEl = $("#status") as HTMLElement;
const promptEl = $("#prompt") as HTMLTextAreaElement;
const generateBtn = $("#generate") as HTMLButtonElement;
const codeArea = $("#code-area") as HTMLElement;
const diagramContainer = $("#diagram-container") as HTMLElement;
const renderBtn = $("#render") as HTMLButtonElement;
const wipeBtn = $("#wipe") as HTMLButtonElement;
const editBtn = $("#edit-diagram") as HTMLButtonElement;
const statSpeed = $("#stat-speed") as HTMLElement;
const statKV = $("#stat-kv") as HTMLElement;

let tokenizer: any = null;
let worker: Worker | null = null;
let modelReady = false;
let busy = false;
let aborted = false;
let currentCode = "";
let lastStmtCount = 0;
let prefillDebounce: ReturnType<typeof setTimeout> | null = null;
let lastPrefillText = "";
let userPrefilled = false;
let eagerFirstToken = 0;
// Snapshot of the token sequence prefilled by the most recent eager prefill,
// so the generate() fast-path can hand it to currentAttemptPrefilled for
// retry-LCP without re-tokenising.
let lastEagerPrefilledTokens: number[] | null = null;
// Set while doUserPrefill is in flight so generate() can await it instead of
// stacking a second prefill on top — that double-post used to race the
// engine's _stagingBuf.mapAsync (only one outstanding map allowed per buffer).
let eagerPrefillPromise: Promise<void> | null = null;
// Token IDs currently in the KV cache past the system-snapshot boundary.
// Updated after each successful prefill. Lets the next generate() rollbackKV
// to the longest common prefix with the new token sequence instead of the
// full restoreCache → re-prefill path — saves most of the ~1-2s re-prefill
// cost on iterative prompts like "draw X" followed by "draw X with 5 nodes".
let cachedUserTokens: number[] | null = null;
// Resolves when the in-flight generate() call finishes its finally block.
// Second click on the Generate button aborts the previous run and awaits this
// before starting a new one — without this, the abort + restart race the
// worker (double prefill, double stream, GPU buffer contention).
let currentGeneration: Promise<void> | null = null;
// Tokens prefilled for the CURRENT attempt inside the active generate() call.
// Used by the retry branch to compute LCP against the new (post-failure)
// conversation's tokenIds — the retry adds {assistant: code, user: error}
// turns on top of the original user prompt, so the first user-turn portion
// matches and can be skipped via rollbackKV. Cleared when generate() exits.
let currentAttemptPrefilled: number[] | null = null;
// Position where the system-prompt snapshot ends. Initialised at init-time
// once we know systemTokenIds. LCP rollback targets are computed as
// `systemCacheEnd + lcpLen`.
let systemCacheEnd = 0;
// Minimum shared prefix length to use the rollback fast-path. Below this the
// savings don't outweigh the rollback (worker round-trip ~1-2ms) + suffix-
// prefill overhead. Set to 8: the rollbackKV worker call is one postMessage
// round-trip (~1ms) and each token in the reused prefix saves ~0.8ms of
// prefill work, so 8 tokens is roughly breakeven and anything above is pure
// win. Lowered from the original 32 after the retry-LCP change — short user
// prompts (<32 tokens) were missing the retry fast-path despite having 25+
// tokens of reusable prefix.
const PREFIX_REUSE_MIN = 8;

const SYSTEM_PROMPT = `Reply with JavaScript SDK calls only. Start with the first call, end with the last call.

Call SDK methods as globals: addBox(...), connect(...), addActor(...), etc. One statement per line. Assign every addXxx result to a const and pass that const to connect/message/addGroup/addLane. Every from/to argument of connect and message MUST be a const from addXxx — never a raw string.

TYPES — pick based on the user's words. For SEQUENCE, call setType("sequence") FIRST (the first line). For direction, call setDirection("LR") etc. as the second line if needed.
1. ARCHITECTURE (default): addBox with row/col/color/icon, connect, addGroup
2. SEQUENCE: setType("sequence") then addActor, message(from, to, label)
3. FLOWCHART: addBox/addEllipse/addDiamond, connect with labels
4. STATE MACHINE: setDirection("LR") then addEllipse for states, connect with transition labels
5. ORG CHART: addBox hierarchy, connect
6. ER: addTable(name, [{name,type,key}], opts), connect with {cardinality:"1:N"}
7. UML CLASS: addClass(name, {attributes:[{name,type}], methods:[{name,type}]}, opts), connect with {relation:"inheritance"}
8. SWIMLANE: addBox for steps, addLane(name, [children]), connect

ARCHITECTURE example:
setType("architecture");
const browser = addBox("Browser", { row: 0, col: 0, color: "frontend", icon: "globe" });
const cdn = addBox("CDN", { row: 0, col: 2, color: "external", icon: "cloud" });
const lb = addBox("Load Balancer", { row: 1, col: 1, color: "orchestration", icon: "server" });
const api = addBox("API Server", { row: 2, col: 1, color: "backend", icon: "api" });
const cache = addBox("Redis", { row: 2, col: 0, color: "cache", icon: "cache" });
const db = addBox("Postgres", { row: 2, col: 2, color: "database", icon: "database" });
const queue = addBox("Job Queue", { row: 3, col: 1, color: "queue", icon: "queue" });
connect(browser, cdn, "GET /static");
connect(browser, lb, "HTTPS");
connect(lb, api, "round-robin");
connect(api, cache, "GET session");
connect(api, db, "SQL");
connect(api, queue, "enqueue job");
addGroup("Backend", [api, cache, db, queue]);

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

UML CLASS example:
setType("class");
const animal = addClass("Animal", {
  attributes: [{ name: "name", type: "string" }, { name: "age", type: "int" }],
  methods: [{ name: "eat()", type: "void" }, { name: "sleep()", type: "void" }],
});
const dog = addClass("Dog", {
  attributes: [{ name: "breed", type: "string" }],
  methods: [{ name: "bark()", type: "void" }],
});
const cat = addClass("Cat", {
  attributes: [{ name: "color", type: "string" }],
  methods: [{ name: "meow()", type: "void" }],
});
connect(dog, animal, "extends", { relation: "inheritance" });
connect(cat, animal, "extends", { relation: "inheritance" });

Pattern: every addBox/addActor/etc. is bound to a const on the same line.
Every connect/message/addGroup/addLane uses those consts (browser, api, db,
...) — never a string label, never a bare identifier that wasn't declared
above. For SWIMLANE, lane names are STRING arguments to addLane (not
standalone boxes): each lane is addLane("Lane Name", [stepConst, ...]).
The activity steps are addBox calls; each lane contains the boxes that
happen in that lane.

Rules:
- Each node must represent a distinct real entity. Do not duplicate.
- EVERY node you create (addBox/addEllipse/addDiamond/addActor/addTable/addClass) MUST be referenced by at least one connect, message, addGroup, or addLane. Orphan nodes (declared but never connected) are invalid — either wire them up or don't add them.
- Aim for 8+ nodes (15+ for architecture). Use \\n in labels.
- Architecture: EVERY node needs row, col, color, icon. Spread across columns 0-4.
- color: "frontend"|"backend"|"database"|"cache"|"queue"|"external"|"orchestration"|"users"|"storage"|"ai"
- icon: "server"|"database"|"lock"|"globe"|"users"|"api"|"cache"|"queue"|"cloud"|"code"|"shield"|"search"
- connect(from, to, "label") — string label only, no options object, no hex colors.
- 4+ groups for architecture diagrams.

Sequence-specific rules:
- Use addActor for every participant (one per column). NEVER addBox in a sequence diagram.
- message(from, to, "label") in chronological order — order in code = order in time, top to bottom.
- Each message must have a label. Include request AND response as separate messages, plus error paths.
- 8+ messages, 3+ actors. NO row/col/color/icon on actors — layout is automatic.

Flowchart / state machine / org chart rules:
- addEllipse for start/end/state nodes, addBox for action steps, addDiamond for decisions.
- For decisions, ALWAYS label both outgoing arrows ("yes"/"no", "true"/"false", or specific conditions).
- direction "TB" for flowcharts and org charts, "LR" for state machines that read left-to-right.

ER rules:
- Use addTable(name, columns) for every entity. NEVER addBox in an ER diagram.
- Each column is { name, type?, key? } where key is "PK" or "FK".
- Use { cardinality: "1:N" } on connect opts for foreign-key edges. Valid: "1:1", "1:N", "N:1", "N:M".
- Always include an "id" column with key: "PK" on every table. FK columns reference another table's PK.
- 4+ tables, every connect labelled with the relationship verb ("places", "owns", "has").

UML class rules:
- Use addClass(name, { attributes, methods }) for every class. NEVER addBox in a class diagram.
- attributes/methods are { name, type?, visibility? }. visibility: "public" (default), "private", "protected", "package".
- Methods go INSIDE the addClass call in the methods array — NEVER as connect() calls. connect is only for relationships between two classes.
- For inheritance: connect(child, parent, "extends", { relation: "inheritance" }). Valid relations: "inheritance", "composition", "aggregation", "dependency", "association".
- For composition (whole owns part): connect(whole, part, "owns", { relation: "composition" }).
- 4+ classes, every class with at least one attribute or method, every connect with a relation opt.

Swimlane rules:
- NEVER addActor in a swimlane. Lane names are the FIRST STRING ARGUMENT to addLane — they are not separate nodes.
- NEVER assign addLane(...) to a const. Lanes are containers, not nodes; their return value is NOT a valid arg to connect() or to another addLane([...]).
- connect(from, to, "label") — both from and to MUST be addBox-returned consts. NEVER pass a lane name (string) or a lane variable to connect.
- Add every activity step with addBox(label, { col, color, icon }) — use col only, NOT row (lane assigns row).
- After all boxes exist, group into lanes via addLane(name, [box1, box2, ...]). Lanes stack in declaration order (first = top). Each box goes in EXACTLY ONE lane — never list the same box in two addLane calls.
- Cross-lane arrows route vertical-then-horizontal automatically.
- 3+ lanes, 6+ activity boxes total, lane names = actor / system / responsibility ("User", "Server", "Database").

${SDK_TYPES}`;

// =============================================================================
// Worker communication
// =============================================================================

function workerCall<T>(msg: any, responseType: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const handler = (e: MessageEvent) => {
      const r = e.data;
      if (r.type === responseType) { worker!.removeEventListener("message", handler); resolve(r as T); }
      if (r.type === "error") { worker!.removeEventListener("message", handler); reject(new Error(r.message)); }
    };
    worker!.addEventListener("message", handler);
    worker!.postMessage(msg);
  });
}

// Scan rendered elements for node-shaped elements whose id isn't referenced
// by any arrow's customData._from/_to. Returns a list of labels for the
// human-readable retry feedback message, or [] if everything is connected.
//
// IMPORTANT: filter by Excalidraw element `type` — id prefix alone is not
// enough because bound labels share the node's id prefix (node id "box_5_X"
// has a bound text element "box_5_X-text" that matches ^(box|ell|…)_ but
// is a "text" element, not a node. Matching on id-only produces spurious
// orphans for every connected node whose label rendered through a bound
// text element.)
const NODE_TYPES = new Set(["rectangle", "ellipse", "diamond", "table", "class"]);
// Mechanical quick-fix: given generated SDK code + a list of orphan
// LABELS (first string arg to addXxx), strip the matching
// `const VAR = addXxx("LABEL", ...)` declarations AND any subsequent
// connect/message/addGroup/addLane calls that reference the dead VAR
// names. Produces a smaller-but-valid diagram without a full model
// regen. Runs before the retry path in generate(); if the result
// still looks broken we fall through to the regular retry.
function stripOrphanDeclarations(code: string, orphanLabels: string[]): string {
  if (orphanLabels.length === 0) return code;
  const labelSet = new Set(orphanLabels.map(s => s.trim()));
  const deadVars = new Set<string>();
  // Pass 1: find every `const VAR = addXxx("LABEL"` where LABEL is
  // orphaned; collect VAR names. Regex is permissive about optional
  // whitespace. Only matches the first string arg.
  const declRe = /^(\s*)(?:const|let|var)\s+(\w+)\s*=\s*add\w+\s*\(\s*"((?:[^"\\]|\\.)*)"/gm;
  let m: RegExpExecArray | null;
  while ((m = declRe.exec(code)) !== null) {
    const varName = m[2];
    const label = m[3];
    if (labelSet.has(label)) deadVars.add(varName);
  }
  if (deadVars.size === 0) return code;
  // Pass 2: drop whole lines that declare a dead var OR reference one
  // as a positional arg to connect/message/addGroup/addLane. Identifier
  // boundaries use \b so "user" doesn't match "userInput".
  const lines = code.split("\n");
  const deadVarPattern = [...deadVars].map(v => v.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|");
  const declLineRe = new RegExp(`^\\s*(?:const|let|var)\\s+(?:${deadVarPattern})\\s*=`);
  const refRe = new RegExp(`\\b(?:${deadVarPattern})\\b`);
  const kept: string[] = [];
  for (const line of lines) {
    if (declLineRe.test(line)) continue;     // drop the dead declaration
    if (refRe.test(line)) continue;          // drop any reference (connect/message/addGroup arg)
    kept.push(line);
  }
  return kept.join("\n");
}

// IDE-style quick fix for the OPPOSITE of orphan nodes: lines that
// REFERENCE identifiers the model never declared. Small models routinely
// emit `addGroup("Hardware", [foo, Bar, baz])` where Bar was declared as
// `const bar = addShape(...)` (case drift) or `connect(x, revenue, ...)`
// where `revenue` was never an addXxx-returned const at all. Running
// executeCode on that throws "<name> is not defined" — which kicks the
// retry loop.
//
// We instead pre-strip any line (connect / message / addGroup / addLane)
// whose positional args reference a bare identifier that doesn't appear
// on the LHS of a `const <name> = addXxx(...)` line earlier in the code.
// Other identifiers (string literals, option keys, option values) stay
// untouched — we only look at bare-word tokens inside the call-args.
//
// This is the codemod an IDE applies when it sees "symbol not defined":
// either rename or delete the offending reference. We delete, because
// a renamed-but-wrong reference would still compile and land a
// mis-connected edge.
function stripUndeclaredReferences(code: string): string {
  // Declared const names: LHS of any `const|let|var NAME = addXxx(...)`.
  const declRe = /^\s*(?:const|let|var)\s+(\w+)\s*=\s*add\w+\s*\(/gm;
  const declared = new Set<string>();
  let dm: RegExpExecArray | null;
  while ((dm = declRe.exec(code)) !== null) declared.add(dm[1]);
  if (declared.size === 0) return code;

  // Identifiers that appear in option-object keys / values across the
  // SDK — not variable references, so they shouldn't trip the check.
  // Kept small and conservative; anything else that's a real identifier
  // reference must have a `const` declaration to not get stripped.
  const optionIdents = new Set([
    "row", "col", "color", "icon", "style", "strokeColor",
    "startArrowhead", "endArrowhead", "labelPosition",
    "cardinality", "relation", "visibility",
    "name", "type", "key", "attributes", "methods",
    "fillStyle", "roughness", "strokeStyle", "strokeWidth",
    "direction", "theme", "opacity",
    "true", "false", "null", "undefined",
  ]);

  const callRe = /^\s*(connect|message|addGroup|addLane)\s*\(([\s\S]*?)\)\s*;?\s*$/;
  const lines = code.split("\n");
  const kept: string[] = [];
  for (const line of lines) {
    const m = callRe.exec(line);
    if (!m) { kept.push(line); continue; }
    // Strip string literals so we only see code tokens.
    const body = m[2].replace(/"(?:[^"\\]|\\.)*"/g, "");
    // Bare identifiers in remaining body.
    const idents = body.match(/\b([a-zA-Z_$][a-zA-Z0-9_$]*)\b/g) ?? [];
    const missing = idents.filter((id) =>
      !declared.has(id) && !optionIdents.has(id),
    );
    if (missing.length > 0) {
      console.log(`[draw] strip undeclared (${missing.join(",")}): ${line.trim()}`);
      continue;
    }
    kept.push(line);
  }
  return kept.join("\n");
}

function detectOrphanNodes(elements: any[]): string[] {
  const referenced = new Set<string>();
  for (const el of elements) {
    if (el.type !== "arrow") continue;
    const cd = el.customData || {};
    if (cd._from) referenced.add(cd._from);
    if (cd._to) referenced.add(cd._to);
  }
  // Nodes are shape elements (rectangle/ellipse/diamond/table/class).
  // Exclude the addGroup container (has no bound text), addFrame, addLane —
  // they're structural, not addressable edge endpoints.
  const nodes = elements.filter(el => NODE_TYPES.has(el.type) && !el.customData?._group);
  if (nodes.length <= 1) return [];
  const orphans: string[] = [];
  for (const n of nodes) {
    if (referenced.has(n.id)) continue;
    const bound = elements.find(e => e.type === "text" && e.containerId === n.id);
    orphans.push(bound?.text?.trim() || n.text?.trim() || n.id);
  }
  return orphans;
}

// Excalidraw text elements render plain text — so when the model emits
// markdown in the thinking channel (headers, bold, bullets, fenced code),
// the raw `**`, `##`, backticks etc. show up as literal characters. Strip
// the syntax and keep the payload so the thinking cloud is readable.
function stripMarkdown(md: string): string {
  return md
    // Fenced code blocks: drop the ``` fences, keep the body.
    .replace(/```[a-zA-Z0-9]*\n?/g, "")
    // Inline code: `x` → x
    .replace(/`([^`]+)`/g, "$1")
    // Headers: `### Foo` → `Foo`
    .replace(/^\s{0,3}#{1,6}\s+/gm, "")
    // Bold/strong: **x** or __x__ → x
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/__([^_]+)__/g, "$1")
    // Italic/em: *x* or _x_ → x (avoid eating bullet-style "* item" by requiring non-space after the *).
    .replace(/(^|[^*])\*([^*\s][^*]*?)\*/g, "$1$2")
    .replace(/(^|[^_])_([^_\s][^_]*?)_/g, "$1$2")
    // List bullets: `- item` / `* item` / `+ item` → `• item`
    .replace(/^(\s*)[-*+]\s+/gm, "$1• ")
    // Numbered lists: keep the number but drop trailing `)` if present → `1.` stays `1.`
    // Links: [text](url) → text
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    // Blockquote prefix
    .replace(/^\s{0,3}>\s+/gm, "");
}

// LRU-ish cache keyed by (full text, template flag) that memoises
// apply_chat_template + encode. Keystroke-driven eager prefill fires several
// times per second under debounce, and the tokenizer's encode path is
// ~30-50ms for a few-hundred-char prompt. Skipping re-tokenisation when the
// text hasn't changed shaves that cost off every coalesced keystroke.
const TOKEN_CACHE_MAX = 8;
const tokenCache = new Map<string, number[]>();
function tokenizeConversation(
  messages: Array<{ role: string; content: string }>,
  addGenerationPrompt: boolean,
): number[] {
  const text = tokenizer.apply_chat_template(messages, { tokenize: false, add_generation_prompt: addGenerationPrompt });
  const key = (addGenerationPrompt ? "g:" : "n:") + text;
  const cached = tokenCache.get(key);
  if (cached) {
    // LRU refresh: re-insert to move to newest position
    tokenCache.delete(key);
    tokenCache.set(key, cached);
    return cached;
  }
  const encoded = tokenizer.encode(text);
  const ids: number[] = Array.from(encoded);
  if (ids[0] === 2) ids.shift();  // strip BOS — SDK prepends it separately
  tokenCache.set(key, ids);
  while (tokenCache.size > TOKEN_CACHE_MAX) {
    const oldest = tokenCache.keys().next().value;
    if (oldest === undefined) break;
    tokenCache.delete(oldest);
  }
  return ids;
}

/** Open a streaming decode from the worker. `onToken` fires on every "token"
 *  event; the returned promise resolves on "streamDone" with the stop reason
 *  (`"eos" | "maxTokens" | "aborted" | "thinkingEnded"`) so the caller can
 *  orchestrate mid-stream work like reminder injection. */
function workerStream(
  msg: { type: "stream" | "streamConstrained"; tokenId: number; maxTokens: number; eosIds: number[]; startInThinking?: boolean },
  onToken: (t: { id: number; stats: any; profile: any }) => void,
): Promise<{ reason: string; lastTokenId: number }> {
  return new Promise((resolve, reject) => {
    const handler = (e: MessageEvent) => {
      const r = e.data;
      if (r.type === "token") onToken(r);
      else if (r.type === "streamDone") {
        worker!.removeEventListener("message", handler);
        resolve({ reason: r.reason ?? "eos", lastTokenId: r.lastTokenId ?? msg.tokenId });
      }
      else if (r.type === "error") { worker!.removeEventListener("message", handler); reject(new Error(r.message)); }
    };
    worker!.addEventListener("message", handler);
    worker!.postMessage(msg);
  });
}

// =============================================================================
// Incremental prefill — eagerly prefill user input as they type
// =============================================================================

function scheduleUserPrefill() {
  if (!modelReady || busy) return;
  if (prefillDebounce) clearTimeout(prefillDebounce);

  if (!promptEl.value.trim()) {
    userPrefilled = false;
    lastPrefillText = "";
    return;
  }

  prefillDebounce = setTimeout(doUserPrefill, 300);
}

async function doUserPrefill() {
  const prompt = promptEl.value.trim();
  if (!prompt || !modelReady || busy) return;
  // If a previous eager prefill is still running, skip this one — when it
  // finishes generate() will see the up-to-date prompt anyway. Coalescing
  // here avoids posting a second prefill that would race the first.
  if (eagerPrefillPromise) return;

  let textPart = prompt;
  const code = getCode();
  if (code) textPart = `Current diagram code:\n\`\`\`\n${code}\n\`\`\`\n\nModify it: ${prompt}`;

  if (textPart === lastPrefillText) return;
  lastPrefillText = textPart;

  const userTokenIds = tokenizeConversation(
    [{ role: "user", content: textPart }],
    true,
  );

  console.log("[draw] eager prefill:", userTokenIds.length, "tokens");
  let resolveEager!: () => void;
  eagerPrefillPromise = new Promise<void>((res) => { resolveEager = res; });
  try {
    worker!.postMessage({ type: "restoreCache" });
    const r = await workerCall<{ firstToken: number }>({ type: "prefill", tokenIds: userTokenIds }, "prefillDone");
    eagerFirstToken = r.firstToken;
    userPrefilled = true;
    cachedUserTokens = userTokenIds;
    lastEagerPrefilledTokens = userTokenIds;
    console.log("[draw] eager prefill done, first token:", eagerFirstToken);
  } catch {
    userPrefilled = false;
    cachedUserTokens = null;
    lastEagerPrefilledTokens = null;
  } finally {
    resolveEager();
    eagerPrefillPromise = null;
  }
}

// =============================================================================
// Generate
// =============================================================================

async function generate() {
  const prompt = promptEl.value.trim();
  if (!prompt) return;

  // Cancel-and-restart: second click aborts the in-flight generation, waits
  // for it to unwind (the stream loop checks `aborted` at every token + at
  // retry-loop boundaries), then falls through to start a fresh run on the
  // (possibly changed) prompt. Previously a second click only aborted —
  // forcing the user to click twice to actually retry.
  if (busy) {
    aborted = true;
    worker?.postMessage({ type: "abort" });
    if (currentGeneration) {
      generateBtn.innerHTML = '<span class="spinner"></span> Cancelling...';
      await currentGeneration;
    }
  }

  busy = true;
  aborted = false;
  let resolveCurrent!: () => void;
  currentGeneration = new Promise<void>((r) => { resolveCurrent = r; });

  if (!modelReady) {
    generateBtn.innerHTML = '<span class="spinner"></span> Waiting for model...';
    while (!modelReady) await new Promise(r => setTimeout(r, 500));
  }
  // If an eager prefill is still running for this same prompt, wait for it
  // to finish so we can reuse its KV state. Without this, generate() would
  // post a second prefill on top of the in-flight one and race the worker.
  if (eagerPrefillPromise) {
    console.log("[draw] generate: awaiting in-flight eager prefill");
    generateBtn.innerHTML = '<span class="spinner"></span> Finishing prefill...';
    await eagerPrefillPromise;
  }
  generateBtn.innerHTML = '<span class="spinner"></span> Generating...';
  setCode("");
  // If the user was editing the previous diagram, leave edit mode so the
  // streaming SVG renderer takes over again.
  if (getMode() === "live") exitEditMode();
  editBtn.style.display = "none";
  resetDiagram();
  statusEl.textContent = "Generating diagram code...";

  const startTime = performance.now();
  let tokenCount = 0;
  let generatedCode = "";
  lastStmtCount = 0;

  try {
    currentCode = getCode();
    let textPart = prompt;
    if (currentCode) {
      textPart = `Current diagram code:\n\`\`\`\n${currentCode}\n\`\`\`\n\nModify it: ${prompt}`;
    }

    // Conversation-style retry: if the executor reports a code error, append
    // the previous response + a "fix this: <error>" turn to the chat and
    // regenerate. Each retry restores the system-cache snapshot and prefills
    // the full conversation, so the model sees its own broken output AND the
    // exact error from the executor — closes the agent loop.
    const MAX_ATTEMPTS = 3;
    const conversation: Array<{ role: string; content: string }> = [
      { role: "user", content: textPart },
    ];
    // Reset per-generate() retry-rollback state. Populated after each
    // attempt's prefill so the NEXT attempt can LCP against it.
    currentAttemptPrefilled = null;

    // Separate per-phase budgets so a long reasoning turn doesn't starve
    // the code stream. Previously a single MAX_NEW_TOKENS=1024 was shared
    // across both phases — 800 tokens of thinking left only ~200 for code,
    // which truncated multi-node diagrams to things like `connect(null)`.
    //
    // Code budget: 4096 is safe for even the largest diagrams we generate
    // (15+ node architecture with addGroup calls, or UML class with
    // attributes/methods arrays). The stream exits on EOS (<turn|>, id=106)
    // the moment the model is done, so the budget only matters when the
    // model fails to terminate — at ~30 tok/s, worst case is ~140s but
    // typical case is unchanged.
    const MAX_THINKING_TOKENS = 1024;
    const MAX_CODE_TOKENS = 4096;
    // Two-phase stream:
    //   Phase A (S_IN_THINK) — free-form reasoning; rendered as an Excalidraw
    //     "thinking cloud" overlay, never written to the code editor.
    //   Phase B (S_FREE)    — SDK code; appended to the editor + run through
    //     the partial executor so the diagram grows live.
    // The worker signals "thinkingEnded" after the token containing
    // `<channel|>` lands; main then prefills a reminder instruction and
    // starts phase B from the model's first post-reminder token.
    let thinkingText = "";
    // Pure-decode rate tracking. statSpeed's end-to-end tok/s includes
    // prefill, thinking-cloud rendering, per-statement executeCode, and
    // Excalidraw layout time — all of which drag the headline number
    // below the engine's actual decode throughput. firstTokenTime marks
    // the moment the FIRST token arrives at main so we can compute
    // (N-1) / (now - firstTokenTime) as a closer proxy for worker-side
    // decode speed. Standard LLM runtimes (MLC, llama.cpp) report
    // roughly this metric, not end-to-end generate wall-clock.
    let firstTokenTime = 0;
    const updateSpeed = () => {
      const now = performance.now();
      const el = (now - startTime) / 1000;
      if (el > 0) {
        const e2e = tokenCount / el;
        if (firstTokenTime === 0 && tokenCount === 1) firstTokenTime = now;
        let label = `${e2e.toFixed(1)} tok/s`;
        if (firstTokenTime > 0 && tokenCount > 1) {
          const decodeEl = (now - firstTokenTime) / 1000;
          const decodeRate = (tokenCount - 1) / decodeEl;
          label = `${decodeRate.toFixed(1)} tok/s (decode) · ${e2e.toFixed(1)} e2e`;
        }
        statSpeed.textContent = label;
      }
    };
    const renderThinkingToken = (id: number) => {
      tokenCount++;
      updateSpeed();
      const raw = tokenizer.decode([id], { skip_special_tokens: false });
      if (!raw) return;
      thinkingText += raw;
      // Strip the `<channel|>` closer (and anything after it in the same
      // chunk) so the final thinking-cloud frame doesn't flash the marker
      // before we clear.
      const visible = thinkingText.replace(/<channel\|>.*$/s, "");
      if (visible) showThinkingCloud(stripMarkdown(visible));
      statusEl.textContent = `Thinking... ${tokenCount} tok`;
    };
    // Branch-phase leading setType suppression. Branch prompts say
    // "setType has been called" but the model often re-emits setType(...)
    // as its first code line anyway (the branch's own examples start
    // with setType). Since the editor was already seeded with the
    // canonical setType on mount, we drop the duplicate first line —
    // buffer branch chunks until the first newline, then decide:
    // - first line is setType(...) → discard, flush rest
    // - first line is anything else → flush whole buffer
    // Reset before each attempt + each post-mount run.
    let suppressLeadingSetType = false;
    let leadingLineBuffer = "";

    const normalizeBranchChunk = (chunk: string): string => {
      if (!suppressLeadingSetType) return chunk;
      leadingLineBuffer += chunk;
      const nl = leadingLineBuffer.indexOf("\n");
      if (nl < 0) {
        // Still within the first line — keep buffering, emit nothing.
        // Defensive cap: if the branch emits a very long first line
        // without newline we don't want to swallow the entire response.
        if (leadingLineBuffer.length > 256) {
          const out = leadingLineBuffer;
          leadingLineBuffer = "";
          suppressLeadingSetType = false;
          return out;
        }
        return "";
      }
      const firstLine = leadingLineBuffer.substring(0, nl);
      const rest = leadingLineBuffer.substring(nl + 1);
      leadingLineBuffer = "";
      suppressLeadingSetType = false;
      if (/^\s*setType\s*\(/.test(firstLine)) {
        return rest; // drop the duplicate setType line
      }
      return firstLine + "\n" + rest;
    };

    const renderCodeToken = (id: number) => {
      tokenCount++;
      updateSpeed();
      const raw = tokenizer.decode([id], { skip_special_tokens: true });
      if (!raw) return;
      const chunk = normalizeBranchChunk(raw);
      if (!chunk) return;
      generatedCode += chunk;
      appendCode(chunk);

      const stmts = (generatedCode.match(/;\s*\n/g) || []).length;
      if (stmts > lastStmtCount) {
        lastStmtCount = stmts;
        let partial = generatedCode.trim();
        partial = partial.replace(/^```(?:javascript|js|typescript|ts)?\s*\n?/i, "");
        const lastSemi = partial.lastIndexOf(";");
        if (lastSemi >= 0) {
          partial = partial.substring(0, lastSemi + 1);
          if (!partial.includes("new Diagram")) partial = `const d = new Diagram({ direction: "TB" });\n${partial}`;
          executeCode(partial).then(({ result }) => {
            if (result.json) updateDiagram(result.json.elements || []);
          });
        }
      }
    };

    // Post-thinking reminder. Encoded once per generate() call and prefilled
    // into KV between the thinking phase and the code phase so the model's
    // most-recent attention context is an explicit "emit code now"
    // instruction. The model sees it as its own recent text — mid-response
    // system turns would break the chat wire format, so we use bare prose.
    const REMINDER_TEXT = "\n\nNow emit the JavaScript SDK calls. No prose, no markdown, no explanation — start with the first SDK call:\n\n";
    const reminderEncoded = tokenizer.encode(REMINDER_TEXT);
    const reminderTokenIds: number[] = Array.from(reminderEncoded);
    if (reminderTokenIds[0] === 2) reminderTokenIds.shift();  // strip BOS if present

    // Thinking channel close marker. Prefilling this after the user turn
    // (which ends with `<|channel>thought\n` from the chat template) skips
    // the thinking phase entirely: the model enters code mode for its very
    // first token. Used on the ROUTER run — the model only needs to emit
    // setType(...) there; full planning-under-branch thinking happens in
    // the post-mount do-over.
    const CHANNEL_CLOSE_TEXT = "<channel|>";
    const channelCloseEncoded = tokenizer.encode(CHANNEL_CLOSE_TEXT);
    const channelCloseTokenIds: number[] = Array.from(channelCloseEncoded);
    if (channelCloseTokenIds[0] === 2) channelCloseTokenIds.shift();  // strip BOS if present

    let lastError: string | undefined;
    let finalResult: any = null;
    // Persists across attempts. On retry, we skip the router run entirely
    // and mount the previously-picked branch directly — the previous
    // attempt's conversation already contains `setType("<target>")`, so
    // the model on retry is unlikely to re-emit it, which used to leave
    // runCodePhaseOnly spinning with no tokens rendered and emitting an
    // empty retry. (Observed on ER: attempt 1 mounted er + syntax-errored,
    // attempts 2+3 ran under router, model didn't re-emit setType, all
    // tokens got silently discarded by the router observer.)
    let lastMountedBranch: BranchName | null = null;
    // Remember the most-substantive code any attempt produced so that if
    // MAX_ATTEMPTS all fail we can restore it to the editor on exit.
    // Previously every retry called setCode("") at its start; if a later
    // attempt produced less useful code than an earlier one (e.g. the
    // class test: attempt 1 had 689ch with one typo, attempt 2 had 18ch)
    // the user ended up with the worst attempt as their hand-edit starting
    // point. We pick the LONGEST version so there's the most context to
    // fix manually.
    let bestAttemptCode = "";

    for (let attempt = 0; attempt < MAX_ATTEMPTS && !aborted; attempt++) {
      // Prefill the full conversation onto the system-cache snapshot. First
      // attempt can reuse the eager prefill if the user prompt matches.
      let nextToken: number;
      if (attempt === 0 && userPrefilled && textPart === lastPrefillText) {
        console.log("[draw] using eager prefill, first token:", eagerFirstToken);
        nextToken = eagerFirstToken;
        // Capture the eager-prefilled token sequence so a retry can LCP
        // against it. doUserPrefill stores it on the module-scope cache.
        currentAttemptPrefilled = lastEagerPrefilledTokens;
      } else {
        const tokenIds = tokenizeConversation(conversation, true);
        const phase = attempt === 0 ? "Processing prompt" : `Fixing error (retry ${attempt}/${MAX_ATTEMPTS - 1})`;
        statusEl.innerHTML = `<span class="spinner"></span> ${phase}...`;

        // Prefix reuse — two sources of LCP:
        //   (a) attempt 0: cross-call reuse from the previous generate()'s
        //       `cachedUserTokens` (iterative prompts share a prefix).
        //   (b) attempt > 0 (retry): reuse the PREVIOUS attempt's prefilled
        //       tokens. The retry appends {assistant: code, user: error} to
        //       the conversation — the chat template's output up through the
        //       first `<|turn>model\n` matches attempt N's prefill exactly,
        //       so LCP ≈ prev_attempt_prefill_len − thought_template (~8 tok).
        //       Unique to the retry-loop architecture — nothing else in the
        //       browser LLM space has an error-feedback retry step to reuse.
        let lcp = 0;
        const lcpSource = attempt === 0 ? cachedUserTokens : currentAttemptPrefilled;
        if (lcpSource) {
          const maxLcp = Math.min(lcpSource.length, tokenIds.length - 1);
          while (lcp < maxLcp && lcpSource[lcp] === tokenIds[lcp]) lcp++;
          if (lcp < PREFIX_REUSE_MIN) lcp = 0;
        }

        if (lcp > 0) {
          console.log(`[draw] ${phase}: ${tokenIds.length} tokens (reusing prefix of ${lcp}, prefilling ${tokenIds.length - lcp})`);
          await workerCall<{ ok: true }>(
            { type: "rollbackKV", targetPosition: systemCacheEnd + lcp }, "rollbackKVDone",
          );
          const prefillResult = await workerCall<{ firstToken: number }>(
            { type: "prefill", tokenIds: tokenIds.slice(lcp) }, "prefillDone",
          );
          nextToken = prefillResult.firstToken;
        } else {
          console.log(`[draw] ${phase}: ${tokenIds.length} tokens`);
          worker!.postMessage({ type: "restoreCache" });
          const prefillResult = await workerCall<{ firstToken: number }>(
            { type: "prefill", tokenIds }, "prefillDone",
          );
          nextToken = prefillResult.firstToken;
        }
        cachedUserTokens = tokenIds;
        currentAttemptPrefilled = tokenIds;
      }
      userPrefilled = false;
      lastPrefillText = "";
      statusEl.textContent = attempt === 0 ? "Generating diagram code..." : `Regenerating (retry ${attempt}/${MAX_ATTEMPTS - 1})...`;

      // Reset per-attempt streaming state. Each attempt starts inside the
      // thinking channel — the template seeds `<|channel>thought\n` so the
      // first predicted token is reasoning text.
      generatedCode = "";
      tokenCount = 0;
      lastStmtCount = 0;
      thinkingText = "";
      setCode("");
      resetDiagram();

      const onStreamMsg = (msg: { id: number; stats: any }) => {
        if (msg.stats?.positions > 0) {
          const s = msg.stats;
          statKV.textContent = `KV: ${s.compressedMB.toFixed(1)}MB / ${s.uncompressedMB.toFixed(1)}MB (${s.ratio.toFixed(1)}x) · ${s.positions} pos`;
        }
      };

      // Dynamic-context mount detection. modeTracker scans decoded code-
      // phase text for `setType("sequence"|"architecture")`. When it
      // fires, pendingMount is set and we abort the stream so the caller
      // can swap the system-cache branch and re-run phase A + phase B
      // under the specialised context.
      //
      // One-shot per attempt: after the first mount we never swap again
      // in this attempt (ModeTracker's own fire-once semantics). If a
      // compile error triggers a retry, the outer attempt loop resets by
      // returning to the router snapshot via restoreCache + prefill.
      const modeTracker = new ModeTracker();
      let pendingMount: BranchName | null = null;
      // One handler per mode — each maps to its own branch with dense
      // per-type content. Adding a new DiagramType value requires adding
      // a handler here, a branch file in prompts/, an entry in BRANCHES,
      // and a constant + route in grammar.ts.
      modeTracker.onEnter(SDK_MODE_ARCHITECTURE, () => { pendingMount = "architecture"; });
      modeTracker.onEnter(SDK_MODE_SEQUENCE,     () => { pendingMount = "sequence"; });
      modeTracker.onEnter(SDK_MODE_FLOWCHART,    () => { pendingMount = "flowchart"; });
      modeTracker.onEnter(SDK_MODE_STATE,        () => { pendingMount = "state"; });
      modeTracker.onEnter(SDK_MODE_ORGCHART,     () => { pendingMount = "orgchart"; });
      modeTracker.onEnter(SDK_MODE_ER,           () => { pendingMount = "er"; });
      modeTracker.onEnter(SDK_MODE_CLASS,        () => { pendingMount = "class"; });
      modeTracker.onEnter(SDK_MODE_SWIMLANE,     () => { pendingMount = "swimlane"; });

      // Router-phase observer. Tokens emitted under the router branch are
      // thrown away — real code is generated under the mounted branch.
      // During the router run we ONLY feed decoded text to modeTracker
      // (to detect the setType) and tick the token / speed counters so
      // the UI shows progress. We deliberately do NOT:
      //   - append anything to the code editor
      //   - accumulate anything in generatedCode
      //   - trigger the partial-executor
      // Once mount fires, the editor is seeded with `setType("<target>");\n`
      // and branch tokens append normally via renderCodeToken from there.
      // Result: editor is never polluted by the partial / truncated router
      // setType that was causing the "setType flickers / disappears" bug.
      const observeForMount = (id: number): boolean => {
        tokenCount++;
        updateSpeed();
        const chunk = tokenizer.decode([id], { skip_special_tokens: true });
        if (chunk) modeTracker.observe(chunk);
        return pendingMount !== null;
      };

      // Router code phase: stream, observe each token, never render. The
      // moment observeForMount sees a completed setType(...) it sets
      // pendingMount and we abort the worker so the mount + do-over
      // flow takes over.
      const runCodePhaseOnly = async (initialToken: number): Promise<{ mountTarget: BranchName | null }> => {
        if (!EOS_TOKENS.has(initialToken)) {
          if (observeForMount(initialToken)) return { mountTarget: pendingMount };
        }

        await workerStream(
          {
            type: "streamConstrained",
            tokenId: initialToken,
            maxTokens: MAX_CODE_TOKENS,
            eosIds: [...EOS_TOKENS],
            startInThinking: false,
          } as any,
          (msg) => {
            if (aborted) return;
            if (EOS_TOKENS.has(msg.id)) return;
            if (observeForMount(msg.id)) {
              worker!.postMessage({ type: "abort" });
              return;
            }
            onStreamMsg(msg);
          },
        );

        return { mountTarget: pendingMount };
      };

      // Phase A (thinking) + reminder + Phase B (code). Used for the
      // post-mount do-over under the specialised branch: the model gets
      // rich per-type context, plans the actual diagram, then emits code.
      const runThinkingAndCode = async (initialToken: number): Promise<{ mountTarget: BranchName | null; phaseAReason: string }> => {
        if (!EOS_TOKENS.has(initialToken)) renderThinkingToken(initialToken);

        const phaseA = await workerStream(
          {
            type: "streamConstrained",
            tokenId: initialToken,
            maxTokens: MAX_THINKING_TOKENS,
            eosIds: [...EOS_TOKENS],
            startInThinking: true,
          } as any,
          (msg) => {
            if (aborted) return;
            if (EOS_TOKENS.has(msg.id)) return;
            renderThinkingToken(msg.id);
            onStreamMsg(msg);
          },
        );

        clearThinkingCloud();

        if (thinkingText) {
          const visible = thinkingText.replace(/<channel\|>.*$/s, "").trim();
          if (visible) console.log(`[draw] thinking (${thinkingText.length}ch):\n${visible}`);
        }

        if (aborted) return { mountTarget: null, phaseAReason: phaseA.reason };

        statusEl.textContent = "Generating diagram code...";
        const prefillTokens = phaseA.reason === "thinkingEnded"
          ? [phaseA.lastTokenId, ...reminderTokenIds]
          : [...reminderTokenIds];
        const prefillResult = await workerCall<{ firstToken: number }>(
          { type: "prefill", tokenIds: prefillTokens }, "prefillDone",
        );
        const codeStartToken = prefillResult.firstToken;

        // Observe the very first code token too — the model may have
        // emitted setType as its first post-thinking token.
        if (!EOS_TOKENS.has(codeStartToken)) {
          renderCodeToken(codeStartToken);
          modeTracker.observe(tokenizer.decode([codeStartToken], { skip_special_tokens: true }));
          if (pendingMount) {
            return { mountTarget: pendingMount, phaseAReason: phaseA.reason };
          }
        }

        await workerStream(
          {
            type: "streamConstrained",
            tokenId: codeStartToken,
            maxTokens: MAX_CODE_TOKENS,
            eosIds: [...EOS_TOKENS],
            startInThinking: false,
          } as any,
          (msg) => {
            if (aborted) return;
            if (EOS_TOKENS.has(msg.id)) return;
            const text = tokenizer.decode([msg.id], { skip_special_tokens: true });
            modeTracker.observe(text);
            if (pendingMount) {
              worker!.postMessage({ type: "abort" });
              return;
            }
            renderCodeToken(msg.id);
            onStreamMsg(msg);
          },
        );

        return { mountTarget: pendingMount, phaseAReason: phaseA.reason };
      };

      // Decide whether to run the router phase at all.
      //
      // Attempt 0 normal path: run router with skip-thinking, let the
      // model pick a mode via setType(...), then do-over on the branch.
      //
      // Retry path (attempt > 0 with a remembered branch from a prior
      // attempt): skip the router phase ENTIRELY. The previous attempt's
      // conversation already contains a committed setType("<target>"),
      // so the model is unlikely to re-emit setType on retry — running
      // the router observer would silently discard every token and
      // produce empty output (the ER failure mode). Mount the remembered
      // branch directly and run full thinking+code under it.
      statusEl.textContent = lastMountedBranch
        ? `Retrying under "${lastMountedBranch}"...`
        : "Picking diagram type...";

      let mountTarget: BranchName | null = null;
      let phaseAReasonFinal = "skipped-under-router";

      if (lastMountedBranch) {
        mountTarget = lastMountedBranch;
      } else {
        // FIRST RUN under router: skip thinking entirely. The chat template
        // left the cache at `...<|channel>thought\n`; prefilling <channel|>
        // + reminder closes the thinking channel and primes code mode, so
        // the model's first generated token IS the first code token. Saves
        // ~5s of wasted router thinking that would be discarded anyway.
        //
        // The nextToken returned by the user-turn prefill was a thinking
        // token (what the model would have emitted WITHOUT our follow-up
        // prefill). We discard it — that's one forward pass of wasted
        // compute, but vs re-running full phase A it's cheap.
        const routerSkipPrefill = [...channelCloseTokenIds, ...reminderTokenIds];
        const routerCodeStartResult = await workerCall<{ firstToken: number }>(
          { type: "prefill", tokenIds: routerSkipPrefill }, "prefillDone",
        );
        const routerCodeStartToken = routerCodeStartResult.firstToken;
        const routerResult = await runCodePhaseOnly(routerCodeStartToken);
        mountTarget = routerResult.mountTarget;
      }

      // Mount triggered? Do the full do-over on the specialised branch.
      // runThinkingAndCode already set pendingMount and aborted the stream;
      // we now mount the branch, re-prefill the user turn, and run the
      // thinking + code phases fresh.
      if (mountTarget && !aborted) {
        console.log(`[draw] mid-stream mount: swap router → "${mountTarget}", re-prefilling + redoing thinking+code`);
        // Router phase rendered nothing to the editor (observeForMount is
        // pure observation). Seed the editor now with the canonical
        // setType("<target>");\n — branch tokens will append below it
        // once the do-over starts streaming. This is the SOURCE OF TRUTH
        // for the visible code from this point on.
        //
        // generatedCode shadows editor content so the statement-boundary
        // partial-executor in renderCodeToken has something to count from.
        // It IS the editor content (plus whatever tokens arrive next).
        //
        // The branch's prompt says setType has been called, so the model
        // usually does NOT re-emit it — output appends below as expected.
        // If it does re-emit, the final dedupe step strips duplicate
        // setType lines before executeCode.
        const setTypeHeader = `setType("${mountTarget}");\n`;
        generatedCode = setTypeHeader;
        setCode(setTypeHeader);
        // Arm the branch-phase leading-setType suppressor: the branch will
        // typically re-emit setType(...) as its first code line (its
        // examples all start with it); we drop that duplicate during
        // streaming so the editor shows exactly one setType header
        // throughout. Disarmed automatically after the first branch-code
        // newline.
        suppressLeadingSetType = true;
        leadingLineBuffer = "";
        // DON'T reset modeTracker: its fire-once semantics are what keep
        // the second pass from aborting again. The model under the mounted
        // branch may re-emit `setType("sequence")` as its first code line
        // (the branch's examples show it); if the tracker were reset it
        // would re-fire and abort the stream mid-string, truncating the
        // code (observed pre-fix: 17ch of `setType("sequence` before EOS).
        // Keeping the tracker "already fired" makes observe return false
        // for the rest of the attempt.
        pendingMount = null;
        // Conversation tokens re-prefilled on the mounted branch (the
        // mountBranchAndPrefill worker call does the prefill internally
        // and returns the model's first-token prediction, which is the
        // opening of a fresh thinking channel).
        const { firstToken: postMountFirst } = await workerCall<{ firstToken: number; branch: string }>(
          {
            type: "mountBranchAndPrefill",
            branch: mountTarget,
            userTurnTokens: currentAttemptPrefilled ?? tokenizeConversation(conversation, true),
          },
          "mountBranchDone",
        );
        // Second run — under the mounted branch. pendingMount stays null
        // throughout; ModeTracker's fire-once semantics mean a second
        // setType in the branch's output won't trigger another mount.
        const second = await runThinkingAndCode(postMountFirst);
        phaseAReasonFinal = second.phaseAReason;
        // Remember the mount target so the NEXT attempt (retry after a
        // compile-gate failure) can skip the router entirely and mount
        // this branch directly. Without this assignment, retries fall
        // into the router path, the model doesn't re-emit setType
        // (it's already in the conversation), runCodePhaseOnly discards
        // every token, and we silently produce empty output.
        lastMountedBranch = mountTarget;
      }

      // Debug: capture how each phase actually terminated — invaluable for
      // diagnosing "attempt 0 empty / retry fills in" pathologies.
      const mountedNote = mountTarget ? ` mount=${mountTarget}` : "";
      console.log(`[draw] attempt ${attempt + 1} phaseA=${phaseAReasonFinal}${mountedNote} thinking=${thinkingText.length}ch code=${generatedCode.length}ch tokCount=${tokenCount}`);
      if (thinkingText.length > 0) {
        console.log(`[draw]   thinking[0..200]=${JSON.stringify(thinkingText.slice(0, 200))}`);
      }
      if (generatedCode.length > 0) {
        console.log(`[draw]   code[0..200]=${JSON.stringify(generatedCode.slice(0, 200))}`);
      }

      let code = generatedCode.trim();
      code = code.replace(/^```(?:javascript|js|typescript|ts)?\s*\n?/i, "");
      code = code.replace(/\n?```\s*$/i, "");
      // If a mount fired, generatedCode was seeded with the canonical
      // setType("<target>");\n at mount time. If the branch ALSO emitted
      // setType(...) as its first code line (some branches prompt
      // ambiguously), we'd have two setType calls — dedupe by removing
      // any non-first setType lines whose arg matches the mount target.
      // Defensive; most branches don't re-emit.
      if (mountTarget) {
        const canonical = `setType("${mountTarget}");`;
        // Strip any DUPLICATE setType call whose target matches mountTarget,
        // keeping only the first occurrence.
        let seen = false;
        code = code.replace(/setType\s*\(\s*"[^"]*"\s*\)\s*;?/g, (match) => {
          if (!seen) {
            seen = true;
            return canonical;
          }
          return "";
        });
        // Also ensure setType is on its own line (strip leftover `;;` or
        // blank-lines-with-whitespace from the dedupe).
        code = code.replace(/\n\s*\n\s*\n+/g, "\n\n").trim();
        // If somehow no setType made it in (e.g., the router's partial
        // was truncated before we seeded), prepend canonical.
        if (!/^\s*setType\s*\(/.test(code)) {
          code = `${canonical}\n${code}`;
        }
      }
      // Auto-strip lines that reference undeclared identifiers. Catches
      // the common small-model failure mode where the model writes
      // connect(foo, Bar, ...) or addGroup("X", [foo, Baz]) with Bar /
      // Baz never declared (case drift or hallucinated name). Instead
      // of throwing "<name> is not defined" and burning a retry, we
      // just drop the broken line. The rest of the diagram still
      // renders — usually with one missing edge / group member, which
      // is a much better UX than "retry 1 of 2".
      const stripped = stripUndeclaredReferences(code);
      if (stripped !== code) {
        code = stripped;
        setCode(code);
      } else {
        setCode(code);
      }
      if (code.trim().length > bestAttemptCode.trim().length) {
        bestAttemptCode = code;
      }

      statusEl.textContent = "Rendering diagram...";
      const { result, error } = await executeCode(code);
      finalResult = result;
      lastError = error;

      // Soft-error: code ran fine but some nodes were declared and never
      // connected. Small-model tax — only retry if a *meaningful* fraction
      // of the graph is disconnected (≥40%). One or two stray nodes in an
      // otherwise-valid diagram are cheaper to accept than to burn another
      // ~30s generation cycle.
      const elementsRendered = result.json?.elements || [];
      const orphans = error ? [] : detectOrphanNodes(elementsRendered);
      const totalNodes = elementsRendered.filter((el: any) => NODE_TYPES.has(el.type) && !el.customData?._group).length;
      const orphanRatio = totalNodes > 0 ? orphans.length / totalNodes : 0;
      const ORPHAN_RETRY_THRESHOLD = 0.4;
      let hasOrphans = orphans.length > 0
        && orphanRatio >= ORPHAN_RETRY_THRESHOLD
        && attempt < MAX_ATTEMPTS - 1;

      // Empty-output guard: `executeCode("")` succeeds silently, produces no
      // elements, and the no-orphan / no-error path would claim success with
      // an empty canvas. Treat any attempt that produced zero actual nodes
      // as a hard failure so the retry loop fires instead of declaring
      // "Diagram ready" with nothing on screen.
      const isEmpty = !code || totalNodes === 0;

      // IDE-style quick-fix: before triggering a full model regen for orphan
      // nodes, try mechanically stripping the `const ORPHAN = addXxx(...)`
      // declarations and re-executing. If the trimmed code produces a
      // connected diagram, skip the retry entirely — a ~15s regen saved
      // whenever the model over-declared. Only nodes on the orphan list get
      // stripped; connect/message/addGroup etc. lines are preserved even if
      // they referenced the stripped names (those calls become no-ops since
      // the names are undefined, but JS would throw — so we also drop lines
      // that REFERENCE orphan names). This is the quick-fix codemod an IDE
      // auto-applies; no other browser LLM stack has it because they lack
      // a compile-gate + semantic orphan detector in-page.
      if (!error && hasOrphans && !isEmpty) {
        const stripped = stripOrphanDeclarations(code, orphans);
        const { result: fixResult, error: fixError } = await executeCode(stripped);
        const fixElements = fixResult.json?.elements || [];
        const fixOrphans = fixError ? orphans : detectOrphanNodes(fixElements);
        const fixTotal = fixElements.filter((el: any) => NODE_TYPES.has(el.type) && !el.customData?._group).length;
        const fixRatio = fixTotal > 0 ? fixOrphans.length / fixTotal : 1;
        if (!fixError && fixTotal > 0 && fixRatio < ORPHAN_RETRY_THRESHOLD) {
          console.log(`[draw] attempt ${attempt + 1}: auto-fixed by stripping ${orphans.length} orphan(s): ${orphans.join(", ")}`);
          code = stripped;
          setCode(code);
          finalResult = fixResult;
          if (fixResult.json) updateDiagram(fixResult.json.elements || []);
          hasOrphans = false;
        }
      }

      if (!error && !hasOrphans && !isEmpty) {
        currentCode = code;
        if (result.json || finalResult?.json) {
          updateDiagram((finalResult?.json || result.json).elements || []);
          fitToScreen();
        }
        statusEl.textContent = attempt === 0 ? "Diagram ready" : `Diagram ready (fixed after ${attempt} ${attempt === 1 ? "retry" : "retries"})`;
        statusEl.classList.remove("error");
        break;
      }

      // Render whatever compiled, then queue retry.
      if (result.json) updateDiagram(result.json.elements || []);
      let retryFeedback: string;
      if (error) {
        console.log(`[draw] attempt ${attempt + 1} failed: ${error}`);
        retryFeedback = `The code above failed at runtime with: ${error}\n\nRegenerate the full code, fixing the error.`;
        conversation.push({ role: "assistant", content: code });
        conversation.push({ role: "user", content: retryFeedback });
      } else if (isEmpty) {
        console.log(`[draw] attempt ${attempt + 1}: empty output (no diagram produced)`);
        retryFeedback = `The previous response produced no diagram code — only reasoning text or an empty string. Respond with the JavaScript SDK calls directly, no prose, no markdown. Start with the first call and end with the last.`;
        conversation.push({ role: "assistant", content: code });
        conversation.push({ role: "user", content: retryFeedback });
        lastError = "empty output";
      } else {
        console.log(`[draw] attempt ${attempt + 1}: ${orphans.length} orphan node(s): ${orphans.join(", ")}`);
        retryFeedback = `The code above runs, but these nodes are declared and never connected to anything: ${orphans.map(o => `"${o}"`).join(", ")}\n\nEither add connect/message edges that reference them, or remove them. Regenerate the full code.`;
        conversation.push({ role: "assistant", content: code });
        conversation.push({ role: "user", content: retryFeedback });
      }
      // Dump the full user-feedback that the next retry will see — lets
      // the curious user inspect what the model is being asked to fix
      // before the retry starts.
      console.log(`[draw] retry feedback for attempt ${attempt + 2}:\n${retryFeedback}`);
    }

    if (lastError && !aborted) {
      if (finalResult?.json) { updateDiagram(finalResult.json.elements || []); fitToScreen(); }
      // Don't leave the editor empty after a failed run. If the current
      // content is shorter than the best attempt we saw (e.g. attempt 1
      // had usable code + one typo, attempt 3 produced almost nothing),
      // restore the best one so the user has the most context to
      // hand-edit before clicking Render.
      if (bestAttemptCode.trim().length > getCode().trim().length) {
        setCode(bestAttemptCode);
      }
      statusEl.textContent = `Code error after ${MAX_ATTEMPTS} attempts: ${lastError}`;
      statusEl.classList.add("error");
    }

    const elapsed = (performance.now() - startTime) / 1000;
    const e2eRate = tokenCount / elapsed;
    let speedLabel = `${e2eRate.toFixed(1)} tok/s · ${tokenCount} tok · ${elapsed.toFixed(1)}s`;
    if (firstTokenTime > 0 && tokenCount > 1) {
      const decodeEl = (performance.now() - firstTokenTime) / 1000;
      const decodeRate = (tokenCount - 1) / decodeEl;
      speedLabel = `${decodeRate.toFixed(1)} tok/s decode · ${e2eRate.toFixed(1)} e2e · ${tokenCount} tok · ${elapsed.toFixed(1)}s`;
    }
    statSpeed.textContent = speedLabel;
    editBtn.style.display = "inline-block";
    editBtn.textContent = "Done editing";
    enterEditMode();
  } catch (e) {
    statusEl.textContent = `Error: ${(e as Error).message}`;
    statusEl.classList.add("error");
    console.error("Generation error:", e);
  }

  busy = false;
  generateBtn.textContent = "Generate Diagram";
  resolveCurrent();
  currentGeneration = null;
}

// =============================================================================
// Init
// =============================================================================

/**
 * Gate the demo on WebGPU availability + the two required features
 * (shader-f16, subgroups). On Safari/older-Chrome/Firefox without these,
 * the worker would die with a raw error deep in init — surface a
 * friendly message BEFORE we spawn anything.
 */
async function checkWebGPUSupport(): Promise<string | null> {
  const ua = navigator.userAgent || "";
  const isIOS = /iPad|iPhone|iPod/.test(ua) || (ua.includes("Mac") && (navigator as any).maxTouchPoints > 1);
  const isAndroid = /Android/.test(ua);
  const isMobile = isIOS || isAndroid;
  const isFirefox = /Firefox\//.test(ua);
  const gpu = (navigator as any).gpu;
  // Error messages name the missing piece. Linking users to "try
  // Chrome" as a catch-all was making Firefox users think the demo
  // is Chrome-exclusive forever; in reality Firefox ships WebGPU on
  // Win + Apple Silicon but hasn't implemented the subgroups
  // extension yet (tracked in bugzilla). Be specific.
  const CHROME_PITCH = "Today that's desktop Chrome 134+ (Windows, macOS, or Linux).";
  const MOBILE_MSG = `Mobile browsers cap per-tab memory below the 3 GB this model holds in GPU RAM. Open on a desktop. ${CHROME_PITCH}`;
  const FIREFOX_MSG = `Firefox has WebGPU but hasn't shipped the \`subgroups\` extension yet — the demo's attention kernels depend on \`subgroupShuffleXor\`. ${CHROME_PITCH} Tracking the Firefox-side work at https://bugzilla.mozilla.org/show_bug.cgi?id=1927856 .`;
  const SAFARI_MSG = `Safari hasn't shipped the WebGPU subgroups extension. ${CHROME_PITCH}`;
  const NO_WEBGPU_MSG = `Your browser doesn't expose \`navigator.gpu\`. This demo needs WebGPU + shader-f16 + subgroups. ${CHROME_PITCH}`;

  if (isMobile) return MOBILE_MSG;
  if (!gpu) {
    return isFirefox ? FIREFOX_MSG : NO_WEBGPU_MSG;
  }
  let adapter: GPUAdapter | null;
  try {
    adapter = await gpu.requestAdapter();
  } catch (e) {
    return `WebGPU adapter request failed: ${(e as Error).message}. Close other GPU-heavy tabs and retry, or switch to desktop Chrome.`;
  }
  if (!adapter) {
    return "WebGPU is enabled but no GPU adapter is available. Try closing other GPU-heavy tabs, check chrome://gpu, or switch to desktop Chrome.";
  }
  const hasF16 = adapter.features.has("shader-f16");
  const hasSubgroups = adapter.features.has("subgroups");
  if (!hasF16 || !hasSubgroups) {
    if (isFirefox) return FIREFOX_MSG;
    const missing = [!hasF16 && "shader-f16", !hasSubgroups && "subgroups"].filter(Boolean).join(" + ");
    return `Your WebGPU adapter is missing required features (${missing}). ${CHROME_PITCH}`;
  }
  return null;
}

async function main() {
  createEditor(codeArea, (code) => { currentCode = code; });
  await loadWasm(drawmodeWasm);
  mountExcalidraw(diagramContainer);

  statusEl.innerHTML = '<span class="spinner"></span> Loading...';

  const gpuError = await checkWebGPUSupport();
  if (gpuError) {
    statusEl.innerHTML = `<span style="color:#f85149;">Unsupported browser/GPU</span>`;
    statusEl.classList.add("error");
    const errorCard = document.createElement("div");
    errorCard.style.cssText = "padding:16px;margin:12px;background:#21262d;border:1px solid #f85149;border-radius:6px;color:#e6edf3;font-size:13px;line-height:1.6;";
    errorCard.innerHTML = `<strong style="color:#f85149;">Can't run in this browser</strong><br><br>${gpuError}`;
    codeArea.appendChild(errorCard);
    generateBtn.disabled = true;
    promptEl.disabled = true;
    return;
  }

  // Singleton lock: only one tab runs the engine at a time. A second tab
  // would allocate its own WebGPU device, duplicate the 3 GB model pointer
  // in memory, and race for OPFS access — the browser would almost
  // certainly OOM or throttle both. navigator.locks.request with
  // `ifAvailable: true` returns null if another holder exists; holding the
  // returned promise forever keeps the lock for this page's lifetime (it
  // releases when the page unloads). Available on all WebGPU browsers
  // (Chrome 69+, Firefox 96+, Safari 15.4+).
  const lockAcquired = await new Promise<boolean>((resolve) => {
    navigator.locks.request(
      "turboquant-draw-engine",
      { ifAvailable: true },
      (lock) => {
        if (lock === null) { resolve(false); return undefined; }
        resolve(true);
        return new Promise<void>(() => { /* held for page lifetime */ });
      },
    );
  });
  if (!lockAcquired) {
    statusEl.innerHTML = `<span style="color:#f85149;">Already running in another tab</span>`;
    statusEl.classList.add("error");
    const errorCard = document.createElement("div");
    errorCard.style.cssText = "padding:16px;margin:12px;background:#21262d;border:1px solid #f85149;border-radius:6px;color:#e6edf3;font-size:13px;line-height:1.6;";
    errorCard.innerHTML = `<strong style="color:#f85149;">Already running in another tab</strong><br><br>This demo keeps a 3 GB Gemma 4 E2B model resident in GPU memory. Running it twice concurrently would OOM the browser. Close the other tab and reload this one.`;
    codeArea.appendChild(errorCard);
    generateBtn.disabled = true;
    promptEl.disabled = true;
    return;
  }

  try {
    console.log("[draw] loading tokenizer...");
    tokenizer = await AutoTokenizer.from_pretrained(TOKENIZER_ID);
    // Gemma 4 E2B uses `<|turn>`/`<turn|>` — see CHAT_TEMPLATE above. The
    // onnx-community tokenizer_config includes a full template with tool and
    // thinking support; we only need the minimal role-alternation form.
    tokenizer.chat_template = CHAT_TEMPLATE;
    console.log("[draw] tokenizer ready");

    // Dynamic-context setup: tokenize every branch's prompt up-front. The
    // router is what the engine actually boots with (its KV becomes the
    // initial system cache); sequence + architecture are pre-registered on
    // the engine so mountKV() can swap them in mid-stream when the grammar
    // recognises a completed setType(...) call.
    //
    // Normal users never pass any query param — branch selection at
    // runtime is driven by what the model emits, which is the whole point
    // of the dynamic context design.
    //
    // ?buildBranch=<name> is an internal-only override used by the
    // build-cache playwright test (tests/build-cache.spec.ts). It forces
    // the boot branch's prompt to be <name> instead of "router" so the
    // test can produce a TQKV dump per branch. The param is NEVER part of
    // the normal user-facing flow.
    const tokenizeBranchPrompt = (text: string): number[] => {
      const msgs = [{ role: "system", content: text }];
      const asText = tokenizer.apply_chat_template(msgs, { tokenize: false, add_generation_prompt: false });
      return Array.from(tokenizer.encode(asText));
    };
    const branchTokenIds: Record<BranchName, number[]> = {
      router: tokenizeBranchPrompt(BRANCHES.router),
      sequence: tokenizeBranchPrompt(BRANCHES.sequence),
      architecture: tokenizeBranchPrompt(BRANCHES.architecture),
      flowchart: tokenizeBranchPrompt(BRANCHES.flowchart),
      state: tokenizeBranchPrompt(BRANCHES.state),
      orgchart: tokenizeBranchPrompt(BRANCHES.orgchart),
      er: tokenizeBranchPrompt(BRANCHES.er),
      class: tokenizeBranchPrompt(BRANCHES.class),
      swimlane: tokenizeBranchPrompt(BRANCHES.swimlane),
    };
    console.log(
      `[draw] branch token counts:\n` +
      Object.entries(branchTokenIds)
        .map(([name, ids]) => `  ${name.padEnd(14)} = ${ids.length}`)
        .join("\n"),
    );

    const buildBranchParam = new URLSearchParams(location.search).get("buildBranch");
    const bootBranch: BranchName =
      buildBranchParam && (buildBranchParam in BRANCHES)
        ? (buildBranchParam as BranchName)
        : "router";
    if (buildBranchParam && bootBranch !== "router") {
      console.log(`[draw] BUILD-ONLY override: boot branch = "${bootBranch}" (via ?buildBranch=)`);
    }

    // The "live" system cache at boot is the bootBranch (router for real
    // users; one of the other branches when the build-cache test is
    // regenerating that specific branch's KV). User prompts prefill on
    // top of this KV. Mid-stream, when setType fires, engine.mountKV
    // replaces it with the chosen branch and the do-over re-prefills
    // the user turn on top of the new branch.
    const systemTokenIds: number[] = branchTokenIds[bootBranch];
    console.log(`[draw] active boot branch: ${bootBranch}, ${systemTokenIds.length} tokens`);
    systemCacheEnd = systemTokenIds.length;

    worker = new EngineWorker();

    // Engine proxy for testing (same API shape as direct engine)
    const engineProxy: any = {
      resetCache: () => { worker!.postMessage({ type: "resetCache" }); },
      generateToken: (tokenId: number) => workerCall<{ id: number; stats: any; profile: any }>(
        { type: "decode", tokenId }, "token",
      ).then(r => { engineProxy.lastProfile = r.profile; return r.id; }),
      streamTokens: (
        tokenId: number, maxTokens: number, eosIds: number[],
        onToken: (t: { id: number; stats: any }) => void,
      ) => workerStream(
        { type: "stream", tokenId, maxTokens, eosIds },
        (r) => { engineProxy.lastStats = r.stats; onToken(r); },
      ),
      prefill: (tokenIds: number[]) => workerCall<{ firstToken: number }>(
        { type: "prefill", tokenIds }, "prefillDone",
      ).then(r => r.firstToken),
      decodeBatch: (tokenIds: number[]) => workerCall<{ argmaxes: number[] }>(
        { type: "decodeBatch", tokenIds }, "decodeBatchDone",
      ).then(r => r.argmaxes),
      rollbackKV: (targetPosition: number) => workerCall<{ ok: true }>(
        { type: "rollbackKV", targetPosition }, "rollbackKVDone",
      ).then(() => {}),
      getStats: () => workerCall<{ type: string; data: any }>({ type: "getStats" }, "stats").then(r => r.data),
      restoreCache: () => { worker!.postMessage({ type: "restoreCache" }); },
      snapshotCache: () => { worker!.postMessage({ type: "snapshotCache" }); },
      dumpSystemCache: () => workerCall<{ type: string; data: ArrayBuffer }>(
        { type: "dumpCache", systemTokenIds }, "cacheDump",
      ).then(r => r.data),
      conformanceBatchedMatmul: (weightName: string, batchSize: number) => workerCall<{ type: string; data: any }>(
        { type: "conformanceBatchedMatmul", weightName, batchSize }, "batchedMatmulResult",
      ).then(r => r.data),
      benchmarkBatchedMatmul: (weightName: string, batchSize: number, iters: number) => workerCall<{ type: string; data: any }>(
        { type: "benchmarkBatchedMatmul", weightName, batchSize, iters }, "benchmarkBatchedMatmulResult",
      ).then(r => r.data),
      decodeBatchGenuine: (tokenIds: number[]) => workerCall<{ type: string; data: number[] }>(
        { type: "decodeBatchGenuine", tokenIds }, "decodeBatchGenuineResult",
      ).then(r => r.data),
      systemTokenIds,
      lastProfile: null as any,
      lastStats: null as any,
      model: { tensors: new Map<string, any>() },
      enableProfiling: () => {},
    };
    (window as any).__engine = engineProxy;
    (window as any).__tokenizer = tokenizer;
    (window as any).__engineWorker = worker;
    // Expose the viewer and executor so tests (and devtools) can drive the
    // live module instance rather than re-importing the module under a
    // different URL, which would give them a fresh module-level state and
    // silently miss the already-mounted svgWrapper.
    (window as any).__viewer = { updateDiagram, resetDiagram, fitToScreen, enterEditMode, exitEditMode, getMode };
    const { executeCode: _exec } = await import("./drawmode/executor.js");
    (window as any).__executeCode = _exec;

    // Gate readiness on BOTH model init and grammar upload — streamConstrained
    // requires the grammar bitmaps to be on-GPU, so we must not accept user
    // submissions until they're there.
    let workerInitDone = false;
    let grammarInitDone = false;
    const markReadyIfBoth = () => {
      if (workerInitDone && grammarInitDone && !modelReady) {
        modelReady = true;
        statusEl.innerHTML = "Ready";
        statusEl.classList.add("ready");
        promptEl.focus();
        console.log("[draw] Worker + grammar ready");
      }
    };
    worker!.onmessage = (e: MessageEvent) => {
      const msg = e.data;
      if (msg.type === "status") statusEl.innerHTML = `<span class="spinner"></span> ${msg.text}`;
      if (msg.type === "tensorMeta") {
        for (const t of msg.data) engineProxy.model.tensors.set(t.name, t);
      }
      if (msg.type === "ready") { workerInitDone = true; markReadyIfBoth(); }
      if (msg.type === "grammarReady") { grammarInitDone = true; markReadyIfBoth(); }
      // Worker may dump a freshly-prefilled router KV if the on-disk cache
      // was missing or stale. We used to forward this to the dev server to
      // auto-replace public/system-cache.bin, but with the multi-branch
      // TQKC format that convenience is actively dangerous: a single-branch
      // TQKV post would overwrite all seven non-router branches with
      // nothing. Ignore the dump — `bun run rebuild-cache` is the
      // authoritative way to refresh every branch.
      if (msg.type === "cacheRebuilt") {
        console.log(`[draw] worker dumped ${(msg.data.byteLength / 1e6).toFixed(1)} MB (auto-write disabled — run 'bun run rebuild-cache' to refresh public/system-cache.bin)`);
      }
      if (msg.type === "error" && !busy) {
        statusEl.textContent = `Error: ${msg.message}`;
        statusEl.classList.add("error");
        console.error("[draw] Worker error:", msg.message);
      }
    };

    // ?skipSysPrompt=1 bypasses the long system-prompt prefill so that conformance
    // tests can start generating immediately.
    const skipSys = new URLSearchParams(location.search).get("skipSysPrompt") === "1";
    // ?noCache=1 forces the worker down the prefill path — used by the build
    // script to regenerate system-cache.bin from scratch.
    const skipBuiltCache = new URLSearchParams(location.search).get("noCache") === "1";
    const initSystemTokenIds = skipSys ? [] : systemTokenIds;
    const enableProfile = new URLSearchParams(location.search).get("profile") === "1";

    // Fetch the prebuilt TQKC multi-branch cache. If missing or malformed,
    // fall through to the slow prefill path (worker will prefill the router
    // from scratch, then the mount infrastructure still works — just with a
    // cold first boot).
    //
    // `no-cache` (not `no-store`) revalidates with the server so HTTP
    // caching still works.
    //
    // Branch blobs are held in owned ArrayBuffers so postMessage transfer
    // works across the worker boundary (can't transfer a slice of a parent
    // buffer; slicing copies).
    type BranchInit = { name: BranchName; blob: ArrayBuffer; tokenCount: number; tokenIds: number[] };
    let cacheBlob: ArrayBuffer | null = null;             // router blob (the boot branch)
    const extraBranchInits: BranchInit[] = [];             // sequence + architecture for later mount
    if (!skipSys && !skipBuiltCache) {
      try {
        const res = await fetch("./system-cache.bin", { cache: "no-cache" });
        if (res.ok && (res.headers.get("content-type") || "").includes("octet-stream")) {
          const fullBuf = await res.arrayBuffer();
          const parsed = parseSystemCache(fullBuf);
          console.log(`[draw] cache fetched: ${(fullBuf.byteLength / 1e6).toFixed(1)} MB, ${parsed.branches.size} branches (version=${parsed.version})`);
          for (const name of Object.keys(branchTokenIds) as BranchName[]) {
            const entry = parsed.branches.get(name);
            if (!entry) {
              console.warn(`[draw] cache missing branch "${name}" — will prefill on mount`);
              continue;
            }
            const owned = entry.blob.slice().buffer as ArrayBuffer;
            if (name === "router") {
              cacheBlob = owned;
            } else {
              extraBranchInits.push({
                name,
                blob: owned,
                tokenCount: entry.tokenCount,
                tokenIds: branchTokenIds[name],
              });
            }
          }
        }
      } catch (e) {
        console.warn(`[draw] cache fetch failed: ${(e as Error).message} — falling back to prefill`);
      }
    }

    // ?benchUpload=1 makes the worker run every tensor-upload strategy once
    // and log timings — used to benchmark which OPFS read pattern wins.
    const benchUpload = new URLSearchParams(location.search).get("benchUpload") === "1";
    const initMsg: any = {
      type: "init",
      ggufUrl: GGUF_URL,
      systemTokenIds: initSystemTokenIds,
      enableProfile,
      cacheBlob,
      benchUpload,
      // Non-active branches: worker registers each on the engine so
      // mountKV() can later swap one of them into the live cache on setType
      // detection. router is handled by cacheBlob (or fresh prefill) above.
      extraBranches: extraBranchInits.map(({ name, blob, tokenCount, tokenIds }) => ({ name, blob, tokenCount, tokenIds })),
    };
    const transfers: Transferable[] = [];
    if (cacheBlob) transfers.push(cacheBlob);
    for (const b of extraBranchInits) transfers.push(b.blob);
    worker!.postMessage(initMsg, transfers);

    // Grammar bitmaps: build in the background while the worker is busy
    // downloading/uploading the GGUF. Ship to the worker as soon as ready.
    // Gemma 4 E2B vocab=262144 → ~256 KB masks + 2 MB transitions, build
    // yields every 4096 tokens so it doesn't freeze the UI.
    //
    // IMPORTANT: must NOT use `tokenizer.decode([id])` — SentencePiece
    // decoders strip the leading `▁` when the token is at position 0 of
    // the decode call, so a token that represents ` Input` decodes to
    // `Input` and the grammar sees two ident tokens as one continuous
    // identifier. Use raw vocab strings (with `▁` for space) instead,
    // then substitute ourselves.
    const idToToken = new Array<string>(VOCAB_SIZE);
    for (const [tok, id] of tokenizer.get_vocab().entries()) idToToken[id] = tok;
    const rawDecode = (id: number): string => {
      const t = idToToken[id];
      if (!t) return "";
      // SentencePiece: U+2581 ("▁") marks a leading space. Also handle
      // BPE bytes for newline/tab that appear on some vocabs.
      return t.replaceAll("\u2581", " ");
    };
    const t0 = performance.now();
    buildGrammar(rawDecode, VOCAB_SIZE).then(({ masks, transitions }) => {
      const dt = ((performance.now() - t0) / 1000).toFixed(1);
      console.log(`[draw] grammar built in ${dt}s: ${masks.length} u32s masks, ${transitions.length} bytes transitions`);
      worker!.postMessage(
        {
          type: "initGrammar",
          masks: masks.buffer,
          transitions: transitions.buffer,
          vocabSize: VOCAB_SIZE,
          maskCount: NUM_STATES,
        },
        [masks.buffer, transitions.buffer],
      );
    }).catch((e) => {
      console.warn("[draw] grammar build failed:", (e as Error).message);
    });

  } catch (e) {
    statusEl.textContent = `Error: ${(e as Error).message}`;
    statusEl.classList.add("error");
    console.error("Load error:", e);
  }
}

// =============================================================================
// Event listeners
// =============================================================================

generateBtn.addEventListener("click", generate);
promptEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) generate();
});
promptEl.addEventListener("input", scheduleUserPrefill);
promptEl.addEventListener("change", scheduleUserPrefill);

document.querySelectorAll(".suggestion").forEach(btn => {
  btn.addEventListener("click", () => {
    promptEl.value = (btn as HTMLElement).dataset.prompt!;
    promptEl.focus();
    scheduleUserPrefill();
  });
});

renderBtn.addEventListener("click", async () => {
  let code = getCode();
  if (!code) return;
  if (!code.includes("d.render()")) code = `${code}\nreturn d.render();`;
  statusEl.textContent = "Rendering...";
  const { result, error } = await executeCode(code);
  if (error) {
    statusEl.textContent = `Code error: ${error}`;
    statusEl.classList.add("error");
  } else if (result.json) {
    updateDiagram(result.json.elements || []);
    fitToScreen();
    statusEl.textContent = "Diagram ready";
    statusEl.classList.remove("error");
  }
});

wipeBtn.addEventListener("click", async () => {
  if (busy) { aborted = true; worker?.postMessage({ type: "abort" }); }
  window.stop();
  statusEl.textContent = "Wiping all data...";
  wipeBtn.disabled = true;

  // Report storage usage before the wipe so we can see what's reclaimed.
  const usageBefore = await navigator.storage.estimate?.().catch(() => null);

  // Wait for the worker's async wipeModels() to actually finish OPFS
  // deletion before killing the worker. Previously we called terminate()
  // in the same tick as postMessage({wipe}), which cancelled the delete
  // mid-execution and left the 3 GB model on disk.
  const w = worker;
  if (w) {
    try {
      await new Promise<void>((resolve) => {
        const handler = (e: MessageEvent) => {
          if (e.data?.type === "wipeDone" || e.data?.type === "error") {
            w.removeEventListener("message", handler);
            resolve();
          }
        };
        w.addEventListener("message", handler);
        w.postMessage({ type: "wipe" });
        // Hard safety net: if the worker never acks (stuck), give up after 15s
        // so the UI isn't wedged forever.
        setTimeout(() => { w.removeEventListener("message", handler); resolve(); }, 15_000);
      });
    } finally {
      w.terminate();
    }
  }
  worker = null;
  tokenizer = null;
  modelReady = false;

  // Cache Storage API — Transformers.js stores tokenizer/model blobs here
  // (separate from OPFS, invisible to wipeModels). Delete every named cache
  // owned by this origin.
  try {
    if (typeof caches !== "undefined") {
      const keys = await caches.keys();
      await Promise.all(keys.map(k => caches.delete(k)));
      console.log(`[draw] cleared ${keys.length} Cache Storage entries`);
    }
  } catch (e) {
    console.warn("[draw] caches.delete failed:", (e as Error).message);
  }

  // IndexedDB — some tokenizer / HF Hub paths use it as a fallback. Enumerate
  // all databases for this origin and drop them. indexedDB.databases() is
  // Chrome/Edge; Firefox/Safari silently skip the loop.
  try {
    const indexedDBAny = indexedDB as any;
    if (typeof indexedDBAny.databases === "function") {
      const dbs = await indexedDBAny.databases();
      for (const db of dbs) {
        if (db.name) {
          await new Promise<void>((resolve) => {
            const req = indexedDB.deleteDatabase(db.name);
            req.onsuccess = () => resolve();
            req.onerror = () => resolve();
            req.onblocked = () => resolve();
          });
        }
      }
      console.log(`[draw] cleared ${dbs.length} IndexedDB databases`);
    }
  } catch (e) {
    console.warn("[draw] indexedDB wipe failed:", (e as Error).message);
  }

  // Service-worker unregistration: if some future version registers one to
  // cache model bytes, leaving it alive would keep serving stale content.
  try {
    if (navigator.serviceWorker) {
      const regs = await navigator.serviceWorker.getRegistrations();
      await Promise.all(regs.map(r => r.unregister()));
      if (regs.length > 0) console.log(`[draw] unregistered ${regs.length} service worker(s)`);
    }
  } catch (e) {
    console.warn("[draw] sw unregister failed:", (e as Error).message);
  }

  const usageAfter = await navigator.storage.estimate?.().catch(() => null);
  if (usageBefore && usageAfter) {
    const before = (usageBefore.usage ?? 0) / 1e9;
    const after = (usageAfter.usage ?? 0) / 1e9;
    console.log(`[draw] storage: ${before.toFixed(2)} GB → ${after.toFixed(2)} GB (reclaimed ${(before - after).toFixed(2)} GB)`);
    statusEl.textContent = `All data wiped — reclaimed ${(before - after).toFixed(2)} GB (${after.toFixed(2)} GB remaining)`;
  } else {
    statusEl.textContent = "All data wiped.";
  }
  setCode("");
  currentCode = "";
  updateDiagram([]);
  statSpeed.textContent = "--";
  statKV.textContent = "KV: --";
  wipeBtn.style.display = "none";
  generateBtn.disabled = true;
});

wipeBtn.style.display = "inline-block";

// Edit button: flip between streaming SVG view and the full interactive
// Excalidraw component. Streaming uses SVG because exportToSvg avoids the
// updateScene cache bugs that plagued live-component partials; when the
// stream is done the user can click Edit to drop into the full editor with
// the final elements.
editBtn.addEventListener("click", async () => {
  if (getMode() === "svg") {
    editBtn.textContent = "Done";
    await enterEditMode();
  } else {
    exitEditMode();
    editBtn.textContent = "Edit";
    fitToScreen();
  }
});

main();
