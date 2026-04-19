# Dynamic Context Window

Grammar-triggered, mid-generation KV swap. The LLM boots with a small
"router" system prompt listing every diagram type. When it emits
`setType("...")`, a runtime observer detects that in the decoded token
stream and the engine swaps the active KV cache to a pre-baked branch
containing the full per-type documentation — zero runtime prefill.

Shipped on branch `feat/dynamic-context-window`. v1 covers flat
whole-cache swaps (no nested sub-shells). v2 items — nested mount,
RoPE shift for arbitrary mount positions, per-type grammar
restrictions — are still open.

## Problem

The original single-blob system prompt was 3005 tokens covering 8
diagram types. Two costs:

1. **Prefill time** — paid once at system-cache build, ~2s. Blob was
   68.9 MB on disk, downloaded every session on a fresh cache.
2. **Attention scan** — every decode step attends over all 3005 tokens
   plus the live generation. Any given prompt uses at most one type's
   worth of docs; the other ~7/8 is noise the model has to ignore.

Naive fix ("inject focused docs mid-generation") runs into prefill
latency: 500 new tokens = ~250 ms of prefill per injection. Fine once,
prohibitive repeatedly.

## Core idea

Pre-prefill one KV cache per diagram type at build time, all packed
into a single container file. At runtime, the router cache is active.
When the model emits `setType("architecture")` (or any other valid
type), swap the active cache to that type's pre-baked KV. The swap is
a GPU memcpy, not a prefill. The model loses the router context but
gains dense per-type context with a re-prefill of just the user turn
on top.

Grammar-triggered because the decision point is a syntactic event the
model produces naturally — no extra routing logic, no pre-request
heuristics.

## Flow

```
build time:
  for each of 9 prompts (router + 8 types):
    tokenize, prefill engine, dumpCache → TQKV blob
  pack all 9 into one TQKC container → public/system-cache.bin

runtime boot:
  fetch system-cache.bin, parse TQKC
  load "router" branch's KV as active system cache
  register the other 8 branches on the engine (CPU-resident blobs,
    ready for mountKV)

generate:
  prefill user turn on top of router
  skip the thinking phase (prefill <channel|> + reminder — router's
    only job is to pick a type, not plan the diagram)
  stream code tokens; ModeTracker scans each decoded chunk

on ModeTracker fire (setType("X") recognised):
  abort stream
  engine.mountKV("X")  — swap active KV to that branch
  re-prefill user turn on top of the mounted branch
  seed editor with canonical setType("X");\n
  run full thinking + code phase under the specialised KV
  (fire-once per attempt — a second setType in branch output is
   suppressed because its editor duplicate was already written)

retry (compile gate fails):
  restoreCache — puts KV back to router snapshot, resets activeBranch
  outer loop re-enters with fresh conversation + error feedback
```

## Prior art

| System | What they do | What's missing |
|---|---|---|
| SGLang RadixAttention | Shares prefix KVs across requests in a tree | Chosen per-request, not swapped mid-generation |
| RP lorebooks (SillyTavern) | Keyword-triggered context injection | String-level prepend, full re-prefill each trigger |
| Constrained decoding (Outlines, Guidance) | Grammar masks output logits | Never touches KV |

The specific combination — grammar-triggered, mid-generation KV swap
with pre-baked per-mode branches — doesn't appear published.

## Implementation

### Multi-branch cache file format (TQKC)

`demo/src/draw/system-cache-container.ts`. Little-endian, all u32
(not u64 — max blob is ~20 MB, u32 is fine):

```
4 bytes   magic "TQKC"
2 bytes   version u16  (= 1)
2 bytes   branch_count u16
per branch index entry:
  1 byte   name_len u8  (max 32)
  name_len bytes         utf-8
  4 bytes  offset u32    (bytes from file start to the branch's TQKV blob)
  4 bytes  length u32    (bytes occupied by that blob)
  4 bytes  token_count u32
per branch blob (concatenated after the index):
  TQKV blob (format unchanged, exactly what Engine.dumpCache emits)
```

`packContainer(entries)` / `unpackContainer(buffer)` are pure — no
engine, no GPU. `parseSystemCache` auto-detects TQKV single-blob vs
TQKC container for backwards compat, though v1 ships TQKC-only.

### Branches (9 total)

Current sizes (`bun run rebuild-cache` reproduces):

```
router       766 tok    5.9 MB
sequence    1202 tok    9.3 MB
architecture 1662 tok   12.8 MB
flowchart   1219 tok    9.4 MB
state       1148 tok    8.9 MB
orgchart    1189 tok    9.2 MB
er          1438 tok   11.1 MB
class       1405 tok   10.9 MB
swimlane    1399 tok   10.8 MB
            total      88.3 MB
```

File is ~30% larger than the old single blob (88 vs 69 MB), but at any
moment only one branch is live so the runtime attention scan is
smaller for every type (766 for router, 1148-1662 for the mounted
branch vs the old uniform 3005).

### KV mount primitive

`engine.mountKV(name)` in `engine.ts`. Semantics:

- Loads the named branch's TQKV blob via the existing `loadCache`
  path.
- Sets `engine.position = branch.tokenCount`.
- Sets `engine._activeBranch = name`.
- Caller is responsible for re-prefilling anything they want past the
  branch's end (user turn, seed tokens).

No RoPE shift in v1: each branch was prefilled starting at position
0, so mounting it as the active cache at position 0 keeps every K
rotation's phase correct. Arbitrary-position mount (for nested
sub-shells) would need a runtime K re-rotation pass — v2 work.

### ModeTracker

`demo/src/draw/grammar.ts`. Side-channel observer, not composed into
the char-level grammar's state machine. Pattern match on each decoded
token chunk:

```
/setType\s*\(\s*"(sequence|architecture|flowchart|state|orgchart|er|class|swimlane)"\s*\)/
```

Nine SDK_MODE constants (UNSET + the 8 types). `sdkModeForSetTypeArg`
maps the matched arg to a constant. `onEnter(mode, fn)` registry fires
synchronously when observe detects the first match — subsequent
matches in the same generation are no-ops. `reset()` re-arms for the
next attempt.

Chose a side-channel over nesting into the transition table: nesting
would multiply mask + transition sizes by 9 for a feature that fires
at most once per generation. Pattern match is O(emitted text len) and
stops on first hit.

### Retry protocol

`engine.restoreCache()` now also restores `_activeBranch` (via a
`_snapshotBranch` field captured by `snapshotCache`). So a retry after
a mid-stream mount cleanly goes back to the router snapshot, and the
bookkeeping stays consistent — no stale `activeBranch = "architecture"`
after restoring to router KV.

## Scope

### v1 (shipped)

- [x] TQKC multi-branch container format (writer + reader + 7 tests)
- [x] `engine.mountKV(name)` + `registerBranch` primitives
- [x] ModeTracker with 9-way routing + onEnter hooks (14 tests)
- [x] Per-type branches: router + 8 types
- [x] Build pipeline iterates all 9 branches
- [x] Mid-stream mount do-over in `main.ts` generate()
- [x] Retry protocol restores activeBranch bookkeeping
- [x] Skip-thinking-on-router optimisation

### v2 (open)

- [ ] Runtime RoPE-shift pass for arbitrary-position mount. Lets a
      branch attach onto the end of an already-active cache instead
      of replacing it, enabling nested sub-shells.
- [ ] Per-mode grammar. Today the char-level grammar is shared by all
      modes; per-mode masks would hard-enforce the allowed function
      vocabulary (addActor only in sequence, addClass only in class,
      etc.) rather than relying on per-type prompt instruction.
- [ ] Finer sub-shells: addBox param help, connect args help, etc.
      Requires nested mount.
- [ ] Conformance test for RoPE-shift-through-TQ numerical drift —
      blocker for v2 since re-RoPE needs decode → rotate → re-encode
      roundtrip through polar + QJL, which is lossy.

## Edge cases

1. **Model never calls setType.** Tracker doesn't fire, no mount, stay
   on router. Router is a valid self-contained prompt for the narrow
   case where the model can answer without per-type docs (rare).
2. **Model calls setType with an unknown arg.** Regex doesn't match,
   no mount, stay on router. Grammar doesn't currently mask this at
   the token level — per-mode grammar (v2) would.
3. **Second setType in the same generation.** ModeTracker's fire-once
   guard returns false. No effect. Branch output's leading setType
   (if it re-emits) is filtered from the editor by the suppressor.
4. **Retry exhaustion.** After MAX_ATTEMPTS (3), surface best-effort
   output, same as pre-dynamic-context behaviour.
5. **Cache file version mismatch.** Magic + version header; on
   mismatch the runtime logs a warning and falls back to fresh
   prefill.
6. **KV budget.** Worker pre-allocates for the largest branch plus
   generation budget. Branches are swapped into the same slots.
