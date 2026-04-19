# Dynamic Context Window

Grammar-triggered, mid-generation KV swap. The LLM's working context
contracts and expands as it navigates the SDK — like pushd/popd on a
filesystem — with zero runtime prefill cost.

Status: design, not implemented. Branch: `feat/dynamic-context-window`.

---

## Problem

Our system prompt is 3005 tokens: full SDK documentation for three
diagram modes (sequence, architecture, UML) plus shared preamble. Two
costs scale with its size:

1. **Prefill time** — paid once at system-cache build (~2s) but the bin
   is 68.9 MB on disk and must be downloaded + loaded each session.
2. **Attention scan** — every decode step attends over all 3005 system
   tokens plus the live generation. Each mode only uses ~1/3 of that
   system prompt; the other 2/3 is clutter the model must learn to
   ignore.

The obvious fix — "just inject relevant docs when needed" — runs into
prefill latency. Adding 500 tokens of focused help mid-generation means
paying ~250 ms of prefill per injection. That's fine once per
generation, prohibitive every few tokens.

## Core idea

Pre-prefill multiple specialised system prompts **offline**. At runtime,
load a small **router** prompt (tells the model "three modes exist, pick
one"). When the model emits `setType("sequence")`, the grammar state
machine fires a **mount** action that swaps in the pre-baked
`sequence.kv` onto the live cache. Zero runtime prefill. The model now
has dense mode-specific context without ever having "seen" it prefilled
in this session.

Grammar is already the state machine (`demo/src/draw/grammar.ts`).
We extend it with SDK-level states and attach side-effect actions to
specific transitions.

## TL;DR

```
build time:   prefill router.kv  +  sequence.kv  +  arch.kv  +  uml.kv
              all baked into public/system-cache.bin

runtime:      load router.kv     (~500 tok,    fast)
              model emits setType("sequence")
              grammar transition: S_ROUTER → S_SEQUENCE
                                  action: mountKV(sequence)
              continue decode with mode-specific KV

retry:        compile gate fails
              rollback to pre-setType snapshot
              remount router.kv (so the model reconsiders the mode)
              inject error feedback
              continue decode
```

## Prior art

Closest three, none of which do this combination:

| System | What they do | What's missing |
|---|---|---|
| SGLang **RadixAttention** | Caches prefix KVs across requests; share tree of prompts | Chosen pre-request, not swapped mid-stream |
| **RP lorebooks** (SillyTavern etc.) | Keyword-triggered context injection | String-level prepend, full re-prefill on every trigger |
| Constrained decoding (Outlines, Guidance) | Grammar masks logits | Controls output tokens, never touches KV |

**Gap:** grammar-triggered, mid-generation KV swap using pre-baked
branches. Unclaimed as far as I can tell.

## Design

### 1. Multi-branch cache file format

Today: `public/system-cache.bin` is one `TQKV`-magic blob.

New format:

```
[magic "TQKV"]
[version u16]
[branch_count u16]
[branch_index: branch_count × {name_len u8, name bytes, offset u64, len u64, token_count u32}]
[branch_0 KV bytes]
[branch_1 KV bytes]
...
```

Worker loads all branches into GPU memory on init. Active branch is a
pointer + position-offset pair.

Branch names (v1):
- `"router"` — mode-picker preamble; small, always loaded initially
- `"sequence"`, `"architecture"`, `"uml"` — per-mode specialised prompts

Branch sizes to fit in current KV budget: router ~500 tok, each mode
branch ~1800 tok. Total baked ~5900 tok vs current 3005 — larger file,
but only one branch is "live" at a time, so attention scan per decode
step is smaller.

### 2. KV mount primitive

**What:** splice a pre-baked branch's KV onto the live cache at the
current position, so the next token attends over
`[live prefix ++ branch KV]`.

**Mechanics per layer:**

- Allocate cache slots at positions `P_current .. P_current + branch_len - 1`.
- Copy branch's K, V tensors into those slots. V needs no transform.
- **K needs re-RoPE**: branch K was encoded with positional phase for
  position 0. Multiply each K head's rotation by
  `exp(i × θ × P_current)` to shift it to `P_current`. One compute-shader
  pass per layer, sub-ms on M1.

Note: TQ encodes K in a rotated space, so the re-RoPE pass runs before
the TQ encode on build, OR after decode on runtime mount. Build-time
re-RoPE is impossible (we don't know `P_current` yet), so it has to be
runtime. Decode TQ K → re-RoPE → re-encode TQ K. This is the only
non-cheap part; needs benchmarking before committing.

**Alternative:** skip RoPE shift entirely. Always mount at a FIXED
position (end of router). Never append more than one sub-branch. On
retry, rollback and remount. This eliminates RoPE-shift complexity at
the cost of "branches are only usable from root" (no nesting).

**Recommendation for v1:** alternative. Prove the idea with flat
branches first. Nested sub-menus (addBox param help, etc.) come later,
once the RoPE-shift cost is measured.

### 3. Grammar extension: SDK-aware states

Current grammar states are char-level (`S_FREE`, `S_PAREN_NEUTRAL`, …).
We add SDK-level states on top:

```
S_MODE_UNSET  — before setType() called
S_SEQUENCE    — inside setType("sequence") mode
S_ARCH        — inside setType("architecture") mode
S_UML         — inside setType("uml") mode
```

These compose with the existing char-level states; the grammar becomes
`(char_state, sdk_state)` pairs. SDK-state transitions fire on
recognising completed SDK calls.

**Hook API:**

```ts
interface GrammarAction {
  // Fires when grammar completes a transition into `state`.
  onEnter?: (state: number) => void;
  // Fires when grammar exits `state` (e.g., rollback).
  onExit?: (state: number) => void;
}
```

Engine wires `onEnter(S_SEQUENCE)` to `engine.mountKV("sequence")`.

### 4. Retry protocol

Today: on compile-gate failure, feed error back, retry from a snapshot.

New: on compile-gate failure:

1. `engine.rollbackKV(routerEnd)` — erase all generation including
   `setType(...)`.
2. `engine.mountKV("router")` — restore full menu.
3. Inject error feedback tokens ("last attempt failed: <error>. Try a
   different approach.").
4. Resume decode.

The key insight: if the sub-menu was wrong (model picked arch when user
wanted sequence), retrying *within* that sub-menu will reproduce the
error. Always rollback to the fork point.

### 5. Build pipeline

`tests/build-cache.spec.ts` today builds one cache. Extend to:

```
for each branch in [router, sequence, arch, uml]:
    page.goto(draw.html?noCache=1&branch=${branch})
    wait for status=Ready
    dump KV
    append to system-cache.bin with branch_index entry
```

Prompt hash includes all four branches' source slices, not just a
single SYSTEM_PROMPT.

---

## Scope

### v1 — flat branches, router + 3 modes

- [ ] Multi-branch cache file format (writer + reader)
- [ ] `engine.mountKV(branchName)` — load from baked blob, no RoPE shift
  (mount at end of router only)
- [ ] Grammar SDK-state extension (4 states: unset/sequence/arch/uml)
- [ ] Grammar action hook API + one hook wiring `setType(x)` → mount
- [ ] Retry protocol rewrite (rollback past setType, remount router)
- [ ] Build pipeline extended to produce 4 branches

**Exit criteria:** generation for each of the three modes is correct
(compile gate passes on a sample prompt per mode); system-cache.bin is
~20% larger than today; first-token latency for decode unchanged; tok/s
unchanged or better (smaller working KV window).

### v2 — nested sub-menus (not in this PR)

- [ ] RoPE-shift runtime pass (K-only, per layer)
- [ ] Multi-level mount stack (mount onto already-mounted branch)
- [ ] Finer SDK states — `addBox({…})` param help, `connect(src, dst)`
  argument help
- [ ] Grammar actions can also fire `onExit`

Deferred until v1 demonstrates value.

---

## Edge cases

1. **Model never calls setType.** Stay on router. Router must be a
   valid self-contained prompt (it is — today's 3005-tok prompt is a
   superset). No regression.

2. **Model calls setType with an invalid arg** (e.g., `setType("foo")`).
   Grammar masks — only "sequence" | "architecture" | "uml" pass. Can't
   happen.

3. **Model calls setType twice.** Second call is still grammar-valid if
   the grammar allows mode switching. For v1, grammar masks a second
   setType after the first one commits. The model's trained behaviour
   is to call it once anyway.

4. **Retry exhaustion.** After N retries (currently 3?), give up and
   surface the best-effort output, same as today.

5. **System-cache.bin version mismatch.** Magic + version header; on
   mismatch, fall back to runtime prefill of router only, no branches.
   Slow first session, still works.

6. **KV budget overflow.** Worker allocates cache for
   `routerLen + max(branchLen) + generationBudget`. Checked at init.

---

## Open questions

1. **How much does a branched system-cache.bin grow?** Today's is
   68.9 MB (3005 tokens, TQ-compressed). If each branch is ~1800 tokens
   we're looking at router + 3×1800 = ~5900 tokens baked, ~135 MB. Is
   that acceptable? Gzip during asset emit would help but TQ data is
   already near-random, bad compression ratio.

2. **Is the model actually better in specialised mode?** Needs
   perplexity-on-mode-specific-prompt measurement before and after to
   prove value. It's plausible — today's 3005-tok prompt includes 2/3
   "other mode noise" — but not demonstrated yet.

3. **Do we need an `unmount` / `exitMode` for v2?** If the SDK supports
   changing modes mid-diagram (currently it doesn't in practice), we'd
   need pop semantics. For v1 mode is terminal; no pop needed.

4. **Does RoPE re-shift survive TurboQuant compression?** K is stored
   TQ-compressed. Re-shifting requires decode → rotate → re-encode. The
   decode-encode roundtrip is lossy (polar quantisation); re-encoding a
   re-rotated K might drift from the reference RoPE'd K by more than an
   ULP. v2 blocker, needs conformance test.

---

## Names considered

- **Dynamic context window** — what user called it, keeps
- Scoped context
- Grammar-scoped KV
- Context sub-shells
- KV mount points
- Context filesystem / `mount`/`umount`

We use "dynamic context window" in docs and commit messages.
