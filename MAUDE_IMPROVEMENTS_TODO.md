# MAUDE Speed & Efficiency TODO

Focus: make MAUDE faster and cheaper without feeling dumber.
Grounded in current stack: `fast_dispatch`, auto-router, tool groups, ~136 tools, Nemotron + subagents, in-memory tool cache.

---

## 1. Context hygiene — DONE ✅

Long chats get slow and dumb. Clean context first; everything else compounds on this.

Implemented in `maude_core/context_hygiene.py`, wired into `chat_local.py`, `gateway/cloud.py`, `maude-client` (incl. portable `maude_client/context_hygiene.py` for Mac/PC), `maude-phone` (`lib/contextHygiene.ts`), memory injection, and MemPalace.

- [x] **Sliding window + rolling summary** of older turns (keep recent turns verbatim; compress the rest)
- [x] **Top-k memory retrieval only** — never dump the whole palace; retrieve relevant chunks only
- [x] **Drop old tool payloads** once the next step is confirmed (browse/search/file dumps should not linger forever)
- [x] **Per-mission scratch state** instead of infinite message history (`scratch_*` tools + prompt injection)
- [x] **Hard-truncate tool results** before re-injecting into context (cap chars/tokens per tool, per turn)
- [x] **Strip redundant system/tool preamble** from repeated turns (dedupe consecutive system msgs; collapse stacked summaries)

### Env knobs
| Var | Default | Meaning |
|---|---|---|
| `MAUDE_CTX_KEEP_RECENT_TURNS` | 12 | Non-system messages kept verbatim |
| `MAUDE_CTX_KEEP_TOOL_ROUNDS` | 2 | Recent tool rounds kept full |
| `MAUDE_CTX_MAX_TOOL_CHARS` | 4000 | Hard cap per tool result |
| `MAUDE_CTX_MEMORY_TOP_K` | 5 | Memories injected into system prompt |
| `MAUDE_CTX_TOKEN_BUDGET` | 0 → 75% of `MAUDE_NUM_CTX` | History token budget |

### Acceptance criteria
- [x] Long multi-tool sessions stay under a defined token budget
- [x] Memory recall injects top-k snippets, not full dumps
- [x] Tool results older than N turns are summarized or dropped
- [x] Tests: `tests/test_context_hygiene.py`

---

## 2. Shrink what the model sees every turn — DONE ✅

Default turn no longer dumps browser/social/sandbox/etc. Full schemas load on
keyword match, sticky session activation, or `activate_tool_domain`.

Implemented in `maude_core/tool_groups.py`, wired into `chat_local.py`,
`gateway/cloud.py`, `run_telegram.py`, `tool_catalog.py`, and `maude-client`.
Phone sends `session_id` per conversation so sticky domain activation works
on the gateway tool-tier path.

- [x] **Tier tools**
  - Always-on (~22): file / shell / web search / memory / scratch / plan + domain controls
  - Session-activated (sticky): browser / google / media / social / github / …
  - Rare: substack / forge / hyperframes (keyword or activate; sticks for multi-step)
- [x] **Lazy tool schemas** — `list_tool_domains` carries names + one-liners; full schemas only for always-on + active domains. Optional `domain_*` stubs via `MAUDE_LAZY_TOOL_STUBS=1`
- [x] **Cap tool results** at injection time (shared with context hygiene caps)
- [x] Audit always-on set — removed browser suite, social_post, dead sandbox_* from core

### Env knobs
| Var | Default | Meaning |
|---|---|---|
| `MAUDE_LAZY_TOOL_STUBS` | `0` | If `1`, emit per-domain `domain_*` stub tools for inactive domains |
| `MAUDE_SESSION_ID` | `default` | Sticky domain activation key |

### Acceptance criteria
- [x] Default turn tool schema payload ~11k chars / 22 full tools vs full catalog ~86k / 176 tools
- [x] Session activation works for browser/google/media (keywords, `activate_tool_domain`, tool history)
- [x] Tests: `tests/test_tool_tiers.py`

---

## 3. Expand fast paths (skip the LLM) — DONE ✅

`fast_dispatch` and `auto_router` already avoid full tool-selection loops for obvious intents. Expanded.

Implemented in `maude_core/fast_dispatch.py` (+ client `tool_router.py` mirrors for local ops).
Wired into `chat_local.py`, `run_telegram.py`, `maude-client`, and **gateway
OpenAI + Claude tool loops** so phone/web get the same list/read/shell/memory/image/URL
skip without a first tool-selection LLM hop.

| Pattern | Skip to |
|---|---|
| “list files in X”, “cat/read this path” | `list_directory` / `read_file` |
| “run tests / git status / docker ps” | `run_command` (whitelist only) |
| “check memory / what do you know about X” | `recall_memory` / `list_memories` |
| “generate image of …” | `generate_image` |
| “summarize this URL” | `web_browse` → short LLM summary |

- [x] Add filesystem list/read fast path
- [x] Add common shell status commands (`git status`, `docker ps`, test runners) fast path
- [x] Add memory-recall fast path
- [x] Add image-gen intent fast path
- [x] Add URL-summarize fast path (browse + small model via existing summary hop)
- [x] Log hit rate so we know which patterns save real round-trips (`get_fast_dispatch_stats()`)

### Guards
- Multi-step cues (`and then`, `fix`, long messages) skip fast path
- Shell is a **whitelist** only — never free-form shell from regex
- “remember that …” does not trigger recall
- Tool errors fall through to normal LLM loop

### Acceptance criteria
- [x] Each pattern above resolves without a full tool-selection LLM loop when confidence is high
- [x] Fast-path misses fall through cleanly to normal routing
- [x] Tests: `tests/test_fast_dispatch.py`

---

## 4. Route by difficulty, not just domain

Auto-router maps code/vision/writing well. Add a **complexity ladder**:

1. **Tiny local model** (or pure rules) — greetings, status, simple file ops
2. **Specialist local** — codestral / llava / writer
3. **Main Nemotron-30B** — multi-step local work
4. **Frontier cloud** — hard reasoning only

- [ ] Classify request difficulty (rules + light classifier)
- [ ] Default to cheapest tier that can finish
- [ ] Escalate on failure / low confidence / explicit user ask
- [ ] Never start at 30B/frontier for trivial turns

### Acceptance criteria
- Trivial turns rarely hit Nemotron-30B or cloud
- Escalation path is observable in logs/metrics

---

## 5. Parallelize independent work harder

Read-only tools already run concurrently. Push further.

- [ ] Fan out research/code/vision subagents in parallel when plans have independent branches
- [ ] Prefer **plan → parallel execute → verify** over serial think→tool→think→tool loops
- [ ] Bound tool-loop iterations hard (e.g. max 6–8 before escalate/summarize)
- [ ] Detect independent vs dependent steps in the plan before fan-out

### Acceptance criteria
- Multi-branch research/code jobs run branches concurrently when safe
- Tool loops cannot spin unbounded

---

## 6. Make caching durable and broader

`ToolCache` is TTL in-memory for web/vision only.

- [ ] Persist cache to SQLite/disk across restarts
- [ ] Cache Drive/Gmail list queries (short TTL)
- [ ] Cache calendar “today” (short TTL)
- [ ] Cache git status (very short TTL)
- [ ] Cache embeddings of stable docs
- [ ] Optional **prompt cache** on cloud providers (Anthropic/OpenAI) for repeated system+tool prefixes

### Acceptance criteria
- Cache survives process restart
- Repeated identical web/vision/list queries hit cache within TTL

---

## 7. Keep the hot path warm

Cold model loads kill snappy MAUDE.

- [ ] Keep primary + codestral (and maybe llava) preloaded
- [ ] Idle policy: demote heavy models after N minutes; keep a small always-on responder
- [ ] Speculative decoding / MTP if vLLM/llama.cpp path supports it for main chat model
- [ ] Warmup on service start (healthcheck that actually loads weights)

### Acceptance criteria
- First real user message after idle does not pay full cold-load cost for primary model
- Idle GPU policy is documented and tunable

---

## 8. Gateway efficiency

- [ ] HTTP connection pooling / keep-alive for cloud providers
- [ ] Retries only on transient errors
- [ ] Timeouts that fail fast and fall back to local
- [ ] Extend transient retry/error recovery to plain provider proxy and frontier calls

### Acceptance criteria
- Cloud calls reuse connections under load
- Transient failures retry; permanent failures fail fast
- Local fallback triggers when cloud times out

---

## 9. Skills as compiled workflows

For repeated jobs (job apps, Pixelus product shots, Flux gens, PR babysit):

- [ ] Turn high-frequency jobs into **fixed skills/workflows** with minimal LLM choice points
- [ ] LLM fills slots; deterministic steps do the rest
- [ ] Less tool-selection thrash = faster and more reliable
- [ ] Inventory top repeated workflows and convert the top 3 first

### Acceptance criteria
- Top repeated workflows have a compiled path with fewer LLM decision points
- Slot-filling still works when inputs vary

---

## 10. Verification that doesn't thrash

- [ ] Lightweight verify steps (lint/tests/schema checks) before expensive re-generation
- [ ] Bound rework loops (max N fix iterations then surface to user)
- [ ] Prefer structured checks over another full model pass when possible

### Acceptance criteria
- Failed verification does not open an unbounded regen loop
- Cheap checks run before expensive ones

---

## Implementation order

1. **Context hygiene** — DONE
2. **Shrink always-on tools / lazy schemas** — DONE
3. **Expand fast paths** — DONE
4. Difficulty ladder
5. Parallel fan-out + loop bounds
6. Durable cache
7. Warm models + idle policy
8. Gateway pooling/timeouts
9. Compiled skills for top workflows
10. Verification bounds

---

## Metrics to track

- [ ] Tokens per turn (system + tools + history + results)
- [x] Fast-path hit rate (`get_fast_dispatch_stats()`)
- [ ] Median / p95 time-to-first-token
- [ ] Cloud vs local call share
- [ ] Tool-loop iteration count
- [ ] Cache hit rate

---
*Scrubbed and replaced: 2026-08-11*
*Source: speed/efficiency plan; priority starts at context hygiene*
