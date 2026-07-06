# MAUDE TODO

## Current Direction

MAUDE is being refactored from a powerful local assistant/workbench into a clearer enterprise-style agent platform: authenticated client surfaces, governed gateway, orchestration layer, provider abstraction, model routing, queue-based jobs, stateless workers, caching, rate limiting, retries/backoff, prompt/model versioning, typed memory, observability, verification gates, and an eval harness.

## Done

- [x] Multi-agent architecture with specialized subagents
- [x] Cross-machine task execution through the collaboration system and client task executor
- [x] Auto-routing with intent detection and model switching
- [x] Cloud model integration for Claude, Mistral, Codestral, Devstral, Grok, Gemini, and OpenAI-compatible providers
- [x] Local model inference through local model servers
- [x] Gateway with HTTPS, HTTP mirror, streaming, websocket/mobile support, and shared file routes
- [x] Telegram, terminal, Python client, and mobile app access paths
- [x] Google Workspace integration for Gmail, Drive, Sheets, Slides, Calendar, Contacts, and YouTube
- [x] Tool system with registry dispatch, tool groups, scoped access, caching, rate limiting, and catalog tests
- [x] Parallel tool execution for read-only tools while mutating tools stay sequential
- [x] Web search, browsing, file tools, PDF/Office parsing, image analysis, media generation, voice, VNC/browser control, Substack, GitHub, and social posting tools
- [x] Mission Kernel tools, durable mission storage, mission templates, checkpoints, scheduled ticks, mission drain, dashboard data, and phone Missions tab
- [x] Typed memory ledger with semantic, episodic, procedural, working, preference, identity, person, project, mission, and artifact memory types
- [x] Memory save verification gate that checks both JSONL persistence and markdown projection
- [x] Enterprise architecture refactor doc in `docs/enterprise-architecture-refactor.md`
- [x] Initial `src/maude/` enterprise package shell with orchestration, providers, tools, memory, gateway, observability, evals, verification, and prompt versioning boundaries
- [x] Architecture boundary tests for model routing, queue jobs, stateless workers, retries, cache, rate limiting, prompt versioning, observability, eval harness, verification gates, gateway policy, and legacy provider capability loading

## Enterprise Refactor: Remaining Work

- [x] Move gateway implementation from `gateway/` into `src/maude/gateway/` behind compatibility entry points.
- [x] Move orchestration runtime from `execution.py`, `agent_executor.py`, `auto_router.py`, `subagents.py`, `scheduler.py`, and `maude_core/execute.py` into `src/maude/orchestration/`.
- [x] Replace in-process `JobQueue` with durable queue-backed job storage and dead-letter handling.
- [x] Add stateless worker process entry points for queued jobs.
- [x] Wire retries/backoff into provider calls, tool calls, queued jobs, and integration adapters.
- [x] Move provider/model code from `providers.py`, `routing.py`, `frontier.py`, and `gateway/cloud.py` into `src/maude/providers/` or a dedicated `src/maude/models/` package.
- [x] Add model aliases, routing policy, provider health checks, fallback policy, cost controls, latency controls, and local/private routing rules.
- [x] Record prompt version, model version, provider, routing decision, token usage, latency, and cost metadata on every model call.
- [x] Split `maude_core/tool_defs.py` into domain-owned tool schemas under `src/maude/tools/domains/`.
- [x] Move tool handlers from `maude_core/tools_*`, `google_tools.py`, `browser.py`, `social_posting.py`, and related top-level modules into domain packages with compatibility shims.
- [x] Add tool risk classification, argument validation, result validation, audit events, and approval gates for publish/delete/spend/external side effects.
- [x] Move memory, conversation, mission, workflow state, and retrieval code into `src/maude/memory/` with scoped user/project/workspace access.
- [x] Add retention, deletion, export, provenance, and access-aware retrieval controls for memory/state.
- [x] Move integration code into `src/maude/integrations/` packages for Google, GitHub, Substack, browser, voice, VNC, Replicate/media, HyperFrames, and social posting.
- [x] Normalize integration errors and add adapter-level retries, token refresh, audit events, and test doubles.
- [x] Move runtime state out of source-like folders and behind config: `runtime/data`, `runtime/shared`, `runtime/transfers`, logs, certs, caches, and generated media.
- [x] Add config-driven path management so source modules do not hard-code runtime storage locations.
- [ ] Add structured observability events across gateway, orchestration, model routing, tools, memory, integrations, and workers.
- [ ] Add trace IDs/run IDs to every model call, tool call, job, memory write, and external action.
- [ ] Add metrics for latency, token usage, cost, cache hit rate, retry count, queue depth, job duration, and failure rate.
- [ ] Expand the eval harness with golden tasks for memory retrieval, tool-call correctness, planning, model routing, and mission execution.
- [ ] Add CI-style verification gates for lint, focused tests, architecture boundary tests, eval smoke tests, schema checks, and side-effect policy checks.
- [ ] Add release/readiness command that runs the verification suite and writes a compact report.
- [ ] Add packaging/import cleanup so all new code imports through `maude.*` and legacy top-level modules become thin shims.
- [ ] Update README/docs with the enterprise architecture, local development commands, package boundaries, and migration rules for adding new features.

## Product/Workflow Backlog

- [ ] Make Content Engine mission plans produce real first drafts instead of `TBD` placeholders by reading mission memory, recent artifacts, and current project notes before writing step artifacts.
- [ ] Make the nightly self-improvement mission execute one scoped improvement automatically: choose a TODO, apply a local change, validate it, write the report, and queue the next target.
- [ ] Finish X/Twitter video publishing reliability: disabled Post button states, retry/reset flows, clear upload error capture, and guarded dry-run readiness checks.
- [ ] Add integration-style tests for the social posting browser flow using mocked page states for successful video, failed upload, and previewless upload.
- [ ] Enforce video pre-publish guardrails at every publish entrypoint: YouTube upload, X/social post, and future content mission publish steps.
- [ ] Add scheduler observability: last run, next run, duration, exit status, repeated failure count, and compact mission dashboard summary.
- [ ] Improve mission runner safety so external side effects require explicit publish/delete/spend approval while local artifact and test steps can run unattended.
- [ ] Improve memory retrieval quality by logging what memories were used, suppressing low-signal recalls, and adding focused durable memory search tests.
- [ ] Reduce token-heavy readiness checks by replacing broad file reads/searches with targeted health probes and concise summaries.

## Mission Ideas

- [x] Content Engine Mission: research, script, generate visuals, render videos, publish, analyze results, and improve the next run
- [ ] Research Lab Mission: monitor papers and news, summarize trends, run experiments, and produce reports
- [ ] Codebase Steward Mission: watch a repo, fix bugs, write tests, improve docs, and prepare PR-ready patches
- [ ] Startup Builder Mission: create landing pages, prototypes, demos, investor updates, and outreach drafts
- [ ] Personal Ops Mission: manage recurring admin, files, reminders, planning, and purchases
