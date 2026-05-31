# MAUDE TODO

## Current Direction

MAUDE is evolving from an assistant with tools into a local Agent OS: durable missions, executable workflows, persistent memory, local multimodal generation, browser control, publishing, and feedback loops that can own real projects across sessions.

## Done

- [x] Multi-agent architecture with specialized subagents
- [x] Cross-machine task execution through the collaboration system and client task executor
- [x] Auto-routing with intent detection and model switching
- [x] Cloud model integration for Claude, Mistral, Codestral, Devstral, Grok, and Gemini
- [x] Local model inference with Nemotron on llama-server
- [x] Streaming responses with typewriter effect
- [x] Google Workspace integration for Gmail, Drive, Sheets, Slides, Calendar, Contacts, and YouTube
- [x] Voice interaction with Nemotron ASR, Magpie TTS, and tool access
- [x] Mobile app through iOS, Capacitor, and EventSource streaming
- [x] Tool system with fast dispatch, keyword-based tool groups, scoped access, and catalog tests
- [x] Image analysis through LLaVA
- [x] Web search and browsing
- [x] Gateway with HTTPS and HTTP mirror
- [x] OAuth app verification for persistent Google tokens
- [x] Parallel tool execution for read-only tools while mutating tools stay sequential
- [x] Pytest suite organized around unit coverage, integration coverage, and gateway HTTP smoke tests
- [x] Phone location awareness
- [x] Long-term memory and context retention across sessions
- [x] PDF and Office document parsing tools
- [x] Mission Kernel tools: `mission_create`, `mission_list`, `mission_get`, `mission_update`, `mission_log`, and `mission_brief`
- [x] Durable mission storage in `data/missions/*.json`
- [x] First Agent OS mission created: `maude-agent-os-missions-3dc7a45f`
- [x] Mission lifecycle and tool catalog tests
- [x] Mission runner: `mission_run_next` executes stored or supplied `execute_plan` stages and logs results
- [x] Mission templates in `mission_create`: `content_engine`, `research_lab`, `codebase_steward`, `personal_ops`, and `startup_builder`
- [x] Mission checkpoint model with next action, recent logs, artifacts, blockers, cadence, and success criteria
- [x] Manual mission tick support with `mission_tick`
- [x] Scheduler integration for recurring mission ticks with `mission_schedule`
- [x] Scheduled missions can drain consecutive executable steps with `mission_drain` instead of relying on prompt tweaks
- [x] Mission dashboard data in Command Center plus a phone Missions tab
- [x] First real Content Engine mission created: `content-engine-maude-agent-os-shorts-39e232fb`
- [x] X video posting path tolerates previewless video uploads and waits for post readiness before failing

## Active Mission: Agent OS / Missions

- [x] Add `mission_run_next` to execute the next mission step through `execute_plan`, log the result, and update mission state
- [x] Add mission templates for `content_engine`, `research_lab`, `codebase_steward`, `personal_ops`, and `startup_builder`
- [x] Expand the mission checkpoint model with objective, status, next action, recent logs, artifacts, blockers, cadence, and success criteria
- [x] Add recurring mission support with a manual `mission_tick` first
- [x] Add scheduler integration for recurring mission ticks
- [x] Build a mission dashboard or phone view showing active missions, progress, next actions, logs, artifacts, and blockers
- [x] Create a Content Engine mission as the first real end-to-end workflow
- [x] Make Content Engine mission steps executable with stored plans for research, script drafting, render review, publishing checkpoints, and performance review

## Overnight Update: May 31, 2026

Tonight's 2 AM MAUDE self-improvement run should focus on turning the current WIP into more reliable operating loops, not broad refactors.

- [ ] Confirm the scheduler still has the nightly self-improvement mission enabled and that completed recurring missions reset into the next cycle.
- [x] Make scheduled mission ticks flow through consecutive executable steps automatically with `mission_drain`.
- [ ] Make the nightly mission execute one scoped improvement automatically: select a weekly TODO, apply a local change, run focused validation, write the report, and queue the next target.
- [ ] Validate the X/Twitter video upload fix with the focused `social_posting` tests and, if a live session is available, a guarded dry-run or non-publishing readiness check.
- [ ] Run a lightweight readiness pass for gateway HTTP, mission tools, HyperFrames doctor, and the mission directory.
- [ ] Review the active Pixelus and Agent OS content missions for stale blockers, missing artifacts, and next actions.
- [ ] Stabilize current WIP before adding broad features: inspect changed files, identify risky/unrelated changes, run targeted tests, and leave a small commit/handoff plan.
- [ ] Produce a short nightly report with actions taken, failed checks, timing notes, and the next concrete fix.

## Improvement TODO: Week of June 1, 2026

- [ ] Make Content Engine mission plans produce real first drafts instead of `TBD` placeholders by reading mission memory, recent artifacts, and current project notes before writing step artifacts.
- [x] Make scheduled mission tasks auto-drain stored-plan steps instead of depending on natural-language prompt instructions.
- [x] Record scheduler task status, duration, failure count, and next run even when a scheduled task errors.
- [ ] Make the nightly self-improvement mission execute improvements end-to-end instead of only reporting: choose one TODO, make a scoped local change, validate it, and update the next night's queue.
- [ ] Finish X/Twitter video publishing reliability: handle previewless accepted uploads, disabled Post button states, retry/reset flows, and clear upload error capture.
- [ ] Add integration-style tests for the social posting browser flow using mocked page states for successful video, failed upload, and previewless upload.
- [ ] Enforce video pre-publish guardrails at every publish entrypoint: YouTube upload, X/social post, and any future content mission publish step.
- [ ] Add scheduler observability: last run, next run, duration, exit status, repeated failure count, and a compact mission dashboard summary.
- [ ] Improve mission runner safety so external side effects require explicit publish/delete/spend approval, while local artifact and test steps can run unattended.
- [ ] Improve memory retrieval quality by logging what memories were used, suppressing low-signal recalls, and adding focused tests around durable memory search.
- [ ] Add gateway/tool latency telemetry for slow calls and use it in nightly reports to pick one concrete optimization target.
- [ ] Reduce token-heavy readiness checks by replacing broad file reads/searches with targeted health probes and concise summaries.
- [ ] Stabilize the current uncommitted WIP into small reviewable commits or a clear branch handoff once tests are green.

## Mission Ideas

- [x] Content Engine Mission: research, script, generate visuals, render videos, publish, analyze results, and improve the next run
- [ ] Research Lab Mission: monitor papers and news, summarize trends, run experiments, and produce reports
- [ ] Codebase Steward Mission: watch a repo, fix bugs, write tests, improve docs, and prepare PR-ready patches
- [ ] Startup Builder Mission: create landing pages, prototypes, demos, investor updates, and outreach drafts
- [ ] Personal Ops Mission: manage recurring admin, files, reminders, planning, and purchases

## Infrastructure

- [x] Gateway connection pooling for cloud API requests
- [x] Extend transient retry and error recovery to cloud tool-loop provider calls and legacy frontier calls
