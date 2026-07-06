# MAUDE Enterprise Architecture Refactor

A target architecture for refactoring MAUDE from a powerful local AI workbench into a cleaner enterprise-style agent platform. This is not a rewrite plan for adding bureaucracy. It is a decomposition plan: clear layers, explicit ownership, governed runtime state, and stable contracts between clients, gateway, orchestration, tools, models, memory, integrations, and operations.

## One-Sentence Framing

> I would refactor MAUDE as layered enterprise software: authenticated client surfaces, a governed API gateway, workflow orchestration, provider-abstracted model routing, permissioned tools, scoped memory, integration adapters, observability, evaluation, and explicit operations controls.

## Current Problem

MAUDE already proves the hard technical core: model routing, local/cloud inference, tool execution, multi-client access, memory, scheduled workflows, Google tools, voice, Telegram, shared files, and generated media workflows.

The repo does not yet communicate that maturity. It looks more like a hobby project because source code, generated assets, local runtime state, models, virtual environments, certs, build artifacts, cloned dependencies, service scripts, and application files all sit in or near the same top-level tree. The code organization also mixes top-level Python modules with newer packages such as `maude_core/`, `gateway/`, `channels/`, `skills/`, `maude-client/`, and `maude-phone/`.

The refactor should make the architecture obvious before anyone reads the code.

## High-Level Target Architecture

```text
Client Surfaces
- Terminal TUI
- Python CLI client
- Mobile PWA
- Telegram
- Future web/admin UI
        |
        v
API Gateway + Auth + Governance
- request validation
- model/tool access control
- rate limits / quotas
- audit logging
- file transfer routes
- streaming/SSE/websocket
        |
        v
Workflow Orchestration Layer
- agent loop
- planner / executor / evaluator
- tool execution
- scheduling
- retries / checkpoints
- human approval gates
        |
        +--------------------+--------------------+--------------------+
        |                    |                    |                    |
        v                    v                    v                    v
Model Gateway          Tool Platform        Knowledge/Memory      Integration Adapters
- local llama          - tool registry       - conversations       - Google
- OpenRouter           - schemas             - memory ledger       - GitHub
- Mistral              - permissions         - project state       - Substack
- Claude               - validation          - shared files        - browser/VNC
- OpenAI               - safe execution      - workflow state      - voice
- Codex CLI            - result contracts                         - Replicate/media
        |
        v
Evaluation + Guardrails
- schema checks
- tool-call correctness
- golden tasks
- regression tests
- prompt/model comparisons
        |
        v
Observability + Operations
- run IDs / trace IDs
- logs / metrics / costs
- service health
- systemd / Workbench deploy
- secrets / certs
- backup / retention
```

## Layer 1: Client Surfaces And Entry Points

### Purpose

This is how users access MAUDE. Clients should provide good interaction patterns without owning model routing, tool execution, memory, or provider-specific logic.

### Current MAUDE Evidence

- Terminal interface through `maude` / `chat_lite.py`
- Python client under `maude-client/`
- Mobile PWA under `maude-phone/`
- Telegram channel through `run_telegram.py`, `maude_telegram_service.py`, and `channels/telegram.py`
- File sharing through `shared/` and gateway routes

### Responsibilities

- Capture user intent
- Stream intermediate and final results
- Display traces and tool progress
- Support file upload/download flows
- Support model selection without hard-wiring provider logic
- Keep user experience responsive during long-running tasks

### Enterprise Refactor

Move clients into a clear `clients/` boundary:

```text
clients/
  python-cli/
  mobile-pwa/
  telegram/        # if treated as a deployable client surface
```

Server-side channel adapters should live under the platform package:

```text
src/maude/channels/
  base.py
  terminal.py
  telegram.py
  mobile.py
```

### Enterprise Controls

- Authenticated client identity
- Client capability negotiation
- Rate limits per client/user
- Audit context attached to every request
- Stable API contracts instead of internal imports

### Refactor Rule

> Clients call MAUDE APIs. They do not import internal orchestration, tool, or provider modules.

## Layer 2: API Gateway, Auth, And Governance

### Purpose

The gateway is the front door. It should be the single controlled path into model calls, tool execution, shared files, streams, and remote clients.

### Current MAUDE Evidence

- `gateway/` package
- HTTPS gateway on port 30000 and HTTP mirror on 30080
- OpenAI-compatible routes
- SSE trace events
- shared file serving
- mobile/websocket routes
- local/cloud routing integration

### Responsibilities

- Validate inbound requests
- Route requests to orchestration/model/tool layers
- Stream responses and trace events
- Serve shared files and transfers through controlled routes
- Attach run IDs and request metadata
- Enforce user/client policy before work starts

### Enterprise Refactor

Target package:

```text
src/maude/gateway/
  app.py
  routes/
    chat.py
    models.py
    tools.py
    files.py
    health.py
  streaming.py
  websocket.py
  auth.py
  governance.py
```

Keep compatibility entry points during migration:

```text
gateway/__main__.py
python -m gateway
```

### Enterprise Controls

- SSO-ready auth boundary, even if local mode uses a simpler token
- RBAC / capability policy hooks
- request validation
- rate limits and quotas
- audit logs
- tenant/project/user scope fields, even if single-user initially

### Refactor Rule

> The gateway should know how to authenticate, validate, route, and stream. It should not contain provider-specific business logic or giant tool implementations.

## Layer 3: Workflow Orchestration

### Purpose

This is the control plane. It decides what happens first, which model to use, when to call tools, how to handle intermediate results, when to retry, when to enqueue background work, and when to stop or escalate.

### Current MAUDE Evidence

- `execution.py`
- `agent_executor.py`
- `auto_router.py`
- `subagents.py`
- `scheduler.py`
- `maude_core/execute.py`
- `maude_core/tools_plan.py`
- planning, parallel tool execution, scheduled tasks, and subagent routing

### Responsibilities

- Maintain workflow state
- Coordinate model calls and tools
- Support planner / executor / evaluator loops
- Run parallel-safe tool groups
- Enqueue long-running work as queue-based jobs
- Execute background jobs with stateless workers
- Apply retries and exponential backoff for transient failures
- Bound retries and failure handling
- Support scheduled and long-running jobs
- Emit trace events for observability

### Enterprise Refactor

Target package:

```text
src/maude/orchestration/
  engine.py
  execution.py
  planner.py
  agents.py
  subagents.py
  scheduler.py
  jobs.py               # queue-based job contracts
  workers.py            # stateless worker execution
  retries.py            # retry and backoff policy
  cache.py
  rate_limits.py
  trace.py
```

### Enterprise Controls

- explicit input/output contracts per workflow step
- queue-backed durable job records for long-running work
- stateless workers that can pick up jobs from durable state
- checkpoints for long-running workflows
- bounded retries with exponential backoff and dead-letter handling
- human approval gates for risky actions
- deterministic state for scheduled jobs
- trace IDs across all model/tool calls

### Refactor Rule

> Prefer stateful orchestration objects and explicit workflow contracts over scattered function calls across top-level modules.

## Layer 4: Model Gateway, Provider Abstraction, And Model Routing

### Purpose

This layer prevents MAUDE from being hard-wired to one model provider. It normalizes local and cloud model calls behind a common interface, routes requests by model capability and policy, and records which prompt/model version produced each result.

### Current MAUDE Evidence

- `providers.py`
- `routing.py`
- `frontier.py`
- local llama-server routes
- OpenRouter / Mistral / Claude / OpenAI / Codex CLI patterns
- gateway model aliases and fallbacks

### Responsibilities

- Resolve model aliases
- Route to local or cloud providers
- Route by capability, latency, privacy, cost, context size, and tool-support requirements
- Normalize request/response shape
- Support streaming where available
- Track prompt version, model version, provider, and routing decision metadata
- Handle fallback and retry policy
- Report cost, latency, token usage, errors, and provider health

### Enterprise Refactor

Target package:

```text
src/maude/models/
  gateway.py
  router.py
  aliases.py
  contracts.py
  prompt_versions.py
  model_versions.py
  routing_policy.py
  providers/
    base.py
    local_llama.py
    openrouter.py
    mistral.py
    anthropic.py
    openai.py
    codex_cli.py
    replicate.py
```

### Enterprise Controls

- provider abstraction
- model version tracking
- prompt version metadata
- routing-policy metadata on every request
- cost controls
- latency controls
- fallback policy
- retry/backoff policy for provider failures
- provider health checks
- local/private routing policy

### Refactor Rule

> Application workflows ask for model capabilities. Provider adapters decide how those capabilities map to actual models.

## Layer 5: Tool Platform

### Purpose

The tool layer turns MAUDE from chat into an action platform. Enterprise-style tools need schemas, permissions, validation, result contracts, and auditability.

### Current MAUDE Evidence

- `tool_registry.py`
- `tool_catalog.py`
- `maude_core/tool_defs.py`
- `maude_core/tool_groups.py`
- `maude_core/tools_*`
- Google, GitHub, browser, file, web, media, memory, schedule, HyperFrames, missions, video, Substack, and social tools

### Responsibilities

- Register tools
- Define tool schemas
- Group tools by domain and risk
- Validate arguments
- Enforce execution policy
- Return structured results
- Emit tool-call telemetry
- Separate read-only tools from mutating tools

### Enterprise Refactor

Target package:

```text
src/maude/tools/
  registry.py
  catalog.py
  schemas.py
  policy.py
  domains/
    file.py
    web.py
    media.py
    memory.py
    google.py
    github.py
    schedule.py
    missions.py
    hyperframes.py
    video.py
    social.py
    substack.py
```

### Enterprise Controls

- read/write/mutate risk classification
- approval gates for side effects
- sandbox boundaries for shell/code execution
- argument validation
- result validation
- audit logs per tool call
- permission checks by user/client/project

### Current Hotspots

These should be split early because they make the project look monolithic:

- `maude_core/tool_defs.py`
- `google_tools.py`
- `browser.py`
- `social_posting.py`
- `maude_core/tools_missions.py`
- `gateway/cloud.py`

### Refactor Rule

> A new tool should be added to a domain module with schema, handler, risk classification, and tests. It should not enlarge a central mega-file.

## Layer 6: Knowledge, Memory, And State

### Purpose

This layer manages what MAUDE remembers and what state workflows depend on. Enterprise systems need scoped, inspectable, and deletable memory.

### Current MAUDE Evidence

- `memory.py`
- `conversation_sync.py`
- `maude_core/memory_ledger.py`
- `maude_core/memory_utils.py`
- `maude_core/mempalace_utils.py`
- `maude_core/chat_sync.py`
- `data/conversations/`, `data/collab/`, `data/missions/`, `data/workflows/`

### Responsibilities

- Conversation history
- Project/user memory
- Workflow state
- Mission state
- Shared files and transfers metadata
- Memory retrieval and summarization
- Retention/deletion policy

### Enterprise Refactor

Target package:

```text
src/maude/memory/
  store.py
  conversation.py
  ledger.py
  workflow_state.py
  mission_state.py
  retrieval.py
  retention.py
```

Runtime paths should move behind config:

```text
runtime/
  data/
  shared/
  transfers/
```

### Enterprise Controls

- memory scoping by user/project/workspace
- explicit retention policy
- deletion/export controls
- access-aware retrieval
- provenance metadata
- validation before memory reuse

### Refactor Rule

> Source modules should never hard-code runtime storage locations. All runtime paths go through platform path config.

## Layer 7: Integration Adapters

### Purpose

Integrations should be isolated adapters, not logic spread through orchestration, gateway, and tool modules.

### Current MAUDE Evidence

- `google_tools.py`
- `maude_core/tools_google.py`
- `maude_core/tools_github.py`
- `substack_tools.py`
- `maude_core/tools_substack.py`
- `browser.py`
- `browser_workflows.py`
- `camofox.py`
- `vnc_session.py`
- `novnc.py`
- `voice.py`
- `voice_server.py`
- `gateway/replicate.py`

### Responsibilities

- Wrap external APIs
- Manage auth/token refresh
- Normalize external response shapes
- Handle API-specific errors
- Keep provider credentials out of business logic
- Provide testable boundaries for mocks/fakes

### Enterprise Refactor

Target package:

```text
src/maude/integrations/
  google/
  github/
  substack/
  browser/
  voice/
  vnc/
  replicate/
  hyperframes/
```

### Enterprise Controls

- secrets management
- token refresh and revocation
- adapter-level retries
- external API error normalization
- audit events for external actions
- test doubles for integration tests

### Refactor Rule

> Tool handlers call integration adapters. They do not directly own OAuth, HTTP client plumbing, or provider-specific response parsing.

## Layer 8: Evaluation, Guardrails, And Quality

### Purpose

Enterprise AI platforms need behavioral testing, not only unit tests. This layer measures whether workflows, tools, prompts, routing, and retrieval keep working as the system changes.

### Current MAUDE Evidence

- `tests/`
- gateway API tests
- tool execution tests
- memory ledger tests
- plan execution tests
- mission tests
- social posting tests
- health checks

### Responsibilities

- Unit tests for deterministic functions
- Integration tests for gateway/tool boundaries
- Smoke tests for command entry points
- Golden tasks for agent workflows
- Tool-call correctness checks
- Regression tests for prompts/model routing
- Failure-mode tests for provider outages

### Enterprise Refactor

Target structure:

```text
tests/
  unit/
  integration/
  smoke/
  evals/
```

### Enterprise Controls

- golden task sets
- deterministic fixtures
- offline provider mocks
- CI smoke tests
- versioned eval results
- release gates for high-risk tools

### Refactor Rule

> Every package move should include import smoke tests and no behavior change. Every risky tool should have at least schema, permission, and failure-path tests.

## Layer 8A: Verification Layer

### Purpose

Verification is the system's proof mechanism. It confirms that an action actually happened, that an output matches the requested contract, and that downstream state should be trusted. This is separate from evaluation: verification happens inside live workflows; evaluation measures system quality across test sets and releases.

### Responsibilities

- Verify tool side effects before reporting success
- Check files exist after write/copy/export operations
- Validate generated artifacts are readable, non-empty, and in the expected format
- Confirm external API operations with read-after-write checks where possible
- Validate structured outputs against schemas
- Compare expected and actual workflow state transitions
- Mark unverified results explicitly instead of treating them as successful

### Target Package

```text
src/maude/verification/
  contracts.py          # shared verification result types
  artifacts.py          # files, images, video, PDFs, generated outputs
  tools.py              # tool result and side-effect verification
  workflows.py          # state transition verification
  external.py           # read-after-write checks for APIs
  policies.py           # what must be verified by risk class
```

### Verification Contracts

Every tool and workflow step should be able to return:

```text
verified: true | false
confidence: high | medium | low
evidence: file path, API response id, checksum, row count, screenshot, log excerpt
reason: human-readable explanation
next_action: continue | retry | repair | escalate | stop
```

### Risk-Based Verification

| Risk Class | Examples | Required Verification |
| --- | --- | --- |
| Read-only | list files, search, web fetch | result schema and source metadata |
| Write | create file, edit doc, export PDF | file exists, checksum/size, parse/readback |
| Execute | shell command, render job, script run | exit code, stderr review, expected artifact check |
| External | Gmail, Drive, GitHub, Telegram, Substack | API response plus read-after-write when available |
| Destructive | delete, overwrite, publish, send | approval gate plus post-action confirmation |

### Refactor Rule

> A mutating tool is not complete until it can prove what changed.

## Layer 8B: Observability Layer

### Purpose

Observability makes model/tool workflows debuggable. It should be possible to answer: what ran, why it ran, which model was used, which tools were called, what failed, how long it took, what it cost, and what evidence supports the final answer.

### Responsibilities

- Generate run IDs, trace IDs, step IDs, model-call IDs, and tool-call IDs
- Emit structured logs for model calls, tool calls, workflow decisions, retries, and verification results
- Track token usage, latency, provider, model, cost, errors, retries, and cache hits
- Preserve trace timelines for debugging and future eval analysis
- Provide local dashboards or report files for inspection
- Redact secrets and sensitive user content from logs

### Target Package

```text
src/maude/observability/
  trace.py              # run/trace/span IDs
  events.py             # structured event model
  logger.py             # structured logging setup
  metrics.py            # latency/cost/token/error counters
  redaction.py          # secret and PII redaction
  exporters.py          # jsonl, sqlite, console, future OTEL
  reports.py            # local run summaries
```

### Required Events

```text
run.started
run.completed
run.failed
model.requested
model.completed
model.failed
tool.requested
tool.completed
tool.failed
verification.completed
workflow.step.started
workflow.step.completed
workflow.retry
approval.required
approval.granted
approval.denied
```

### Storage

Local-first default:

```text
runtime/logs/events.jsonl
runtime/logs/runs.sqlite
runtime/logs/service.log
```

Future enterprise-compatible export:

```text
OpenTelemetry traces
Prometheus metrics
structured JSON logs
SIEM/audit export
```

### Refactor Rule

> If a workflow cannot be traced after the fact, it is not production-ready.

## Layer 8C: Evaluation Harness

### Purpose

Evaluation measures whether MAUDE is getting better or worse over time. It protects against regressions caused by prompt changes, model swaps, provider outages, tool changes, and refactors.

### Responsibilities

- Maintain golden task sets for common workflows
- Score final answers, tool choices, tool arguments, verification behavior, and recovery behavior
- Compare model/provider performance across tasks
- Run offline evals with mocked providers when possible
- Run live evals selectively for model/tool integration quality
- Store eval results by git commit, model version, prompt version, and tool catalog version

### Target Package

```text
evals/
  datasets/
    tool_use.jsonl
    file_ops.jsonl
    web_research.jsonl
    coding_tasks.jsonl
    google_workspace.jsonl
    missions.jsonl
  runners/
    run_eval.py
    compare_models.py
    score_tool_calls.py
  reports/
    latest.md

src/maude/evaluation/
  datasets.py
  runner.py
  scorers.py
  judges.py
  reports.py
```

### Eval Dimensions

| Dimension | What It Measures |
| --- | --- |
| Task completion | Did the workflow satisfy the user request? |
| Tool selection | Did the agent choose appropriate tools? |
| Tool arguments | Were arguments correct and bounded? |
| Verification | Did it verify side effects before claiming success? |
| Grounding | Did it cite or use source material correctly? |
| Safety | Did it avoid unauthorized/destructive actions? |
| Efficiency | Did it avoid unnecessary model/tool calls? |
| Recovery | Did it repair failures instead of hallucinating success? |

### Release Gates

Before major refactors merge:

```text
unit tests pass
integration tests pass
smoke tests pass
import boundary tests pass
golden tool-use eval does not regress
gateway smoke eval passes
high-risk tool verification tests pass
```

### Refactor Rule

> Model quality and agent behavior must be measured across versions, not judged by one successful demo.

## Layer 8D: Guardrails, Policy, And Approval Gates

### Purpose

Guardrails define what the system is allowed to do, when it must ask for approval, and how risky actions are controlled. This is the governance layer around tools and workflows.

### Responsibilities

- Classify tools by risk
- Require approval for destructive, publishing, financial, credential, or external-send actions
- Enforce path allowlists and workspace boundaries
- Block secrets from being sent to models or logs
- Validate requested actions against user/client permissions
- Support dry-run mode for risky workflows

### Target Package

```text
src/maude/policy/
  risk.py
  permissions.py
  approvals.py
  path_policy.py
  secret_policy.py
  tool_policy.py
  workflow_policy.py
```

### Policy Examples

```text
read_file: allowed by default inside workspace
write_file: allowed inside workspace, verify after write
run_command: requires command policy check
rm/delete: approval required
send_email/send_telegram/publish: approval or explicit user request required
edit_external_doc: read-after-write verification required
access_secret: never logged, never sent to model context
```

### Refactor Rule

> Tool execution should pass through policy before it reaches the handler.


## Layer 9: Observability And Operations

### Purpose

Operations is what makes a platform maintainable after the demo works. MAUDE needs clean service ownership, logs, runtime directories, health checks, and deployment scripts.

### Current MAUDE Evidence

- `start_*.sh`
- `setup_*.sh`
- `maude-telegram.service`
- `services/*.service`
- `logs/`
- `certs/`
- `variables.env`
- gateway logs and health endpoints
- systemd-style service usage

### Responsibilities

- Start/stop services
- Health checks
- Logs and traces
- Runtime data management
- Cert management
- Secrets management
- Backups and retention
- Workbench deployment hooks

### Enterprise Refactor

Target structure:

```text
deploy/
  scripts/
  systemd/
  workbench/
  certs-template/

docs/operations/
  service-map.md
  local-dev.md
  production-like-runbook.md
  secrets-and-certs.md
```

Runtime state:

```text
runtime/
  data/
  logs/
  certs/
  shared/
  transfers/
  models/
```

Environment variables:

```text
MAUDE_HOME
MAUDE_RUNTIME_DIR
MAUDE_DATA_DIR
MAUDE_SHARED_DIR
MAUDE_TRANSFER_DIR
MAUDE_MODEL_DIR
MAUDE_LOG_DIR
MAUDE_CERT_DIR
```

### Enterprise Controls

- run IDs and trace IDs
- structured logs
- service health checks
- startup smoke tests
- secrets out of git
- cert generation separated from generated certs
- runtime data excluded from source

### Refactor Rule

> Deployment and runtime operations should be explicit enough that another engineer can start, stop, debug, and update MAUDE without knowing its history.

## Layer 10: Repository And Packaging

### Purpose

The repository should communicate product architecture immediately.

### Target Layout

```text
terminal-llm/
  README.md
  pyproject.toml
  uv.lock

  src/
    maude/
      platform/
      gateway/
      orchestration/
      models/
      tools/
      memory/
      integrations/
      channels/
      services/
      cli/

  clients/
    python-cli/
    mobile-pwa/

  deploy/
    scripts/
    systemd/
    workbench/

  docs/
    architecture/
    operations/
    product/
    screenshots/

  tests/
    unit/
    integration/
    smoke/
    evals/

  runtime/       # ignored
```

### What Moves Out Of The Source View

- `venv/`
- `.venv/`
- `models/`
- `llama.cpp/`
- `shared/` generated files
- `data/` runtime state
- `certs/` generated certs
- `logs/`
- `transfers/`
- `maude-phone/node_modules/`
- generated media and videos

### What May Stay Tracked Deliberately

- source code
- tests
- docs
- service templates
- Workbench config
- curated screenshots
- client source
- minimal sample configs
- `.gitkeep` placeholders for runtime dirs if needed

## Required Platform Capabilities And Exact Homes

| Capability | Exact Home | What It Does | First Implementation |
| --- | --- | --- | --- |
| Orchestration layer | `src/maude/orchestration/engine.py` | Owns workflow control flow across models, tools, memory, verification, and policy | Wrap existing `execution.py`, `agent_executor.py`, and `maude_core/execute.py` behind one orchestration API |
| Provider abstraction | `src/maude/models/providers/base.py` | Defines the common provider interface | Create `ModelProvider` protocol with `complete`, `stream`, `supports_tools`, `supports_vision`, and `health` |
| Model routing | `src/maude/models/router.py` and `routing_policy.py` | Chooses provider/model based on capability, privacy, cost, latency, context, and fallback rules | Move alias and provider-choice logic out of gateway/cloud code |
| Queue-based jobs | `src/maude/orchestration/jobs.py` | Represents long-running work as durable jobs | Start with SQLite-backed local queue under `runtime/data/jobs.sqlite` |
| Stateless workers | `src/maude/orchestration/workers.py` | Pull jobs, execute steps, write state, and exit/retry safely | Worker receives job ID, loads state, executes, writes result; no hidden process-local state required |
| Caching | `src/maude/orchestration/cache.py` | Caches expensive/repeatable tool and model-adjacent calls | Move existing TTL cache behind typed cache API with cache key metadata |
| Rate limiting | `src/maude/orchestration/rate_limits.py` and `src/maude/policy/permissions.py` | Controls provider/tool usage per client/user/tool | Replace global counters with scoped counters keyed by user/client/provider/tool |
| Retries and backoff | `src/maude/orchestration/retries.py` | Handles transient failures without infinite loops | Add retry policy: max attempts, exponential backoff, retryable errors, dead-letter state |
| Prompt/model versioning | `src/maude/models/prompt_versions.py` and `model_versions.py` | Records exact prompt template, model alias, provider model, and parameters | Add prompt/model metadata to every trace event and eval result |
| Observability | `src/maude/observability/` | Emits traces, events, metrics, logs, run summaries | Start with JSONL events and SQLite run index under `runtime/logs/` |
| Eval harness | `src/maude/evaluation/` and `evals/` | Measures behavior across golden tasks and regressions | Start with JSONL golden tasks for tool use, gateway routing, file ops, and model routing |

## Concrete Runtime Control Flow

```text
Client request
  -> gateway validates request and creates run_id
  -> policy checks user/client/tool/model permissions
  -> orchestration engine creates workflow state
  -> model router selects provider/model from routing policy
  -> prompt_versions records prompt/template/model metadata
  -> model call emits observability events
  -> tool request passes policy and risk classification
  -> tool executes through registry/domain handler
  -> verification checks side effects and artifacts
  -> orchestration decides continue/retry/repair/escalate/stop
  -> cache stores eligible deterministic results
  -> final response includes trace/evidence summary
  -> eval harness can replay the same workflow as a golden task
```

## Concrete Background Job Flow

```text
Long-running request
  -> gateway returns accepted job_id
  -> orchestration writes job record to runtime/data/jobs.sqlite
  -> worker claims pending job
  -> worker loads workflow state by job_id
  -> worker executes one bounded step
  -> worker writes step result, trace events, verification result
  -> retry policy decides complete/retry/backoff/dead-letter
  -> client polls or subscribes to job status/events
```


## MAUDE Layer Mapping

| Enterprise Layer | Current MAUDE Evidence | Refactor Direction |
| --- | --- | --- |
| Client surfaces | terminal TUI, Python CLI, phone PWA, Telegram | Move clients under `clients/`; keep channel adapters under `src/maude/channels/` |
| API gateway | `gateway/`, SSE, shared files, HTTPS routes | Move to `src/maude/gateway/`; separate auth/governance/model routing |
| Orchestration | agent executor, plans, scheduler, subagents | Move to `src/maude/orchestration/`; formalize workflow contracts |
| Model gateway | providers, routing, frontier, local/cloud models | Move to `src/maude/models/`; provider adapters behind common contracts |
| Tool layer | registry, catalog, `maude_core/tools_*` | Move to `src/maude/tools/`; split mega-files by domain |
| Memory/state | memory, ledger, chat sync, data folders | Move to `src/maude/memory/`; centralize runtime path config |
| Integrations | Google, GitHub, Substack, browser, voice, VNC | Move to `src/maude/integrations/`; keep adapters isolated |
| Evaluation | tests for gateway/tools/memory/plans | Split tests into unit/integration/smoke/evals |
| Observability/ops | systemd scripts, logs, health, setup scripts | Move to `deploy/` and `docs/operations/`; add service map |
| Packaging | root modules plus packages | Move toward `src/maude` with compatibility wrappers |

## First Refactor Milestones

### Milestone 1: Make The Repo Stop Looking Like Runtime Storage

Deliverables:

- Add missing ignore rules for generated runtime artifacts.
- Add `runtime/` placeholder structure or document external runtime paths.
- Move no code yet.
- Document which large directories are runtime-only.

Success criteria:

- `git status --short` stops showing generated artifacts such as HyperFrames data, mission state, and mobile cert exports.
- README or operations docs explain runtime directories.

### Milestone 2: Introduce Enterprise Package Skeleton

Deliverables:

- Add `src/maude/` skeleton.
- Add package namespaces for `platform`, `gateway`, `orchestration`, `models`, `tools`, `memory`, `integrations`, `channels`, `services`, and `cli`.
- Add compatibility imports so existing commands keep working.

Success criteria:

- No behavior changes.
- Tests still pass or baseline failures are unchanged.
- New code has an obvious home.

### Milestone 3: Move Platform Core

Deliverables:

- Move config, paths, logging, health, cache, and rate-limit primitives into `src/maude/platform/` and `src/maude/orchestration/` as appropriate.
- Centralize runtime path config.

Success criteria:

- Runtime path decisions are no longer scattered.
- Tests cover path resolution and default directories.

### Milestone 4: Move Gateway Boundary

Deliverables:

- Move `gateway/` into `src/maude/gateway/`.
- Keep `python -m gateway` compatibility wrapper.
- Extract provider routing calls behind `src/maude/models/` contracts.

Success criteria:

- Gateway tests pass.
- Gateway request handlers are thinner.

### Milestone 5: Split Tool Platform

Deliverables:

- Move registry/catalog into `src/maude/tools/`.
- Split `tool_defs.py` into domain catalogs.
- Add tool schema and handler validation tests.

Success criteria:

- No single tool definition file owns the whole platform.
- Every registered tool has schema, handler, risk class, and test coverage.

### Milestone 6: Separate Clients And Deploy

Deliverables:

- Move `maude-client/` to `clients/python-cli/`.
- Move `maude-phone/` to `clients/mobile-pwa/`.
- Move scripts/services to `deploy/`.
- Decide whether `maude-phone/dist/` remains tracked.

Success criteria:

- Product surfaces are visibly separate from server internals.
- Deploy scripts and services are discoverable.

## Strong Positioning For The Refactor

> MAUDE is already technically closer to an agent platform than a chatbot. The refactor is about making the architecture visible: clients at the edge, a governed gateway, orchestration in the middle, provider-abstracted model routing, permissioned tools, scoped memory, integration adapters, and operations controls. Once those boundaries are reflected in the repo, the project stops reading as a pile of scripts and starts reading as enterprise AI platform infrastructure.

## Immediate Next Step

Start with the lowest-risk PR:

1. Keep this architecture document.
2. Add ignore rules for current runtime artifacts.
3. Add `docs/operations/service-map.md` documenting running services and ports.
4. Do not move code until the target architecture is accepted.
