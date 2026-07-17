import importlib

import pytest

from maude.config import runtime_paths
from maude.evals import EvalCase, EvalHarness, EvalResult
from maude.gateway import CapabilityPolicy, RequestContext
from maude.integrations import IntegrationAdapter, TestDoubleAdapter
from maude.memory import MemoryProvenance, MemoryScope, ScopedMemoryStore
from maude.observability import EventSink, ObservabilityEvent, RunContext
from maude.orchestration import (
    FileJobStore,
    Job,
    JobPriority,
    JobQueue,
    JobStatus,
    OrchestrationEngine,
    RetryPolicy,
    run_with_retries,
    tool_execution,
)
from maude.orchestration.cache import CacheStore
from maude.orchestration.engine import WorkflowRequest
from maude.orchestration.rate_limits import SlidingWindowRateLimiter
from maude.orchestration.worker import main as worker_main
from maude.orchestration.workers import StatelessWorker
from maude.prompts import PromptSpec, PromptVersionRegistry
from maude.providers import (
    ModelCallMetadata,
    ModelRequest,
    ModelRouter,
    ModelRoutingPolicy,
    ProviderCapability,
    ProviderRegistry,
)
from maude.providers.config import Provider, ProviderConfig
from maude.providers.frontier import _build_frontier_response
from maude.providers.legacy import load_legacy_provider_capabilities
from maude.tools import ToolPlatform, ToolRequest, ToolRisk, classify_tool
from maude.tools.domains import all_schemas as all_domain_tool_schemas
from maude.tools.domains import filesystem, memory, missions, web
from maude.verification import GateResult, VerificationSuite
from maude_core.tool_defs import TOOLS as LEGACY_TOOL_SCHEMAS


class _ProviderStub:
    def __init__(self, capability):
        self.capability = capability


def test_model_router_selects_eligible_provider_by_requirements():
    registry = ProviderRegistry()
    registry.register(
        _ProviderStub(
            ProviderCapability(
                name="cheap-text",
                provider="local",
                model="local-text",
                supports_tools=False,
                supports_vision=False,
                local=True,
            )
        )
    )
    registry.register(
        _ProviderStub(
            ProviderCapability(
                name="tool-model",
                provider="openai",
                model="tool-capable",
                supports_tools=True,
                supports_vision=True,
                cost_per_1k_input=0.1,
            )
        )
    )

    route = ModelRouter(registry).route(ModelRequest(messages=[], requires_tools=True))

    assert route is not None
    assert route.provider_name == "tool-model"
    assert route.reason == "selected lowest declared cost eligible model"


def test_queue_based_jobs_execute_through_stateless_worker(tmp_path):
    worker = StatelessWorker({"echo": lambda payload: payload["message"]})
    queue = JobQueue(FileJobStore(tmp_path))
    engine = OrchestrationEngine(worker, queue)

    queued = engine.submit(WorkflowRequest(kind="echo", payload={"message": "ok"}, background=True))
    completed = engine.run_next()

    assert queued.status == JobStatus.SUCCEEDED
    assert completed is queued
    assert completed.result == "ok"
    assert queue.list() == []


def test_retry_policy_bounds_attempts():
    attempts = {"count": 0}

    def flaky(_payload):
        attempts["count"] += 1
        raise RuntimeError("temporary")

    worker = StatelessWorker({"flaky": flaky}, retry_policy=RetryPolicy(max_attempts=2))
    engine = OrchestrationEngine(worker)

    completed = engine.submit(WorkflowRequest(kind="flaky", payload={}))

    assert completed.status == JobStatus.FAILED
    assert completed.attempts == 2
    assert attempts["count"] == 2


def test_cache_rate_limit_prompt_observability_eval_and_verification_contracts():
    cache = CacheStore()
    cache.put("provider:model:prompt", "cached", ttl_seconds=60)
    assert cache.get("provider:model:prompt") == "cached"

    limiter = SlidingWindowRateLimiter(limit=1, window_seconds=60)
    assert limiter.check("client-a").allowed is True
    assert limiter.check("client-a").allowed is False

    prompts = PromptVersionRegistry()
    prompts.register(PromptSpec(name="planner", version="001", template="Plan: {task}", model="tool-model"))
    assert prompts.latest("planner").model == "tool-model"

    context = RunContext(user_id="user-1", client="test")
    sink = EventSink()
    sink.emit(ObservabilityEvent(name="tool.started", context=context, payload={"tool": "save_memory"}))
    assert sink.for_run(context.run_id)[0].payload["tool"] == "save_memory"

    harness = EvalHarness(
        [EvalCase(name="golden", input={"prompt": "hello"})],
        lambda case: EvalResult(name=case.name, passed=True, score=1.0, details="matched"),
    )
    assert harness.summary() == {"total": 1, "passed": 1, "failed": 0, "score": 1.0}

    suite = VerificationSuite([lambda: GateResult(name="unit", passed=True, details="ok")])
    assert suite.passed() is True


def test_gateway_governance_scope_policy():
    policy = CapabilityPolicy()

    assert policy.allows(RequestContext(user_id="u", client="cli", scopes={"tool:memory"}), "tool:memory")
    assert not policy.allows(RequestContext(user_id="u", client="cli", scopes={"tool:file"}), "tool:memory")


def test_legacy_provider_config_exposes_provider_capabilities():
    capabilities = load_legacy_provider_capabilities()

    assert capabilities
    assert any(capability.provider == "openai" for capability in capabilities)


def test_durable_job_queue_reloads_queued_jobs(tmp_path):
    store = FileJobStore(tmp_path)
    queue = JobQueue(store)
    job = queue.enqueue(Job(kind="render", payload={"asset": "clip"}, priority=JobPriority.HIGH))

    reloaded = JobQueue(store)

    assert reloaded.get(job.job_id) is not None
    assert reloaded.dequeue().job_id == job.job_id


def test_failed_background_jobs_move_to_dead_letter(tmp_path):
    def failing(_payload):
        raise RuntimeError("boom")

    queue = JobQueue(FileJobStore(tmp_path))
    worker = StatelessWorker({"fail": failing}, retry_policy=RetryPolicy(max_attempts=1))
    engine = OrchestrationEngine(worker, queue)

    queued = engine.submit(WorkflowRequest(kind="fail", payload={}, background=True))
    completed = engine.run_next()

    assert completed.job_id == queued.job_id
    assert completed.status == JobStatus.FAILED
    assert queue.list() == []
    assert queue.dead_letters()[0].job_id == queued.job_id
    assert (tmp_path / "dead_letter.json").exists()


def test_runtime_paths_are_configurable_without_hard_coded_storage(monkeypatch, tmp_path):
    runtime_root = tmp_path / "runtime-root"
    config_dir = tmp_path / "config"
    shared_dir = tmp_path / "shared-override"

    monkeypatch.setenv("MAUDE_RUNTIME_ROOT", str(runtime_root))
    monkeypatch.setenv("MAUDE_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("MAUDE_SHARED_DIR", str(shared_dir))

    paths = runtime_paths()

    assert paths.shared_dir == shared_dir
    assert paths.transfers_dir == runtime_root / "runtime" / "transfers"
    assert paths.conversations_dir == runtime_root / "runtime" / "data" / "conversations"
    assert paths.jobs_dir == config_dir / "jobs"
    assert paths.schedules_file == config_dir / "schedules.json"


def test_default_file_job_store_uses_configured_runtime_path(monkeypatch, tmp_path):
    jobs_dir = tmp_path / "jobs"
    monkeypatch.setenv("MAUDE_JOBS_DIR", str(jobs_dir))

    store = FileJobStore()
    store.save_jobs([Job(kind="probe", payload={})])

    assert store.root == jobs_dir
    assert (jobs_dir / "jobs.json").exists()


def test_legacy_gateway_modules_are_compatibility_entrypoints():
    module_symbols = {
        "gateway": "GatewayHandler",
        "gateway.main": "main",
        "gateway.state": "get_model_route",
        "gateway.server": "GatewayHandler",
        "gateway.routes": "RoutesMixin",
        "gateway.cloud": "CloudMixin",
        "gateway.replicate": "stylize_image",
        "gateway.websocket": "ws_accept_key",
    }

    for legacy_name, symbol in module_symbols.items():
        legacy_module = importlib.import_module(legacy_name)
        canonical_name = "maude.gateway" if legacy_name == "gateway" else f"maude.{legacy_name}"
        canonical_module = importlib.import_module(canonical_name)

        assert getattr(legacy_module, symbol) is getattr(canonical_module, symbol)


def test_legacy_orchestration_modules_are_compatibility_entrypoints():
    module_symbols = {
        "agent_executor": ("maude.orchestration.agents", "AgentTask"),
        "auto_router": ("maude.orchestration.intent_routing", "route_message"),
        "execution": ("maude.orchestration.execution", "execute_subagent"),
        "scheduler": ("maude.orchestration.scheduler", "ProactiveScheduler"),
        "subagents": ("maude.orchestration.subagents", "SUBAGENTS"),
        "maude_core.execute": ("maude.orchestration.tool_execution", "execute_tool"),
    }

    for legacy_name, (canonical_name, symbol) in module_symbols.items():
        legacy_module = importlib.import_module(legacy_name)
        canonical_module = importlib.import_module(canonical_name)

        assert getattr(legacy_module, symbol) is getattr(canonical_module, symbol)


def test_stateless_worker_process_entrypoint_drains_one_queued_job(tmp_path, capsys):
    queue = JobQueue(FileJobStore(tmp_path))
    job = queue.enqueue(Job(kind="echo", payload={"message": "ok"}))

    exit_code = worker_main(["--once", "--jobs-dir", str(tmp_path)])
    output = capsys.readouterr().out
    reloaded = JobQueue(FileJobStore(tmp_path))

    assert exit_code == 0
    assert f'"job_id": "{job.job_id}"' in output
    assert '"status": "succeeded"' in output
    assert reloaded.list() == []


def test_retry_runner_applies_bounded_backoff_without_exceeding_attempts():
    attempts = {"count": 0}
    sleeps = []

    def flaky():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("temporary")
        return "ok"

    result = run_with_retries(
        flaky,
        RetryPolicy(max_attempts=3, initial_delay_seconds=0.1, multiplier=2.0),
        sleep=sleeps.append,
    )

    assert result == "ok"
    assert attempts["count"] == 3
    assert sleeps == [0.1, 0.2]


def test_tool_execution_accepts_explicit_retry_policy(monkeypatch):
    attempts = {"count": 0}

    def flaky_handler(_args):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("temporary")
        return "ok"

    monkeypatch.setattr(tool_execution, "get_handler", lambda _name: flaky_handler)
    monkeypatch.setattr(tool_execution, "is_cacheable", lambda _name: False)

    result = tool_execution.execute_tool(
        "retry_probe",
        {},
        retry_policy=RetryPolicy(max_attempts=2, initial_delay_seconds=0.0),
    )

    assert result == "ok"
    assert attempts["count"] == 2


def test_retry_runner_reraises_after_attempt_budget():
    attempts = {"count": 0}

    def always_fails():
        attempts["count"] += 1
        raise RuntimeError("still failing")

    with pytest.raises(RuntimeError, match="still failing"):
        run_with_retries(
            always_fails, RetryPolicy(max_attempts=2, initial_delay_seconds=0.0), sleep=lambda _delay: None
        )

    assert attempts["count"] == 2


def test_legacy_provider_modules_are_compatibility_entrypoints():
    module_symbols = {
        "providers": ("maude.providers.config", "PROVIDERS"),
        "routing": ("maude.providers.capability_routing", "CapabilityRouter"),
        "frontier": ("maude.providers.frontier", "ask_frontier"),
    }

    for legacy_name, (canonical_name, symbol) in module_symbols.items():
        legacy_module = importlib.import_module(legacy_name)
        canonical_module = importlib.import_module(canonical_name)

        assert getattr(legacy_module, symbol) is getattr(canonical_module, symbol)


def test_model_router_applies_alias_health_cost_latency_and_privacy_policy():
    registry = ProviderRegistry()
    registry.register(
        _ProviderStub(
            ProviderCapability(
                name="remote-fast",
                provider="openai",
                model="frontier-model",
                supports_tools=True,
                healthy=True,
                latency_ms=100,
                cost_per_1k_input=0.01,
                cost_per_1k_output=0.01,
            )
        )
    )
    registry.register(
        _ProviderStub(
            ProviderCapability(
                name="local-private",
                provider="local",
                model="private-model",
                supports_tools=True,
                local=True,
                private=True,
                healthy=True,
                latency_ms=20,
            )
        )
    )
    registry.register(
        _ProviderStub(
            ProviderCapability(
                name="unhealthy-cheap",
                provider="local",
                model="cheap-model",
                local=True,
                private=True,
                healthy=False,
            )
        )
    )

    policy = ModelRoutingPolicy(
        aliases={"fast": "frontier-model"},
        max_cost_per_1k=0.05,
        max_latency_ms=150,
        fallback_enabled=True,
    )
    route = ModelRouter(registry, policy).route(ModelRequest(messages=[], model="fast", requires_tools=True))

    assert route is not None
    assert route.provider_name == "remote-fast"
    assert route.model == "frontier-model"
    assert route.fallback_provider_names == ["local-private"]

    private_route = ModelRouter(registry, ModelRoutingPolicy(require_private=True)).route(
        ModelRequest(messages=[], requires_tools=True)
    )
    assert private_route.provider_name == "local-private"

    no_route = ModelRouter(registry, ModelRoutingPolicy(max_latency_ms=10)).route(ModelRequest(messages=[]))
    assert no_route is None


def test_frontier_response_records_model_call_metadata():
    config = ProviderConfig(
        name="Test Provider",
        provider=Provider.OPENAI,
        api_key_env="TEST_API_KEY",
        base_url="https://example.test/v1",
        default_model="test-model-1",
        supports_vision=False,
        supports_tools=True,
        cost_per_1k_input=0.2,
        cost_per_1k_output=0.4,
    )

    response = _build_frontier_response(
        content="ok",
        config=config,
        input_tokens=100,
        output_tokens=50,
        latency_seconds=1.23456,
        prompt_version="planner:001",
        routing_decision="selected provider: test",
    )

    assert isinstance(ModelCallMetadata(**response.metadata), ModelCallMetadata)
    assert response.metadata["provider"] == "Test Provider"
    assert response.metadata["model_version"] == "test-model-1"
    assert response.metadata["prompt_version"] == "planner:001"
    assert response.metadata["routing_decision"] == "selected provider: test"
    assert response.metadata["input_tokens"] == 100
    assert response.metadata["output_tokens"] == 50
    assert response.metadata["latency_seconds"] == 1.235
    assert response.metadata["cost_usd"] == 0.04


def test_tool_schemas_are_exposed_through_domain_catalogs():
    def names(schemas):
        return {tool["function"]["name"] for tool in schemas}

    assert {"read_file", "write_file", "run_command"}.issubset(names(filesystem.schemas()))
    assert {"web_search", "web_browse"}.issubset(names(web.schemas()))
    assert {"save_memory", "recall_memory"}.issubset(names(memory.schemas()))
    assert {"mission_create", "mission_tick"}.issubset(names(missions.schemas()))
    assert len(all_domain_tool_schemas()) >= 30


def test_tool_policy_classifies_risk_and_blocks_unapproved_side_effects(monkeypatch):
    assert classify_tool("read_file") == ToolRisk.READ
    assert classify_tool("drive_delete") == ToolRisk.DELETE
    assert classify_tool("youtube_upload") == ToolRisk.PUBLISH

    platform = ToolPlatform()
    blocked = platform.execute(ToolRequest(name="drive_delete", arguments={"file_id": "abc"}))

    assert "approval required" in blocked.output
    assert blocked.metadata["blocked"] is True
    assert blocked.metadata["risk"] == "delete"

    monkeypatch.setattr("maude_core.execute.execute_tool", lambda name, args: "deleted")
    approved = platform.execute(
        ToolRequest(name="drive_delete", arguments={"file_id": "abc"}, approvals={"delete"}, run_id="run-1")
    )

    assert approved.output == "deleted"
    assert approved.metadata == {"tool": "drive_delete", "risk": "delete", "run_id": "run-1"}


def test_tool_policy_validates_arguments_before_execution():
    result = ToolPlatform().execute(ToolRequest(name="read_file", arguments={}))

    assert "requires a path argument" in result.output
    assert result.metadata["blocked"] is True


def test_legacy_memory_ledger_is_compatibility_entrypoint():
    legacy = importlib.import_module("maude_core.memory_ledger")
    canonical = importlib.import_module("maude.memory.ledger")

    assert legacy.MemoryLedger is canonical.MemoryLedger
    assert legacy.get_ledger is canonical.get_ledger


def test_domain_tool_schemas_are_canonical_for_legacy_tool_defs():
    domain_names = {tool["function"]["name"] for tool in all_domain_tool_schemas()}
    legacy_names = {tool["function"]["name"] for tool in LEGACY_TOOL_SCHEMAS}

    assert len(domain_names) == 198
    assert legacy_names == domain_names


def test_integration_adapter_retries_transient_errors():
    adapter = IntegrationAdapter(RetryPolicy(max_attempts=2, initial_delay_seconds=0.0))
    attempts = {"count": 0}

    def transient_operation():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise TimeoutError("connection timeout")
        return "ok"

    assert adapter.run_with_retries(transient_operation) == "ok"
    assert attempts["count"] == 2


def test_scoped_memory_store_filters_by_user_project_and_workspace(tmp_path):
    scope_a = MemoryScope(user_id="u1", project_id="p1", workspace_id="w1")
    scope_b = MemoryScope(user_id="u2", project_id="p1", workspace_id="w1")
    store_a = ScopedMemoryStore.open(tmp_path, scope_a)
    store_b = ScopedMemoryStore.open(tmp_path, scope_b)

    store_a.save("brief", "Project alpha private context", "project")
    store_b.save("brief", "Project beta private context", "project")

    assert [record.value for record in store_a.search("private context", category="project")] == [
        "Project alpha private context"
    ]
    assert [record.value for record in store_b.search("private context", category="project")] == [
        "Project beta private context"
    ]


def test_memory_retention_export_provenance_and_delete_controls(tmp_path):
    scope = MemoryScope(user_id="u1", project_id="p1", workspace_id="w1")
    store = ScopedMemoryStore.open(tmp_path, scope)
    store.save("decision", "Use scoped memory.", "project", provenance=MemoryProvenance(source="test", run_id="run-1"))

    exported = store.export(category="project").to_dict()

    assert exported["scope"] == "user=u1|project=p1|workspace=w1"
    assert exported["records"][0]["category"] == "project"
    assert "source=test" in exported["records"][0]["value"]
    assert store.delete("decision") is True
    assert store.search("scoped memory", category="project") == []


def test_integration_test_double_and_audit_contracts():
    adapter = TestDoubleAdapter({"list": ["ok"]})

    assert adapter.call("list", limit=1) == ["ok"]
    assert adapter.calls == [("list", {"limit": 1})]
    assert adapter.audit_event("list", run_id="run-1") == {
        "integration": "integration",
        "action": "list",
        "run_id": "run-1",
    }


def test_runtime_paths_cover_stateful_runtime_locations(monkeypatch, tmp_path):
    monkeypatch.setenv("MAUDE_RUNTIME_ROOT", str(tmp_path / "runtime"))
    paths = runtime_paths()

    assert paths.data_dir == tmp_path / "runtime" / "runtime" / "data"
    assert paths.shared_dir == tmp_path / "runtime" / "runtime" / "shared"
    assert paths.transfers_dir == tmp_path / "runtime" / "runtime" / "transfers"
    assert paths.logs_dir == tmp_path / "runtime" / "runtime" / "logs"
    assert paths.certs_dir == tmp_path / "runtime" / "runtime" / "certs"
    assert paths.cache_dir == tmp_path / "runtime" / "runtime" / "cache"
    assert paths.generated_media_dir == paths.data_dir / "generated_media"
    assert paths.comfyui_output_dir.name == "output"
    assert paths.hyperframes_dir == paths.data_dir / "hyperframes"
    assert paths.browser_data_dir == paths.data_dir / "browser_data"
    assert paths.screenshots_dir == paths.data_dir / "screenshots"
