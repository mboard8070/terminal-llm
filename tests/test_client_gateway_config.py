import importlib
import sys
from pathlib import Path


CLIENT_PATH = Path(__file__).resolve().parents[1] / "maude-client"
if str(CLIENT_PATH) not in sys.path:
    sys.path.insert(0, str(CLIENT_PATH))


def _reload_config(monkeypatch, gateway_url=None, llm_url=None):
    monkeypatch.delenv("MAUDE_GATEWAY_HOST", raising=False)
    monkeypatch.delenv("MAUDE_GATEWAY_PORT", raising=False)
    monkeypatch.delenv("MAUDE_GATEWAY_BASE_URL", raising=False)
    monkeypatch.delenv("MAUDE_FILE_SERVER_URL", raising=False)
    monkeypatch.delenv("MAUDE_SERVER_SSH_HOST", raising=False)
    monkeypatch.delenv("SERVER_SSH_HOST", raising=False)
    if gateway_url is None:
        monkeypatch.delenv("MAUDE_GATEWAY_URL", raising=False)
    else:
        monkeypatch.setenv("MAUDE_GATEWAY_URL", gateway_url)
    if llm_url is None:
        monkeypatch.delenv("LLM_SERVER_URL", raising=False)
    else:
        monkeypatch.setenv("LLM_SERVER_URL", llm_url)

    import maude_client.config as config

    return importlib.reload(config)


def test_client_config_defaults_to_local_gateway(monkeypatch):
    config = _reload_config(monkeypatch)

    assert config.GATEWAY_URL == "http://127.0.0.1:8080/v1"
    assert config.GATEWAY_BASE_URL == "http://127.0.0.1:8080"
    assert config.SERVER_HOST == "127.0.0.1"
    assert config.SERVER_LLM_PORT == 8080
    assert config.FILE_SERVER_URL == "http://127.0.0.1:8080"


def test_client_config_normalizes_base_gateway_url(monkeypatch):
    config = _reload_config(monkeypatch, gateway_url="http://maude-host:9090")

    assert config.GATEWAY_URL == "http://maude-host:9090/v1"
    assert config.GATEWAY_BASE_URL == "http://maude-host:9090"
    assert config.SERVER_HOST == "maude-host"
    assert config.SERVER_LLM_PORT == 9090


def test_client_config_keeps_llm_server_url_alias(monkeypatch):
    config = _reload_config(monkeypatch, llm_url="https://legacy-host:30000/v1")

    assert config.GATEWAY_URL == "https://legacy-host:30000/v1"
    assert config.GATEWAY_BASE_URL == "https://legacy-host:30000"
    assert config.SERVER_HOST == "legacy-host"
    assert config.SERVER_LLM_PORT == 30000
