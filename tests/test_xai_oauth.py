import frontier
import providers
from maude_core import execute_tool
from maude_core import tools_xai_oauth as xai


def test_finish_accepts_manual_callback_url(tmp_path, monkeypatch):
    auth_path = tmp_path / "xai_oauth.json"
    pending_path = tmp_path / "xai_oauth_pending.json"
    callback_path = tmp_path / "xai_oauth_callback.json"
    monkeypatch.setattr(xai, "AUTH_PATH", auth_path)
    monkeypatch.setattr(xai, "PENDING_PATH", pending_path)
    monkeypatch.setattr(xai, "CALLBACK_PATH", callback_path)
    monkeypatch.setattr(xai, "_pending_result", {})

    pending = {
        "created_at": "2026-05-21T00:00:00Z",
        "redirect_uri": "http://127.0.0.1:56121/callback",
        "code_verifier": "verifier",
        "code_challenge": "challenge",
        "state": "state123",
        "nonce": "nonce123",
        "discovery": {"token_endpoint": "https://auth.x.ai/oauth/token"},
    }
    xai._json_save(pending_path, pending)

    def fake_exchange(token_endpoint, *, code, redirect_uri, verifier, challenge, timeout_seconds=30.0):
        assert token_endpoint == "https://auth.x.ai/oauth/token"
        assert code == "code123"
        assert redirect_uri == "http://127.0.0.1:56121/callback"
        assert verifier == "verifier"
        assert challenge == "challenge"
        return {"access_token": "access", "refresh_token": "refresh", "token_type": "Bearer"}

    monkeypatch.setattr(xai, "_exchange_code", fake_exchange)

    result = execute_tool(
        "xai_oauth_finish",
        {"callback_url": "http://127.0.0.1:56121/callback?code=code123&state=state123"},
    )

    assert "xAI OAuth connected" in result
    assert auth_path.exists()
    assert not pending_path.exists()
    assert not callback_path.exists()


def test_finish_accepts_fragment_callback_url(tmp_path, monkeypatch):
    auth_path = tmp_path / "xai_oauth.json"
    pending_path = tmp_path / "xai_oauth_pending.json"
    callback_path = tmp_path / "xai_oauth_callback.json"
    monkeypatch.setattr(xai, "AUTH_PATH", auth_path)
    monkeypatch.setattr(xai, "PENDING_PATH", pending_path)
    monkeypatch.setattr(xai, "CALLBACK_PATH", callback_path)
    monkeypatch.setattr(xai, "_pending_result", {})

    pending = {
        "created_at": "2026-05-21T00:00:00Z",
        "redirect_uri": "http://127.0.0.1:56121/callback",
        "code_verifier": "verifier",
        "code_challenge": "challenge",
        "state": "state123",
        "nonce": "nonce123",
        "discovery": {"token_endpoint": "https://auth.x.ai/oauth/token"},
    }
    xai._json_save(pending_path, pending)

    def fake_exchange(token_endpoint, *, code, redirect_uri, verifier, challenge, timeout_seconds=30.0):
        assert code == "code123"
        return {"access_token": "access", "refresh_token": "refresh", "token_type": "Bearer"}

    monkeypatch.setattr(xai, "_exchange_code", fake_exchange)

    result = execute_tool(
        "xai_oauth_finish",
        {"callback_url": "http://127.0.0.1:56121/callback#code=code123&state=state123"},
    )

    assert "xAI OAuth connected" in result
    assert auth_path.exists()


def test_access_token_refreshes_using_stored_expiry(tmp_path, monkeypatch):
    auth_path = tmp_path / "xai_oauth.json"
    monkeypatch.setattr(xai, "AUTH_PATH", auth_path)
    xai._json_save(
        auth_path,
        {
            "tokens": {
                "access_token": "old-access",
                "refresh_token": "refresh",
                "expires_at": 1,
            },
            "discovery": {"token_endpoint": "https://auth.x.ai/oauth/token"},
        },
    )

    def fake_refresh(tokens, discovery):
        assert tokens["access_token"] == "old-access"
        assert discovery["token_endpoint"] == "https://auth.x.ai/oauth/token"
        return {
            "access_token": "new-access",
            "refresh_token": "refresh",
            "expires_in": 3600,
        }

    monkeypatch.setattr(xai, "_refresh", fake_refresh)

    assert xai.get_oauth_access_token() == "new-access"
    saved = xai._json_load(auth_path)
    assert saved["tokens"]["access_token"] == "new-access"
    assert saved["last_refresh"]


def test_grok_oauth_provider_available_when_token_exists(tmp_path, monkeypatch):
    auth_path = tmp_path / "xai_oauth.json"
    xai._json_save(auth_path, {"tokens": {"access_token": "access", "refresh_token": "refresh"}})
    monkeypatch.setattr(xai, "AUTH_PATH", auth_path)
    monkeypatch.setattr(providers.Path, "home", lambda: tmp_path)

    config_dir = tmp_path / ".config" / "maude"
    config_dir.mkdir(parents=True)
    (config_dir / "xai_oauth.json").write_text('{"tokens":{"access_token":"access"}}')

    assert providers.get_api_key("grok-oauth") == "oauth"
    assert "grok-oauth" in frontier.list_available_providers()
