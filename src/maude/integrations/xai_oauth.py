"""xAI Grok OAuth bridge for MAUDE.

Implements the same OAuth 2.0 PKCE loopback flow used by Hermes/OpenClaw for
SuperGrok or X Premium+ accounts, then reuses the bearer token against xAI's
OpenAI-compatible API.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import threading
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from maude.config import runtime_paths
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from tool_registry import register_tool

XAI_OAUTH_ISSUER = "https://auth.x.ai"
XAI_OAUTH_DISCOVERY_URL = f"{XAI_OAUTH_ISSUER}/.well-known/openid-configuration"
XAI_OAUTH_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
XAI_OAUTH_SCOPE = "openid profile email offline_access grok-cli:access api:access"
XAI_OAUTH_REDIRECT_HOST = "127.0.0.1"
XAI_OAUTH_REDIRECT_PORT = 56121
XAI_OAUTH_REDIRECT_PATH = "/callback"
XAI_API_BASE_URL = "https://api.x.ai/v1"
DEFAULT_MODEL = "grok-4.3"

_PATHS = runtime_paths()
AUTH_PATH = _PATHS.xai_oauth_file
PENDING_PATH = _PATHS.xai_oauth_pending_file
CALLBACK_PATH = _PATHS.xai_oauth_callback_file

_pending_server: HTTPServer | None = None
_pending_thread: threading.Thread | None = None
_pending_result: dict[str, Any] = {}


class XaiOAuthError(RuntimeError):
    pass


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _code_verifier() -> str:
    return _b64url(secrets.token_bytes(48))


def _code_challenge(verifier: str) -> str:
    return _b64url(hashlib.sha256(verifier.encode("ascii")).digest())


def _json_load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def _json_save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.chmod(0o600)
    tmp.replace(path)


def _json_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _validate_xai_url(url: str, *, field: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or (host != "x.ai" and not host.endswith(".x.ai")):
        raise XaiOAuthError(f"xAI discovery returned invalid {field}: {url!r}")
    return url


def _discovery(timeout_seconds: float = 20.0) -> dict[str, str]:
    resp = httpx.get(XAI_OAUTH_DISCOVERY_URL, headers={"Accept": "application/json"}, timeout=timeout_seconds)
    if resp.status_code != 200:
        raise XaiOAuthError(f"xAI OIDC discovery failed with HTTP {resp.status_code}: {resp.text[:500]}")
    payload = resp.json()
    auth_endpoint = str(payload.get("authorization_endpoint") or "").strip()
    token_endpoint = str(payload.get("token_endpoint") or "").strip()
    if not auth_endpoint or not token_endpoint:
        raise XaiOAuthError("xAI OIDC discovery response did not include authorization/token endpoints.")
    return {
        "authorization_endpoint": _validate_xai_url(auth_endpoint, field="authorization_endpoint"),
        "token_endpoint": _validate_xai_url(token_endpoint, field="token_endpoint"),
    }


def _authorize_url(discovery: dict[str, str], redirect_uri: str, challenge: str, state: str, nonce: str) -> str:
    params = {
        "response_type": "code",
        "client_id": XAI_OAUTH_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": XAI_OAUTH_SCOPE,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "nonce": nonce,
        "plan": "generic",
        "referrer": "maude",
    }
    return f"{discovery['authorization_endpoint']}?{urlencode(params)}"


def _callback_from_args(args: dict[str, Any]) -> dict[str, str]:
    callback_url = str(args.get("callback_url") or args.get("url") or "").strip()
    if callback_url:
        parsed = urlparse(callback_url)
        if parsed.scheme not in {"http", "https"}:
            raise XaiOAuthError("callback_url must be a full http/https URL.")
        callback = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        if parsed.fragment:
            callback.update({k: v[0] for k, v in parse_qs(parsed.fragment).items()})
        return callback

    code = str(args.get("code") or "").strip()
    if not code:
        return {}
    callback = {"code": code}
    state = str(args.get("state") or "").strip()
    if state:
        callback["state"] = state
    return callback


def _exchange_code(
    token_endpoint: str,
    *,
    code: str,
    redirect_uri: str,
    verifier: str,
    challenge: str,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": XAI_OAUTH_CLIENT_ID,
        "code_verifier": verifier,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    resp = httpx.post(
        token_endpoint,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        data=data,
        timeout=max(20.0, timeout_seconds),
    )
    if resp.status_code != 200:
        detail = resp.text.strip()
        if resp.status_code == 403:
            raise XaiOAuthError(
                "xAI token exchange failed with HTTP 403. The OAuth login worked, "
                "but this account/tier is not authorized for xAI API access. "
                f"Response: {detail}"
            )
        raise XaiOAuthError(f"xAI token exchange failed with HTTP {resp.status_code}: {detail}")
    payload = resp.json()
    if not payload.get("access_token") or not payload.get("refresh_token"):
        raise XaiOAuthError("xAI token exchange response was missing access_token or refresh_token.")
    _stamp_token_expiry(payload)
    return payload


def _refresh(tokens: dict[str, Any], discovery: dict[str, str]) -> dict[str, Any]:
    refresh_token = str(tokens.get("refresh_token") or "").strip()
    if not refresh_token:
        raise XaiOAuthError("Stored xAI OAuth credentials are missing refresh_token; run xai_oauth_start again.")
    token_endpoint = discovery.get("token_endpoint") or _discovery()["token_endpoint"]
    _validate_xai_url(token_endpoint, field="token_endpoint")
    resp = httpx.post(
        token_endpoint,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        data={
            "grant_type": "refresh_token",
            "client_id": XAI_OAUTH_CLIENT_ID,
            "refresh_token": refresh_token,
        },
        timeout=30.0,
    )
    if resp.status_code != 200:
        raise XaiOAuthError(f"xAI token refresh failed with HTTP {resp.status_code}: {resp.text.strip()}")
    payload = resp.json()
    if not payload.get("access_token"):
        raise XaiOAuthError("xAI token refresh response was missing access_token.")
    updated = dict(tokens)
    updated["access_token"] = payload["access_token"]
    updated["refresh_token"] = payload.get("refresh_token") or refresh_token
    updated["id_token"] = payload.get("id_token") or updated.get("id_token")
    updated["expires_in"] = payload.get("expires_in")
    updated["token_type"] = payload.get("token_type") or "Bearer"
    _stamp_token_expiry(updated)
    return updated


def _stamp_token_expiry(tokens: dict[str, Any]) -> None:
    """Persist a refresh deadline for opaque tokens as well as JWT tokens."""
    expires_in = tokens.get("expires_in")
    try:
        seconds = int(float(expires_in))
    except (TypeError, ValueError):
        return
    if seconds > 0:
        tokens["expires_at"] = time.time() + seconds


def _access_token_expiring(token: str, skew_seconds: int = 300) -> bool:
    if not token or "." not in token:
        return False
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload.encode("ascii")).decode("utf-8"))
        exp = data.get("exp")
        return isinstance(exp, (int, float)) and float(exp) <= time.time() + skew_seconds
    except Exception:
        return False


def _tokens_expiring(tokens: dict[str, Any], skew_seconds: int = 300) -> bool:
    expires_at = tokens.get("expires_at")
    if isinstance(expires_at, (int, float)):
        return float(expires_at) <= time.time() + skew_seconds
    try:
        if expires_at is not None:
            return float(expires_at) <= time.time() + skew_seconds
    except (TypeError, ValueError):
        pass
    return _access_token_expiring(str(tokens.get("access_token") or ""), skew_seconds=skew_seconds)


def _get_access_token(force_refresh: bool = False) -> tuple[str, dict[str, Any]]:
    state = _json_load(AUTH_PATH)
    tokens = dict(state.get("tokens") or {})
    token = str(tokens.get("access_token") or "").strip()
    if not token:
        raise XaiOAuthError("No xAI OAuth token stored. Run xai_oauth_start, complete login, then xai_oauth_finish.")
    if force_refresh or _tokens_expiring(tokens):
        tokens = _refresh(tokens, dict(state.get("discovery") or {}))
        state["tokens"] = tokens
        state["last_refresh"] = _now_iso()
        _json_save(AUTH_PATH, state)
        token = str(tokens.get("access_token") or "").strip()
    return token, state


def get_oauth_access_token(force_refresh: bool = False) -> str:
    """Return a fresh xAI OAuth bearer token for provider integrations."""
    token, _state = _get_access_token(force_refresh=force_refresh)
    return token


def has_oauth_credentials() -> bool:
    """Return true when MAUDE has stored xAI OAuth credentials."""
    state = _json_load(AUTH_PATH)
    return bool(str((state.get("tokens") or {}).get("access_token") or "").strip())


def _stop_callback_server() -> None:
    global _pending_server, _pending_thread
    if _pending_server is not None:
        try:
            _pending_server.shutdown()
            _pending_server.server_close()
        except Exception:
            pass
    _pending_server = None
    _pending_thread = None


def _start_callback_server() -> str:
    global _pending_server, _pending_thread, _pending_result
    _stop_callback_server()

    _pending_result = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path != XAI_OAUTH_REDIRECT_PATH:
                self.send_response(404)
                self.end_headers()
                return
            params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            _pending_result.clear()
            _pending_result.update(params)
            _json_save(
                CALLBACK_PATH,
                {
                    "captured_at": _now_iso(),
                    "params": params,
                },
            )
            body = b"xAI login captured. You can return to MAUDE."
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            return

    _pending_server = HTTPServer((XAI_OAUTH_REDIRECT_HOST, XAI_OAUTH_REDIRECT_PORT), Handler)
    _pending_thread = threading.Thread(target=_pending_server.serve_forever, daemon=True, name="xai-oauth-callback")
    _pending_thread.start()
    return f"http://{XAI_OAUTH_REDIRECT_HOST}:{XAI_OAUTH_REDIRECT_PORT}{XAI_OAUTH_REDIRECT_PATH}"


@register_tool("xai_oauth_start")
def tool_xai_oauth_start(args: dict) -> str:
    """Begin xAI OAuth and return the authorization URL."""
    del args
    discovery = _discovery()
    redirect_uri = _start_callback_server()
    verifier = _code_verifier()
    challenge = _code_challenge(verifier)
    pending = {
        "created_at": _now_iso(),
        "redirect_uri": redirect_uri,
        "code_verifier": verifier,
        "code_challenge": challenge,
        "state": secrets.token_urlsafe(24),
        "nonce": secrets.token_urlsafe(24),
        "discovery": discovery,
    }
    pending["authorize_url"] = _authorize_url(
        discovery,
        redirect_uri,
        challenge,
        pending["state"],
        pending["nonce"],
    )
    _json_save(PENDING_PATH, pending)
    _json_unlink(CALLBACK_PATH)
    return (
        "xAI OAuth started.\n"
        f"Open this URL and approve access:\n{pending['authorize_url']}\n\n"
        "After the page says the login was captured, run xai_oauth_finish.\n"
        "If the browser cannot reach the local callback, paste the final callback URL into "
        "xai_oauth_finish as callback_url."
    )


@register_tool("xai_oauth_finish")
def tool_xai_oauth_finish(args: dict) -> str:
    """Complete xAI OAuth after the callback has been captured."""
    pending = _json_load(PENDING_PATH)
    if not pending:
        return "No pending xAI OAuth login. Run xai_oauth_start first."
    try:
        callback = _callback_from_args(args)
    except XaiOAuthError as exc:
        return f"xAI authorization failed: {exc}"
    if not callback:
        callback = dict(_pending_result)
    if not callback:
        callback = dict(_json_load(CALLBACK_PATH).get("params") or {})
    if not callback:
        return (
            "No xAI OAuth callback captured yet. Complete the authorization page, then run xai_oauth_finish again. "
            "If the local callback page did not load, pass the final redirected URL as callback_url."
        )
    if callback.get("error"):
        return f"xAI authorization failed: {callback.get('error_description') or callback.get('error')}"
    if callback.get("state") and callback.get("state") != pending.get("state"):
        return "xAI authorization failed: OAuth state mismatch."
    if not callback.get("state"):
        return "xAI authorization failed: callback did not include OAuth state. Pass the full callback_url instead of only code."
    code = str(callback.get("code") or "").strip()
    if not code:
        return "xAI authorization failed: callback did not include an authorization code."

    tokens = _exchange_code(
        pending["discovery"]["token_endpoint"],
        code=code,
        redirect_uri=pending["redirect_uri"],
        verifier=pending["code_verifier"],
        challenge=pending["code_challenge"],
    )
    state = {
        "provider": "xai-oauth",
        "auth_mode": "oauth_pkce",
        "base_url": XAI_API_BASE_URL,
        "model": DEFAULT_MODEL,
        "tokens": tokens,
        "discovery": pending.get("discovery") or {},
        "redirect_uri": pending.get("redirect_uri"),
        "last_refresh": _now_iso(),
    }
    _json_save(AUTH_PATH, state)
    _json_unlink(PENDING_PATH)
    _json_unlink(CALLBACK_PATH)
    _stop_callback_server()
    return f"xAI OAuth connected. Token saved at {AUTH_PATH}. Default model: {DEFAULT_MODEL}."


@register_tool("xai_oauth_doctor")
def tool_xai_oauth_doctor(args: dict) -> str:
    """Run non-secret diagnostics for xAI OAuth readiness."""
    force_refresh = bool(args.get("refresh"))
    test_api = bool(args.get("test_api"))
    lines = ["xAI OAuth doctor:"]

    pending = _json_load(PENDING_PATH)
    callback = _json_load(CALLBACK_PATH)
    lines.append(f"- pending login: {'yes' if pending else 'no'}")
    if pending:
        lines.append(f"- pending created_at: {pending.get('created_at', 'unknown')}")
        lines.append(f"- callback captured: {'yes' if callback else 'no'}")

    try:
        discovery = _discovery(timeout_seconds=10.0)
        lines.append("- discovery: ok")
        lines.append(f"- authorization endpoint: {discovery['authorization_endpoint']}")
        lines.append(f"- token endpoint: {discovery['token_endpoint']}")
    except Exception as exc:
        lines.append(f"- discovery: failed ({exc})")

    state = _json_load(AUTH_PATH)
    token = str((state.get("tokens") or {}).get("access_token") or "")
    refresh_token = str((state.get("tokens") or {}).get("refresh_token") or "")
    lines.append(f"- stored credentials: {'yes' if token else 'no'}")
    lines.append(f"- refresh token: {'yes' if refresh_token else 'no'}")
    lines.append(
        f"- frontier provider: {'available as grok-oauth' if token else 'not available until login completes'}"
    )
    if token:
        try:
            _get_access_token(force_refresh=force_refresh)
            lines.append(f"- token usable: yes{' (refresh checked)' if force_refresh else ''}")
        except Exception as exc:
            lines.append(f"- token usable: no ({exc})")

    if test_api:
        result = tool_xai_oauth_test({"prompt": "Reply with exactly: xAI OAuth OK", "max_output_tokens": 32})
        first_line = result.splitlines()[0] if result else "empty response"
        lines.append(f"- api test: {first_line}")
    else:
        lines.append("- api test: skipped (pass test_api=true to run)")

    return "\n".join(lines)


@register_tool("xai_oauth_status")
def tool_xai_oauth_status(args: dict) -> str:
    """Show xAI OAuth status without exposing tokens."""
    del args
    state = _json_load(AUTH_PATH)
    token = str((state.get("tokens") or {}).get("access_token") or "")
    if not token:
        return "xAI OAuth is not connected."
    exp_text = "unknown"
    if "." in token:
        try:
            payload = token.split(".")[1]
            payload += "=" * (-len(payload) % 4)
            data = json.loads(base64.urlsafe_b64decode(payload.encode("ascii")).decode("utf-8"))
            if isinstance(data.get("exp"), (int, float)):
                exp_text = datetime.fromtimestamp(data["exp"], tz=UTC).isoformat()
        except Exception:
            pass
    expires_at = (state.get("tokens") or {}).get("expires_at")
    if exp_text == "unknown" and isinstance(expires_at, (int, float)):
        exp_text = datetime.fromtimestamp(expires_at, tz=UTC).isoformat()
    return (
        "xAI OAuth is connected.\n"
        f"Provider: {state.get('provider', 'xai-oauth')}\n"
        f"Model: {state.get('model', DEFAULT_MODEL)}\n"
        f"Access token expires: {exp_text}\n"
        f"Last refresh: {state.get('last_refresh', 'unknown')}"
    )


@register_tool("xai_oauth_test")
def tool_xai_oauth_test(args: dict) -> str:
    """Make a small xAI request using the stored OAuth bearer."""
    prompt = str(args.get("prompt") or "Reply with exactly: xAI OAuth OK").strip()
    token, _state = _get_access_token()
    payload = {
        "model": str(args.get("model") or DEFAULT_MODEL),
        "input": prompt,
        "max_output_tokens": int(args.get("max_output_tokens") or 32),
    }
    resp = httpx.post(
        f"{XAI_API_BASE_URL}/responses",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload,
        timeout=60.0,
    )
    if resp.status_code == 401:
        token, _state = _get_access_token(force_refresh=True)
        resp = httpx.post(
            f"{XAI_API_BASE_URL}/responses",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=60.0,
        )
    if resp.status_code >= 400:
        return f"xAI OAuth test failed with HTTP {resp.status_code}: {resp.text[:1200]}"
    data = resp.json()
    text = data.get("output_text")
    if not text:
        chunks = []
        for item in data.get("output", []) or []:
            for content in item.get("content", []) or []:
                if content.get("type") in {"output_text", "text"} and content.get("text"):
                    chunks.append(str(content["text"]))
        text = "\n".join(chunks).strip()
    return f"xAI OAuth test succeeded.\n{text or json.dumps(data)[:1200]}"


@register_tool("xai_x_search")
def tool_xai_x_search(args: dict) -> str:
    """Search X through xAI's Responses API server-side x_search tool."""
    query = str(args.get("query") or "").strip()
    if not query:
        return "Error: query is required."
    token, _state = _get_access_token()
    payload = {
        "model": str(args.get("model") or DEFAULT_MODEL),
        "input": query,
        "tools": [{"type": "x_search"}],
    }
    resp = httpx.post(
        f"{XAI_API_BASE_URL}/responses",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload,
        timeout=float(args.get("timeout_seconds") or 120),
    )
    if resp.status_code == 401:
        token, _state = _get_access_token(force_refresh=True)
        resp = httpx.post(
            f"{XAI_API_BASE_URL}/responses",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=float(args.get("timeout_seconds") or 120),
        )
    if resp.status_code >= 400:
        return f"xAI X search failed with HTTP {resp.status_code}: {resp.text[:1200]}"
    data = resp.json()
    return data.get("output_text") or json.dumps(data, indent=2)[:4000]
