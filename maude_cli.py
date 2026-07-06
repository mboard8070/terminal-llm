"""Portable MAUDE command line entrypoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import webbrowser
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


DEFAULT_LOCAL_PORT = 8080
DEFAULT_LOCAL_BIND = "127.0.0.1"
LEGACY_SPARK_URL = "http://spark-e26c:30000/v1"
PROJECT_ROOT = Path(__file__).resolve().parent


def _api_url(gateway: str) -> str:
    gateway = gateway.rstrip("/")
    if gateway.endswith("/v1"):
        return gateway
    return f"{gateway}/v1"


def _base_url(gateway: str) -> str:
    gateway = gateway.rstrip("/")
    if gateway.endswith("/v1"):
        return gateway[:-3].rstrip("/")
    return gateway


def _default_config_dir() -> Path:
    if os.environ.get("MAUDE_CONFIG_DIR"):
        return Path(os.environ["MAUDE_CONFIG_DIR"]).expanduser()
    if platform.system() == "Windows":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "Maude"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "maude"


def _config_path() -> Path:
    if os.environ.get("MAUDE_CONFIG_FILE"):
        return Path(os.environ["MAUDE_CONFIG_FILE"]).expanduser()
    return _default_config_dir() / "config.json"


def _load_config() -> dict:
    path = _config_path()
    try:
        data = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_config(config: dict) -> Path:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return path


def _config_value(config: dict, key: str, env_names: tuple[str, ...], default=None):
    for env_name in env_names:
        value = os.environ.get(env_name)
        if value not in (None, ""):
            return value
    return config.get(key, default)


def _configured_gateway(config: dict | None = None) -> str:
    config = _load_config() if config is None else config
    return str(_config_value(config, "gateway_url", ("MAUDE_GATEWAY_URL", "LLM_SERVER_URL"), f"http://127.0.0.1:{DEFAULT_LOCAL_PORT}/v1"))


def _configured_bind(config: dict | None = None) -> str:
    config = _load_config() if config is None else config
    return str(_config_value(config, "gateway_host", ("MAUDE_GATEWAY_HOST",), DEFAULT_LOCAL_BIND))


def _configured_port(config: dict | None = None) -> int:
    config = _load_config() if config is None else config
    return int(_config_value(config, "gateway_port", ("MAUDE_GATEWAY_PORT",), DEFAULT_LOCAL_PORT))


def _set_gateway_env(gateway: str) -> str:
    api_url = _api_url(gateway)
    os.environ["MAUDE_GATEWAY_URL"] = api_url
    os.environ.setdefault("LLM_SERVER_URL", api_url)
    return api_url


def _default_data_dir() -> Path:
    if os.environ.get("MAUDE_DATA_DIR"):
        return Path(os.environ["MAUDE_DATA_DIR"]).expanduser()
    config = _load_config()
    if config.get("data_dir"):
        return Path(config["data_dir"]).expanduser()
    if platform.system() == "Windows":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "Maude"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "maude"


def _command_summary(command: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5)
    except FileNotFoundError:
        return False, "not installed"
    except Exception as exc:
        return False, f"error: {exc}"

    output = result.stdout or result.stderr
    first = output.splitlines()[0] if output else "installed"
    if result.returncode == 0:
        return True, first
    return False, first or f"exit {result.returncode}"


def _module_available(import_name: str) -> tuple[bool, str]:
    if importlib.util.find_spec(import_name):
        return True, "installed"
    return False, "not installed"


def _path_check(path: Path, kind: str = "dir") -> tuple[bool, str]:
    path = path.expanduser()
    exists = path.is_dir() if kind == "dir" else path.is_file()
    return exists, str(path)


def _status_text(ok: bool, value: str) -> str:
    return f"ok - {value}" if ok else f"missing - {value}"


def _run_gateway(args: argparse.Namespace) -> None:
    os.environ["MAUDE_GATEWAY_HOST"] = args.bind
    os.environ["MAUDE_GATEWAY_PORT"] = str(args.port)
    if args.data_dir:
        os.environ["MAUDE_DATA_DIR"] = str(Path(args.data_dir).expanduser())
    if args.shared_dir:
        os.environ["MAUDE_SHARED_DIR"] = str(Path(args.shared_dir).expanduser())
    if args.pwa_dir:
        os.environ["MAUDE_PWA_DIR"] = str(Path(args.pwa_dir).expanduser())

    from gateway.main import main as gateway_main

    gateway_main()


def _run_client(args: argparse.Namespace) -> None:
    _set_gateway_env(args.gateway)
    from chat_lite import main as chat_main

    chat_main()


def _run_default(args: argparse.Namespace) -> None:
    _set_gateway_env(args.gateway)
    from chat_lite import main as chat_main

    chat_main()


def _run_web(args: argparse.Namespace) -> None:
    base = _base_url(args.gateway)
    pwa_dir = Path(args.pwa_dir or _load_config().get("pwa_dir") or PROJECT_ROOT / "maude-phone" / "dist").expanduser()
    if not (pwa_dir / "index.html").exists():
        print(f"Warning: PWA build not found at {pwa_dir}. Run `npm --prefix maude-phone run build` or set MAUDE_PWA_DIR.")
    url = f"{base}/app/"
    print(f"Opening {url}")
    webbrowser.open(url)


def _collect_doctor_checks(gateway: str) -> list[tuple[str, str]]:
    config = _load_config()
    base = _base_url(gateway)
    api = _api_url(gateway)
    data_dir = _default_data_dir()
    shared_dir = Path(os.environ.get("MAUDE_SHARED_DIR", config.get("shared_dir", data_dir / "shared"))).expanduser()
    transfers_dir = Path(os.environ.get("MAUDE_TRANSFERS_DIR", config.get("transfers_dir", data_dir / "transfers"))).expanduser()
    conversations_dir = Path(os.environ.get("MAUDE_CONVERSATIONS_DIR", config.get("conversations_dir", data_dir / "conversations"))).expanduser()
    pwa_dir = Path(os.environ.get("MAUDE_PWA_DIR", config.get("pwa_dir", PROJECT_ROOT / "maude-phone" / "dist"))).expanduser()

    checks: list[tuple[str, str]] = [
        ("python", sys.version.split()[0]),
        ("platform", f"{platform.system()} {platform.release()}"),
        ("hostname", socket.gethostname()),
        ("config file", str(_config_path())),
        ("profile", str(config.get("profile", "default"))),
        ("gateway api", api),
        ("gateway base", base),
    ]

    try:
        with urlopen(f"{base}/health", timeout=5) as resp:
            raw = resp.read().decode(errors="replace")
            report = json.loads(raw) if raw else {}
            status = report.get("status", "unknown")
            checks.append(("gateway health", f"ok HTTP {resp.status} ({status})"))
            if "gateway_port" in report:
                checks.append(("gateway port", str(report["gateway_port"])))
            services = report.get("services") or {}
            for name in ("llama_server", "voice_server"):
                if name in services:
                    svc = services[name]
                    checks.append((name, f"{svc.get('status', 'unknown')} on {svc.get('port', '?')}"))
    except URLError as exc:
        checks.append(("gateway health", f"unreachable: {exc.reason}"))
    except Exception as exc:
        checks.append(("gateway health", f"unreachable: {exc}"))

    for label, path, kind in (
        ("data dir", data_dir, "dir"),
        ("shared dir", shared_dir, "dir"),
        ("transfers dir", transfers_dir, "dir"),
        ("conversations dir", conversations_dir, "dir"),
        ("pwa build", pwa_dir / "index.html", "file"),
    ):
        ok, value = _path_check(path, kind)
        checks.append((label, _status_text(ok, value)))

    for key in ("OPEN_ROUTER_API_KEY", "MISTRAL_API_KEY", "OPENAI_API_KEY", "CLAUDE_API_KEY", "CODESTRAL_API_KEY"):
        checks.append((key, "set" if os.environ.get(key) else "not set"))

    for name, command in {
        "git": ["git", "--version"],
        "gh": ["gh", "--version"],
        "tailscale": ["tailscale", "version"],
        "playwright cli": ["playwright", "--version"],
        "node": ["node", "--version"],
        "npm": ["npm", "--version"],
    }.items():
        ok, value = _command_summary(command)
        checks.append((name, _status_text(ok, value)))

    if shutil.which("tailscale"):
        ok, value = _command_summary(["tailscale", "status"])
        checks.append(("tailscale status", _status_text(ok, value)))

    for label, import_name in {
        "playwright py": "playwright",
        "rich": "rich",
        "openai": "openai",
        "requests": "requests",
        "google api": "googleapiclient",
        "croniter": "croniter",
        "telegram": "telegram",
    }.items():
        ok, value = _module_available(import_name)
        checks.append((label, _status_text(ok, value)))

    checks.append(("browser tools", "available" if importlib.util.find_spec("playwright") else "unavailable - playwright not installed"))
    checks.append(("github tools", "available" if shutil.which("gh") else "unavailable - gh not installed"))
    checks.append(("scheduler", "available" if importlib.util.find_spec("croniter") else "unavailable - croniter not installed"))
    checks.append(("google tools", "available" if importlib.util.find_spec("googleapiclient") else "unavailable - google-api-python-client not installed"))
    checks.append(("comfyui", f"configured at {os.environ.get('COMFYUI_HOST', '127.0.0.1')}:{os.environ.get('COMFYUI_PORT', '8188')}"))

    return checks


def _run_doctor(args: argparse.Namespace) -> int:
    checks = _collect_doctor_checks(args.gateway)
    width = max(len(name) for name, _ in checks)
    for name, value in checks:
        print(f"{name:<{width}}  {value}")
    return 0


def _setup_defaults(profile: str) -> tuple[str, str, int]:
    if profile == "spark":
        return LEGACY_SPARK_URL, "0.0.0.0", 30000
    bind = "0.0.0.0" if profile in {"lan", "tailscale"} else DEFAULT_LOCAL_BIND
    return f"http://127.0.0.1:{DEFAULT_LOCAL_PORT}/v1", bind, DEFAULT_LOCAL_PORT


def _run_setup(args: argparse.Namespace) -> int:
    default_gateway, default_bind, default_port = _setup_defaults(args.profile)
    data_dir = Path(args.data_dir or _default_data_dir()).expanduser()
    config = {
        "profile": args.profile,
        "gateway_url": _api_url(args.gateway or default_gateway),
        "gateway_host": args.bind or default_bind,
        "gateway_port": int(args.port or default_port),
        "data_dir": str(data_dir),
        "shared_dir": str(Path(args.shared_dir or data_dir / "shared").expanduser()),
        "transfers_dir": str(Path(args.transfers_dir or data_dir / "transfers").expanduser()),
        "conversations_dir": str(Path(args.conversations_dir or data_dir / "conversations").expanduser()),
        "pwa_dir": str(Path(args.pwa_dir or PROJECT_ROOT / "maude-phone" / "dist").expanduser()),
    }
    path = _write_config(config)
    for key in ("data_dir", "shared_dir", "transfers_dir", "conversations_dir"):
        Path(config[key]).mkdir(parents=True, exist_ok=True)

    print("MAUDE setup")
    print(f"Config file: {path}")
    print(f"Profile: {config['profile']}")
    print(f"Gateway URL: {config['gateway_url']}")
    print(f"Gateway bind: {config['gateway_host']}:{config['gateway_port']}")
    print(f"Data dir: {config['data_dir']}")
    print("Run `maude doctor` to verify the setup.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    config = _load_config()
    configured_gateway = _api_url(_configured_gateway(config))
    configured_base = _base_url(configured_gateway)
    configured_bind = _configured_bind(config)
    configured_port = _configured_port(config)

    parser = argparse.ArgumentParser(prog="maude", description="MAUDE portable gateway and client")
    parser.add_argument("--gateway", default=configured_gateway, help="Gateway URL for TUI/client mode")

    sub = parser.add_subparsers(dest="command")

    gateway = sub.add_parser("gateway", help="Run a MAUDE gateway on this machine")
    gateway.add_argument("--bind", default=configured_bind, help="Bind host, e.g. 127.0.0.1 or 0.0.0.0")
    gateway.add_argument("--port", type=int, default=configured_port, help="Gateway port")
    gateway.add_argument("--data-dir", default=config.get("data_dir"), help="Runtime data directory")
    gateway.add_argument("--shared-dir", default=config.get("shared_dir"), help="Shared files directory")
    gateway.add_argument("--pwa-dir", default=config.get("pwa_dir"), help="Built web/PWA static assets directory")
    gateway.set_defaults(func=_run_gateway)

    client = sub.add_parser("client", help="Run the terminal client against a gateway")
    client.add_argument("--gateway", default=configured_gateway, help="Gateway URL")
    client.set_defaults(func=_run_client)

    web = sub.add_parser("web", help="Open the gateway-served web app")
    web.add_argument("--gateway", default=configured_base, help="Gateway base URL")
    web.add_argument("--pwa-dir", default=config.get("pwa_dir"), help="Built web/PWA static assets directory")
    web.set_defaults(func=_run_web)

    doctor = sub.add_parser("doctor", help="Check local MAUDE dependencies and gateway reachability")
    doctor.add_argument("--gateway", default=configured_gateway, help="Gateway URL")
    doctor.set_defaults(func=_run_doctor)

    setup = sub.add_parser("setup", help="Write first-run setup configuration")
    setup.add_argument("--profile", choices=("local", "lan", "tailscale", "spark"), default="local", help="Gateway access profile")
    setup.add_argument("--gateway", default=None, help="Gateway URL to store")
    setup.add_argument("--bind", default=None, help="Gateway bind host to store")
    setup.add_argument("--port", type=int, default=None, help="Gateway port to store")
    setup.add_argument("--data-dir", default=None, help="Runtime data directory")
    setup.add_argument("--shared-dir", default=None, help="Shared files directory")
    setup.add_argument("--transfers-dir", default=None, help="Transfers directory")
    setup.add_argument("--conversations-dir", default=None, help="Conversations directory")
    setup.add_argument("--pwa-dir", default=None, help="Built web/PWA static assets directory")
    setup.set_defaults(func=_run_setup)

    spark = sub.add_parser("spark", help="Run client against the legacy Spark gateway profile")
    spark.set_defaults(func=lambda _args: _run_client(argparse.Namespace(gateway=LEGACY_SPARK_URL)))

    return parser


def main(argv: list[str] | None = None) -> int | None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "func"):
        return args.func(args)
    return _run_default(args)


if __name__ == "__main__":
    raise SystemExit(main())
