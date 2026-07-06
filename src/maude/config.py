"""Runtime path configuration for MAUDE.

Centralizing runtime paths keeps source modules from baking in local storage
locations while preserving legacy defaults for existing clients.
"""

from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_config_dir() -> Path:
    if os.environ.get("MAUDE_CONFIG_DIR"):
        return Path(os.environ["MAUDE_CONFIG_DIR"]).expanduser()
    if platform.system() == "Windows":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "Maude"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "maude"


def _config_file() -> Path:
    if os.environ.get("MAUDE_CONFIG_FILE"):
        return Path(os.environ["MAUDE_CONFIG_FILE"]).expanduser()
    return _default_config_dir() / "config.json"


def _load_config() -> dict:
    try:
        data = json.loads(_config_file().read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _platform_data_dir() -> Path:
    if platform.system() == "Windows":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "Maude"
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Application Support" / "maude"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "maude"


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if not value:
        return default
    return Path(value).expanduser()


@dataclass(frozen=True)
class RuntimePaths:
    """Filesystem locations used by gateway, orchestration, and stateful tools."""

    project_root: Path
    runtime_root: Path
    data_dir: Path
    logs_dir: Path
    certs_dir: Path
    cache_dir: Path
    generated_media_dir: Path
    comfyui_output_dir: Path
    screenshots_dir: Path
    browser_data_dir: Path
    hyperframes_dir: Path
    xai_oauth_file: Path
    xai_oauth_pending_file: Path
    xai_oauth_callback_file: Path
    chat_sync_file: Path
    config_dir: Path
    shared_dir: Path
    transfers_dir: Path
    conversations_dir: Path
    jobs_dir: Path
    schedules_file: Path
    missions_dir: Path
    pwa_dir: Path

    @classmethod
    def from_env(cls) -> RuntimePaths:
        config = _load_config()
        project_root = _path_from_env("MAUDE_PROJECT_ROOT", PROJECT_ROOT)
        config_dir = _path_from_env("MAUDE_CONFIG_DIR", _default_config_dir())

        if os.environ.get("MAUDE_RUNTIME_ROOT"):
            runtime_root = _path_from_env("MAUDE_RUNTIME_ROOT", _platform_data_dir())
            data_default = runtime_root / "runtime" / "data"
            shared_default = runtime_root / "runtime" / "shared"
            transfers_default = runtime_root / "runtime" / "transfers"
        else:
            configured_data = config.get("data_dir")
            data_default = Path(configured_data).expanduser() if configured_data else _platform_data_dir()
            runtime_root = _path_from_env("MAUDE_RUNTIME_ROOT", data_default)
            shared_default = Path(config.get("shared_dir", data_default / "shared")).expanduser()
            transfers_default = Path(config.get("transfers_dir", data_default / "transfers")).expanduser()

        runtime_data_dir = _path_from_env("MAUDE_DATA_DIR", data_default)

        return cls(
            project_root=project_root,
            runtime_root=runtime_root,
            data_dir=runtime_data_dir,
            logs_dir=_path_from_env("MAUDE_LOGS_DIR", runtime_root / "runtime" / "logs"),
            certs_dir=_path_from_env("MAUDE_CERTS_DIR", runtime_root / "runtime" / "certs"),
            cache_dir=_path_from_env("MAUDE_CACHE_DIR", runtime_root / "runtime" / "cache"),
            generated_media_dir=_path_from_env("MAUDE_GENERATED_MEDIA_DIR", runtime_data_dir / "generated_media"),
            comfyui_output_dir=_path_from_env(
                "MAUDE_COMFYUI_OUTPUT_DIR", Path.home() / "nvidia-workbench" / "ComfyUI" / "app" / "output"
            ),
            screenshots_dir=_path_from_env("MAUDE_SCREENSHOTS_DIR", runtime_data_dir / "screenshots"),
            browser_data_dir=_path_from_env("MAUDE_BROWSER_DATA_DIR", runtime_data_dir / "browser_data"),
            hyperframes_dir=_path_from_env("MAUDE_HYPERFRAMES_DIR", runtime_data_dir / "hyperframes"),
            xai_oauth_file=_path_from_env("MAUDE_XAI_OAUTH_FILE", config_dir / "xai_oauth.json"),
            xai_oauth_pending_file=_path_from_env(
                "MAUDE_XAI_OAUTH_PENDING_FILE", config_dir / "xai_oauth_pending.json"
            ),
            xai_oauth_callback_file=_path_from_env(
                "MAUDE_XAI_OAUTH_CALLBACK_FILE", config_dir / "xai_oauth_callback.json"
            ),
            chat_sync_file=_path_from_env("MAUDE_CHAT_SYNC_FILE", config_dir / "chat_sync.jsonl"),
            config_dir=config_dir,
            shared_dir=_path_from_env("MAUDE_SHARED_DIR", shared_default),
            transfers_dir=_path_from_env("MAUDE_TRANSFERS_DIR", transfers_default),
            conversations_dir=_path_from_env("MAUDE_CONVERSATIONS_DIR", Path(config.get("conversations_dir", runtime_data_dir / "conversations")).expanduser()),
            jobs_dir=_path_from_env("MAUDE_JOBS_DIR", config_dir / "jobs"),
            schedules_file=_path_from_env("MAUDE_SCHEDULES_FILE", config_dir / "schedules.json"),
            missions_dir=_path_from_env("MAUDE_MISSIONS_DIR", runtime_data_dir / "missions"),
            pwa_dir=_path_from_env("MAUDE_PWA_DIR", Path(config.get("pwa_dir", project_root / "maude-phone" / "dist")).expanduser()),
        )

    def ensure_gateway_dirs(self) -> None:
        for path in (
            self.shared_dir,
            self.transfers_dir,
            self.conversations_dir,
            self.logs_dir,
            self.certs_dir,
            self.cache_dir,
            self.generated_media_dir,
            self.screenshots_dir,
            self.browser_data_dir,
            self.hyperframes_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def runtime_paths() -> RuntimePaths:
    """Return runtime paths resolved from the current environment."""

    return RuntimePaths.from_env()
