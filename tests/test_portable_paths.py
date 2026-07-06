import importlib
import sys
from pathlib import Path

import maude_cli


CLIENT_PATH = Path(__file__).resolve().parents[1] / "maude-client"
if str(CLIENT_PATH) not in sys.path:
    sys.path.insert(0, str(CLIENT_PATH))


def test_default_config_dir_windows(monkeypatch, tmp_path):
    monkeypatch.delenv("MAUDE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("MAUDE_CONFIG_FILE", raising=False)
    monkeypatch.setattr(maude_cli.platform, "system", lambda: "Windows")
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))

    assert maude_cli._default_config_dir() == tmp_path / "Roaming" / "Maude"


def test_default_config_dir_posix_xdg(monkeypatch, tmp_path):
    monkeypatch.delenv("MAUDE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("MAUDE_CONFIG_FILE", raising=False)
    monkeypatch.setattr(maude_cli.platform, "system", lambda: "Linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

    assert maude_cli._default_config_dir() == tmp_path / "config" / "maude"


def test_default_data_dir_windows(monkeypatch, tmp_path):
    monkeypatch.delenv("MAUDE_DATA_DIR", raising=False)
    monkeypatch.setenv("MAUDE_CONFIG_FILE", str(tmp_path / "missing.json"))
    monkeypatch.setattr(maude_cli.platform, "system", lambda: "Windows")
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))

    assert maude_cli._default_data_dir() == tmp_path / "Roaming" / "Maude"


def test_default_data_dir_saved_config(monkeypatch, tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text('{"data_dir":"/tmp/saved-maude-data"}')
    monkeypatch.delenv("MAUDE_DATA_DIR", raising=False)
    monkeypatch.setenv("MAUDE_CONFIG_FILE", str(config_file))

    assert maude_cli._default_data_dir() == Path("/tmp/saved-maude-data")


def test_client_ssh_tools_fail_without_ssh_host(monkeypatch):
    monkeypatch.delenv("MAUDE_SERVER_SSH_HOST", raising=False)
    monkeypatch.delenv("SERVER_SSH_HOST", raising=False)
    import maude_client.config as config
    import maude_client.client_tools as client_tools

    importlib.reload(config)
    importlib.reload(client_tools)

    assert "MAUDE_SERVER_SSH_HOST" in client_tools.run_server_command("pwd")
    assert "MAUDE_SERVER_SSH_HOST" in client_tools.send_to_server_maude("hello")
