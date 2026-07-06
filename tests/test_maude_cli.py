import os

import maude_cli


def test_api_url_appends_v1():
    assert maude_cli._api_url("http://127.0.0.1:8080") == "http://127.0.0.1:8080/v1"


def test_api_url_preserves_v1_suffix():
    assert maude_cli._api_url("http://127.0.0.1:8080/v1") == "http://127.0.0.1:8080/v1"


def test_base_url_strips_v1_suffix():
    assert maude_cli._base_url("http://127.0.0.1:8080/v1") == "http://127.0.0.1:8080"


def test_set_gateway_env_sets_primary_and_legacy_alias(monkeypatch):
    monkeypatch.delenv("MAUDE_GATEWAY_URL", raising=False)
    monkeypatch.delenv("LLM_SERVER_URL", raising=False)

    api_url = maude_cli._set_gateway_env("http://example.test:8080")

    assert api_url == "http://example.test:8080/v1"
    assert os.environ["MAUDE_GATEWAY_URL"] == "http://example.test:8080/v1"
    assert os.environ["LLM_SERVER_URL"] == "http://example.test:8080/v1"


def test_default_gateway_command_is_local_only_8080():
    parser = maude_cli.build_parser()
    args = parser.parse_args(["gateway"])

    assert args.bind == "127.0.0.1"
    assert args.port == 8080


def test_doctor_collects_gateway_and_paths(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    shared_dir = data_dir / "shared"
    transfers_dir = data_dir / "transfers"
    conversations_dir = data_dir / "conversations"
    pwa_dir = tmp_path / "pwa"
    for path in (shared_dir, transfers_dir, conversations_dir, pwa_dir):
        path.mkdir(parents=True)
    (pwa_dir / "index.html").write_text("ok")

    monkeypatch.setenv("MAUDE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MAUDE_PWA_DIR", str(pwa_dir))

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"status":"ok","gateway_port":8080,"services":{"llama_server":{"status":"down","port":30010}}}'

    monkeypatch.setattr(maude_cli, "urlopen", lambda url, timeout=0: _Resp())
    monkeypatch.setattr(maude_cli, "_command_summary", lambda command: (True, "stub"))

    checks = dict(maude_cli._collect_doctor_checks("http://example.test:8080/v1"))

    assert checks["gateway api"] == "http://example.test:8080/v1"
    assert checks["gateway base"] == "http://example.test:8080"
    assert checks["gateway health"] == "ok HTTP 200 (ok)"
    assert checks["gateway port"] == "8080"
    assert checks["shared dir"].startswith("ok - ")
    assert checks["pwa build"].startswith("ok - ")


def test_run_doctor_prints_report(monkeypatch, capsys):
    monkeypatch.setattr(maude_cli, "_collect_doctor_checks", lambda gateway: [("gateway api", gateway), ("browser tools", "available")])

    result = maude_cli._run_doctor(type("Args", (), {"gateway": "http://example.test:8080/v1"})())

    out = capsys.readouterr().out
    assert result == 0
    assert "gateway api" in out
    assert "browser tools" in out


def test_setup_writes_config_file(monkeypatch, tmp_path, capsys):
    config_file = tmp_path / "config.json"
    data_dir = tmp_path / "data"
    monkeypatch.setenv("MAUDE_CONFIG_FILE", str(config_file))
    monkeypatch.delenv("MAUDE_GATEWAY_URL", raising=False)
    monkeypatch.delenv("LLM_SERVER_URL", raising=False)

    args = type(
        "Args",
        (),
        {
            "profile": "lan",
            "gateway": "http://maude-host:9090",
            "bind": None,
            "port": None,
            "data_dir": str(data_dir),
            "shared_dir": None,
            "transfers_dir": None,
            "conversations_dir": None,
            "pwa_dir": None,
        },
    )()

    result = maude_cli._run_setup(args)

    assert result == 0
    config = maude_cli._load_config()
    assert config["profile"] == "lan"
    assert config["gateway_url"] == "http://maude-host:9090/v1"
    assert config["gateway_host"] == "0.0.0.0"
    assert config["gateway_port"] == 8080
    assert (data_dir / "shared").is_dir()
    assert "Config file" in capsys.readouterr().out


def test_parser_uses_saved_config(monkeypatch, tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(
        '{"gateway_url":"http://saved-host:9000/v1","gateway_host":"0.0.0.0","gateway_port":9000,"data_dir":"/tmp/maude-data"}'
    )
    monkeypatch.setenv("MAUDE_CONFIG_FILE", str(config_file))
    monkeypatch.delenv("MAUDE_GATEWAY_URL", raising=False)
    monkeypatch.delenv("LLM_SERVER_URL", raising=False)
    monkeypatch.delenv("MAUDE_GATEWAY_HOST", raising=False)
    monkeypatch.delenv("MAUDE_GATEWAY_PORT", raising=False)

    parser = maude_cli.build_parser()

    assert parser.parse_args([]).gateway == "http://saved-host:9000/v1"
    gateway_args = parser.parse_args(["gateway"])
    assert gateway_args.bind == "0.0.0.0"
    assert gateway_args.port == 9000
    assert gateway_args.data_dir == "/tmp/maude-data"
    assert parser.parse_args(["doctor"]).gateway == "http://saved-host:9000/v1"


def test_env_overrides_saved_config(monkeypatch, tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text('{"gateway_url":"http://saved-host:9000/v1","gateway_host":"0.0.0.0","gateway_port":9000}')
    monkeypatch.setenv("MAUDE_CONFIG_FILE", str(config_file))
    monkeypatch.setenv("MAUDE_GATEWAY_URL", "http://env-host:7000")
    monkeypatch.setenv("MAUDE_GATEWAY_HOST", "127.0.0.2")
    monkeypatch.setenv("MAUDE_GATEWAY_PORT", "7000")

    parser = maude_cli.build_parser()

    assert parser.parse_args([]).gateway == "http://env-host:7000/v1"
    gateway_args = parser.parse_args(["gateway"])
    assert gateway_args.bind == "127.0.0.2"
    assert gateway_args.port == 7000


def test_doctor_reports_config_file(monkeypatch, tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text('{"profile":"local","gateway_url":"http://saved-host:9000/v1"}')
    monkeypatch.setenv("MAUDE_CONFIG_FILE", str(config_file))
    monkeypatch.setattr(maude_cli, "urlopen", lambda url, timeout=0: (_ for _ in ()).throw(OSError("offline")))
    monkeypatch.setattr(maude_cli, "_command_summary", lambda command: (True, "stub"))

    checks = dict(maude_cli._collect_doctor_checks("http://saved-host:9000/v1"))

    assert checks["config file"] == str(config_file)
    assert checks["profile"] == "local"
