"""Muse Spark routing and Muse Image tool."""

import base64
import json

from gateway.state import get_model_route
from maude_core.tools_media import tool_generate_image_muse


class TestMuseSparkRoute:
    def test_canonical_id(self):
        name, route = get_model_route("muse-spark-1.3")
        assert name == "muse-spark-1.3"
        assert route["provider"] == "meta"
        assert route["api_key_env"] == "MODEL_API_KEY"
        assert route["base_url"] == "https://api.meta.ai"

    def test_aliases(self):
        for alias in ("muse", "spark", "muse-spark"):
            name, route = get_model_route(alias)
            assert name == "muse-spark-1.3"
            assert route["provider"] == "meta"


class TestGenerateImageMuse:
    def test_missing_key(self, monkeypatch):
        monkeypatch.delenv("MODEL_API_KEY", raising=False)
        monkeypatch.delenv("META_API_KEY", raising=False)
        out = tool_generate_image_muse("a red fox")
        assert "MODEL_API_KEY" in out

    def test_saves_image(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MODEL_API_KEY", "test-key")
        from pathlib import Path

        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

        png = b"\x89PNG\r\n\x1a\n"
        payload = json.dumps({"data": [{"b64_json": base64.b64encode(png).decode()}]}).encode()

        class FakeResp:
            status = 200

            def read(self):
                return payload

        class FakeConn:
            def request(self, *args, **kwargs):
                return None

            def getresponse(self):
                return FakeResp()

            def close(self):
                return None

        monkeypatch.setattr("maude_core.tools_media.http.client.HTTPSConnection", lambda *a, **k: FakeConn())
        out = tool_generate_image_muse("a red fox", width=1024, height=1024, output_format="png")
        assert "Image generated successfully" in out
        assert "/download/muse_" in out
        shared = tmp_path / "nvidia-workbench" / "terminal-llm" / "shared"
        files = list(shared.glob("muse_*.png"))
        assert len(files) == 1
        assert files[0].read_bytes() == png
