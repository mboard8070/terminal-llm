import os
import time
from pathlib import Path

import social_posting


def test_x_requires_preview_for_images_not_videos():
    assert social_posting._x_requires_preview_before_post("/tmp/product.png") is True
    assert social_posting._x_requires_preview_before_post("/tmp/product.jpg") is True
    assert social_posting._x_requires_preview_before_post("/tmp/product.mp4") is False
    assert social_posting._x_requires_preview_before_post("/tmp/product.mov") is False


def test_latest_shared_video_uses_recent_video(tmp_path, monkeypatch):
    monkeypatch.setattr(social_posting, "__file__", str(tmp_path / "social_posting.py"))
    shared = tmp_path / "shared"
    shared.mkdir()
    old = shared / "old.mp4"
    new = shared / "new.mov"
    old.write_text("old")
    new.write_text("new")
    now = time.time()
    os.utime(old, (now - 60, now - 60))
    os.utime(new, (now, now))

    assert Path(social_posting._latest_shared_video()).name == "new.mov"


def test_x_can_continue_without_preview_requires_enabled_post_button(monkeypatch):
    monkeypatch.setattr(
        social_posting,
        "_x_media_state",
        lambda page: {"hasPreview": False, "processing": False, "error": None, "postEnabled": True},
    )
    assert social_posting._x_can_continue_without_preview(object(), True) is True
    assert social_posting._x_can_continue_without_preview(object(), False) is False

    monkeypatch.setattr(
        social_posting,
        "_x_media_state",
        lambda page: {"hasPreview": False, "processing": False, "error": "Upload failed", "postEnabled": True},
    )
    assert social_posting._x_can_continue_without_preview(object(), True) is False
