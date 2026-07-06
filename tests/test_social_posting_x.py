import social_posting


class _EnabledButton:
    def is_enabled(self):
        return True


def test_x_post_button_waits_until_media_processing_clears(monkeypatch):
    states = [
        {"error": None, "processing": True, "hasPreview": False, "postEnabled": True},
        {"error": None, "processing": True, "hasPreview": True, "postEnabled": True},
        {"error": None, "processing": False, "hasPreview": True, "postEnabled": True},
    ]
    seen = []

    def fake_media_state(page):
        state = states[min(len(seen), len(states) - 1)]
        seen.append(state)
        return state

    monkeypatch.setattr(social_posting, "_x_media_state", fake_media_state)
    monkeypatch.setattr(social_posting, "_first_match", lambda root, selectors: _EnabledButton())
    monkeypatch.setattr(social_posting.time, "sleep", lambda seconds: None)

    button, error = social_posting._x_wait_for_post_button(object(), object(), has_media=True)

    assert error is None
    assert isinstance(button, _EnabledButton)
    assert len(seen) == 3


def test_x_post_button_returns_media_error(monkeypatch):
    monkeypatch.setattr(
        social_posting,
        "_x_media_state",
        lambda page: {"error": "Your video file could not be processed.", "processing": False},
    )
    monkeypatch.setattr(social_posting, "_first_match", lambda root, selectors: _EnabledButton())

    button, error = social_posting._x_wait_for_post_button(object(), object(), has_media=True)

    assert button is None
    assert error == "Error: X media upload failed: Your video file could not be processed."
