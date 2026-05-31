import io
import sys
import types


def test_windows_input_ignores_tab_and_extended_keys(monkeypatch):
    import maude_client.cli as cli

    keys = iter(["q", "\t", "u", "\xe0", "H", "i", "t", "\r"])
    fake_msvcrt = types.SimpleNamespace(getwch=lambda: next(keys))
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    monkeypatch.setattr(cli.sys, "stdout", io.StringIO())

    assert cli._windows_input("You") == "quit"
