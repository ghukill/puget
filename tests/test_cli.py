"""Tests for CLI stdin handling helpers."""

import io

from puget.cli import _combine_message, _read_stdin


class _FakeTTY(io.StringIO):
    def isatty(self) -> bool:
        return True


class _FakePipe(io.StringIO):
    def isatty(self) -> bool:
        return False


def test_read_stdin_returns_none_when_tty(monkeypatch):
    monkeypatch.setattr("sys.stdin", _FakeTTY("ignored"))
    assert _read_stdin() is None


def test_read_stdin_returns_piped_data(monkeypatch):
    monkeypatch.setattr("sys.stdin", _FakePipe("hello from pipe"))
    assert _read_stdin() == "hello from pipe"


def test_read_stdin_returns_none_for_whitespace_only(monkeypatch):
    monkeypatch.setattr("sys.stdin", _FakePipe("   \n\t\n"))
    assert _read_stdin() is None


def test_read_stdin_returns_none_for_empty_pipe(monkeypatch):
    monkeypatch.setattr("sys.stdin", _FakePipe(""))
    assert _read_stdin() is None


def test_read_stdin_handles_os_error(monkeypatch):
    class _Broken:
        def isatty(self):
            return False

        def read(self):
            raise OSError("stdin unavailable")

    monkeypatch.setattr("sys.stdin", _Broken())
    assert _read_stdin() is None


def test_combine_message_no_stdin_returns_message():
    assert _combine_message("hello", None) == "hello"


def test_combine_message_stdin_only_returns_stdin():
    assert _combine_message("", "piped data") == "piped data"


def test_combine_message_both_joins_with_double_newline():
    assert _combine_message("summarize", "file contents") == "summarize\n\nfile contents"


def test_combine_message_preserves_stdin_verbatim():
    stdin = "line1\nline2\n  indented\n"
    assert _combine_message("msg", stdin) == f"msg\n\n{stdin}"
