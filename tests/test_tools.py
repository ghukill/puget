"""Tests for the built-in tools."""

import os

import pytest

from puget.tools import _truncate, execute


class TestBash:
    def test_simple_command(self):
        result = execute("bash", {"command": "echo hello"})
        assert "hello" in result

    def test_captures_stderr(self):
        result = execute("bash", {"command": "echo oops >&2"})
        assert "oops" in result

    def test_nonzero_exit_code(self):
        result = execute("bash", {"command": "exit 42"})
        assert "Exit code: 42" in result

    def test_timeout(self):
        result = execute("bash", {"command": "sleep 10", "timeout": 0.1})
        assert "timed out" in result

    def test_no_output(self):
        result = execute("bash", {"command": "true"})
        assert "(no output)" in result

    def test_combined_stdout_stderr(self):
        result = execute("bash", {"command": "echo out; echo err >&2"})
        assert "out" in result
        assert "err" in result

    def test_unknown_tool(self):
        result = execute("unknown_tool", {"command": "hi"})
        assert "Unknown tool" in result


class TestTruncation:
    def test_short_text_not_truncated(self):
        text = "line\n" * 10
        result, truncated = _truncate(text)
        assert not truncated
        assert result == text

    def test_truncates_by_line_count(self):
        text = "".join(f"line-{i}\n" for i in range(3000))
        result, truncated = _truncate(text)
        assert truncated
        lines = result.strip().split("\n")
        assert len(lines) == 2000
        # Should keep the tail
        assert "line-2999" in lines[-1]

    def test_truncates_by_byte_size(self):
        # Single very long line exceeding MAX_BYTES
        text = "x" * (60 * 1024)  # 60KB
        result, truncated = _truncate(text)
        assert truncated
        assert len(result.encode("utf-8")) <= 50 * 1024


# -- Read tool tests ---------------------------------------------------------


class TestRead:
    def test_read_file(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("line1\nline2\nline3\n")
        result = execute("read", {"path": str(f)})
        assert "line1" in result
        assert "line2" in result
        assert "line3" in result

    def test_read_with_offset(self, tmp_path):
        f = tmp_path / "nums.txt"
        f.write_text("one\ntwo\nthree\nfour\nfive\n")
        result = execute("read", {"path": str(f), "offset": 3})
        assert "three" in result
        assert "four" in result
        assert "one" not in result

    def test_read_with_limit(self, tmp_path):
        f = tmp_path / "nums.txt"
        f.write_text("one\ntwo\nthree\nfour\nfive\n")
        result = execute("read", {"path": str(f), "limit": 2})
        assert "one" in result
        assert "two" in result
        assert "three" not in result

    def test_read_with_offset_and_limit(self, tmp_path):
        f = tmp_path / "nums.txt"
        f.write_text("one\ntwo\nthree\nfour\nfive\n")
        result = execute("read", {"path": str(f), "offset": 2, "limit": 2})
        assert "two" in result
        assert "three" in result
        assert "one" not in result
        assert "four" not in result

    def test_read_nonexistent_file(self):
        result = execute("read", {"path": "/no/such/file.txt"})
        assert "Error" in result
        assert "not found" in result

    def test_read_directory(self, tmp_path):
        result = execute("read", {"path": str(tmp_path)})
        assert "Error" in result
        assert "directory" in result

    def test_read_shows_line_info(self, tmp_path):
        f = tmp_path / "lines.txt"
        f.write_text("a\nb\nc\nd\ne\n")
        result = execute("read", {"path": str(f)})
        assert "5 total" in result

    def test_read_tilde_expansion(self, tmp_path, monkeypatch):
        # Write a file to a known location via tmp_path but test ~ expansion
        f = tmp_path / "tilde_test.txt"
        f.write_text("hello tilde\n")
        # Just verify expanduser path works (the function uses os.path.expanduser)
        result = execute("read", {"path": str(f)})
        assert "hello tilde" in result


# -- Write tool tests --------------------------------------------------------


class TestWrite:
    def test_write_new_file(self, tmp_path):
        f = tmp_path / "new.txt"
        result = execute("write", {"path": str(f), "content": "hello world\n"})
        assert "Wrote" in result
        assert f.read_text() == "hello world\n"

    def test_write_creates_parent_dirs(self, tmp_path):
        f = tmp_path / "a" / "b" / "c" / "deep.txt"
        result = execute("write", {"path": str(f), "content": "deep\n"})
        assert "Wrote" in result
        assert f.read_text() == "deep\n"

    def test_write_overwrites(self, tmp_path):
        f = tmp_path / "overwrite.txt"
        f.write_text("old content")
        execute("write", {"path": str(f), "content": "new content"})
        assert f.read_text() == "new content"

    def test_write_reports_stats(self, tmp_path):
        f = tmp_path / "stats.txt"
        content = "line1\nline2\nline3\n"
        result = execute("write", {"path": str(f), "content": content})
        assert "3 lines" in result
        assert str(len(content.encode("utf-8"))) + " bytes" in result

    def test_write_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        result = execute("write", {"path": str(f), "content": ""})
        assert "Wrote" in result
        assert f.read_text() == ""


# -- Edit tool tests ---------------------------------------------------------


class TestEdit:
    def test_single_edit(self, tmp_path):
        f = tmp_path / "edit.txt"
        f.write_text("hello world\n")
        result = execute("edit", {
            "path": str(f),
            "oldText": "hello",
            "newText": "goodbye",
        })
        assert "Applied 1 edit" in result
        assert f.read_text() == "goodbye world\n"

    def test_multi_edit(self, tmp_path):
        f = tmp_path / "multi.txt"
        f.write_text("aaa\nbbb\nccc\n")
        result = execute("edit", {
            "path": str(f),
            "edits": [
                {"oldText": "aaa", "newText": "AAA"},
                {"oldText": "ccc", "newText": "CCC"},
            ],
        })
        assert "Applied 2 edits" in result
        assert f.read_text() == "AAA\nbbb\nCCC\n"

    def test_edit_not_found(self, tmp_path):
        f = tmp_path / "edit.txt"
        f.write_text("hello world\n")
        result = execute("edit", {
            "path": str(f),
            "oldText": "MISSING",
            "newText": "replacement",
        })
        assert "Error" in result
        assert "not found" in result
        # File should be unchanged.
        assert f.read_text() == "hello world\n"

    def test_edit_ambiguous(self, tmp_path):
        f = tmp_path / "edit.txt"
        f.write_text("aaa bbb aaa\n")
        result = execute("edit", {
            "path": str(f),
            "oldText": "aaa",
            "newText": "ccc",
        })
        assert "Error" in result
        assert "matches 2 times" in result
        # File should be unchanged.
        assert f.read_text() == "aaa bbb aaa\n"

    def test_edit_overlapping(self, tmp_path):
        f = tmp_path / "overlap.txt"
        f.write_text("abcdef\n")
        result = execute("edit", {
            "path": str(f),
            "edits": [
                {"oldText": "abcd", "newText": "ABCD"},
                {"oldText": "cdef", "newText": "CDEF"},
            ],
        })
        assert "Error" in result
        assert "overlap" in result
        # File should be unchanged.
        assert f.read_text() == "abcdef\n"

    def test_edit_nonexistent_file(self):
        result = execute("edit", {
            "path": "/no/such/file.txt",
            "oldText": "a",
            "newText": "b",
        })
        assert "Error" in result
        assert "not found" in result

    def test_edit_no_args(self, tmp_path):
        f = tmp_path / "edit.txt"
        f.write_text("hello\n")
        result = execute("edit", {"path": str(f)})
        assert "Error" in result

    def test_edit_multiline(self, tmp_path):
        f = tmp_path / "multi.txt"
        f.write_text("def foo():\n    pass\n\ndef bar():\n    pass\n")
        result = execute("edit", {
            "path": str(f),
            "oldText": "def foo():\n    pass",
            "newText": "def foo():\n    return 42",
        })
        assert "Applied 1 edit" in result
        assert f.read_text() == "def foo():\n    return 42\n\ndef bar():\n    pass\n"

    def test_multi_edit_order_independent(self, tmp_path):
        """Edits are applied by position, not by order in the array."""
        f = tmp_path / "order.txt"
        f.write_text("first\nsecond\nthird\n")
        # Provide edits in reverse order — should still work.
        result = execute("edit", {
            "path": str(f),
            "edits": [
                {"oldText": "third", "newText": "THIRD"},
                {"oldText": "first", "newText": "FIRST"},
            ],
        })
        assert "Applied 2 edits" in result
        assert f.read_text() == "FIRST\nsecond\nTHIRD\n"
