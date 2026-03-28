"""Tests for the built-in tools."""

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
