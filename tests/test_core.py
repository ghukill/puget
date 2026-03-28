"""Tests for the core turn execution engine.

These tests mock model.chat() to avoid hitting Ollama. The focus is on
verifying that turn() and run() correctly store turns, loop on tool calls,
and terminate on text responses.
"""

import json
from unittest.mock import patch

import pytest

from puget import core, db


@pytest.fixture()
def conn(tmp_path, monkeypatch):
    """Create a test database."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("PUGET_DB", str(db_path))
    return db.connect()


def _text_response(content: str) -> dict:
    """Build a mock model response with text only."""
    return {"role": "assistant", "content": content, "tool_calls": None}


def _tool_response(
    content: str,
    tool_name: str,
    arguments: dict,
) -> dict:
    """Build a mock model response with a tool call."""
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [{"function": {"name": tool_name, "arguments": arguments}}],
    }


# -- turn() ------------------------------------------------------------------

class TestTurn:
    @patch("puget.core.model.chat")
    def test_stores_user_message(self, mock_chat, conn):
        mock_chat.return_value = _text_response("hi")
        wid = db.new_wave(conn)

        core.turn(conn, wid, "hello")

        turns = db.get_turns(conn, wid)
        assert turns[0]["role"] == "user"
        assert turns[0]["content"] == "hello"

    @patch("puget.core.model.chat")
    def test_stores_text_response(self, mock_chat, conn):
        mock_chat.return_value = _text_response("I'm here to help")
        wid = db.new_wave(conn)

        response = core.turn(conn, wid, "hello")

        assert response["content"] == "I'm here to help"
        assert response["tool_calls"] is None

        turns = db.get_turns(conn, wid)
        assistant_turn = turns[1]
        assert assistant_turn["role"] == "assistant"
        assert assistant_turn["content"] == "I'm here to help"
        assert assistant_turn["tool_calls"] is None

    @patch("puget.core.model.chat")
    def test_stores_tool_call_response(self, mock_chat, conn):
        mock_chat.return_value = _tool_response("checking...", "bash", {"command": "ls"})
        wid = db.new_wave(conn)

        response = core.turn(conn, wid, "list files")

        assert response["tool_calls"] is not None
        assert len(response["tool_calls"]) == 1

        turns = db.get_turns(conn, wid)
        assistant_turn = turns[1]
        assert assistant_turn["content"] == "checking..."
        assert json.loads(assistant_turn["tool_calls"]) == response["tool_calls"]

    @patch("puget.core.model.chat")
    def test_continues_without_message(self, mock_chat, conn):
        """turn() with message=None doesn't add a user turn."""
        mock_chat.return_value = _text_response("done")
        wid = db.new_wave(conn)
        db.add_turn(conn, wid, "tool", "file list here")

        core.turn(conn, wid)  # no message

        turns = db.get_turns(conn, wid)
        roles = [t["role"] for t in turns]
        assert roles == ["tool", "assistant"]


# -- run() -------------------------------------------------------------------

class TestRun:
    @patch("puget.core.model.chat")
    def test_simple_text_response(self, mock_chat, conn):
        mock_chat.return_value = _text_response("Hello!")
        wid = db.new_wave(conn)

        response = core.run(conn, wid, "hi")

        assert response["content"] == "Hello!"
        assert response["tool_calls"] is None

    @patch("puget.core.tools.execute")
    @patch("puget.core.model.chat")
    def test_tool_call_loop(self, mock_chat, mock_execute, conn):
        """run() should execute tools and continue until text response."""
        mock_chat.side_effect = [
            _tool_response("", "bash", {"command": "ls"}),
            _text_response("Found 3 files."),
        ]
        mock_execute.return_value = "file1.py\nfile2.py\nfile3.py"
        wid = db.new_wave(conn)

        response = core.run(conn, wid, "list files")

        assert response["content"] == "Found 3 files."
        mock_execute.assert_called_once_with("bash", {"command": "ls"})

        # Verify the full turn sequence in the db
        turns = db.get_turns(conn, wid)
        roles = [t["role"] for t in turns]
        assert roles == ["user", "assistant", "tool", "assistant"]
        assert turns[2]["content"] == "file1.py\nfile2.py\nfile3.py"

    @patch("puget.core.tools.execute")
    @patch("puget.core.model.chat")
    def test_chained_tool_calls(self, mock_chat, mock_execute, conn):
        """run() handles multiple rounds of tool calls."""
        mock_chat.side_effect = [
            _tool_response("", "bash", {"command": "find . -name '*.py'"}),
            _tool_response("", "bash", {"command": "wc -l *.py"}),
            _text_response("Total: 500 lines."),
        ]
        mock_execute.side_effect = [
            "foo.py\nbar.py",
            "  200 foo.py\n  300 bar.py\n  500 total",
        ]
        wid = db.new_wave(conn)

        response = core.run(conn, wid, "count lines")

        assert response["content"] == "Total: 500 lines."
        assert mock_execute.call_count == 2
        assert mock_chat.call_count == 3

        turns = db.get_turns(conn, wid)
        roles = [t["role"] for t in turns]
        assert roles == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
