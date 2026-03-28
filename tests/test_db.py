"""Tests for the SQLite storage layer."""

import json

import pytest

from puget import db


@pytest.fixture()
def conn(tmp_path, monkeypatch):
    """Create an in-memory-style db in a temp directory."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("PUGET_DB", str(db_path))
    return db.connect()


# -- Waves -------------------------------------------------------------------

class TestWaves:
    def test_new_wave_returns_id(self, conn):
        wid = db.new_wave(conn)
        assert isinstance(wid, int)
        assert wid >= 1

    def test_current_wave_id_empty(self, conn):
        assert db.current_wave_id(conn) is None

    def test_current_wave_id_returns_latest(self, conn):
        w1 = db.new_wave(conn)
        w2 = db.new_wave(conn)
        assert db.current_wave_id(conn) == w2
        assert w2 > w1

    def test_ensure_wave_creates_if_none(self, conn):
        assert db.current_wave_id(conn) is None
        wid = db.ensure_wave(conn)
        assert isinstance(wid, int)
        assert db.current_wave_id(conn) == wid

    def test_ensure_wave_reuses_existing(self, conn):
        w1 = db.new_wave(conn)
        w2 = db.ensure_wave(conn)
        assert w1 == w2

    def test_wave_with_label(self, conn):
        wid = db.new_wave(conn, label="debugging imports")
        preview = db.wave_preview(conn, wid)
        assert preview == "debugging imports"

    def test_wave_preview_falls_back_to_first_message(self, conn):
        wid = db.new_wave(conn)
        db.add_turn(conn, wid, "user", "How do I fix this bug?")
        preview = db.wave_preview(conn, wid)
        assert preview == "How do I fix this bug?"

    def test_wave_preview_truncates(self, conn):
        wid = db.new_wave(conn)
        long_msg = "x" * 300
        db.add_turn(conn, wid, "user", long_msg)
        preview = db.wave_preview(conn, wid, max_chars=50)
        assert len(preview) == 51  # 50 chars + "…"
        assert preview.endswith("…")

    def test_wave_preview_empty(self, conn):
        wid = db.new_wave(conn)
        assert db.wave_preview(conn, wid) == "(empty)"


# -- Turns -------------------------------------------------------------------

class TestTurns:
    def test_add_and_get_turns(self, conn):
        wid = db.new_wave(conn)
        db.add_turn(conn, wid, "user", "hello")
        db.add_turn(conn, wid, "assistant", "hi there")

        turns = db.get_turns(conn, wid)
        assert len(turns) == 2
        assert turns[0]["role"] == "user"
        assert turns[0]["content"] == "hello"
        assert turns[1]["role"] == "assistant"
        assert turns[1]["content"] == "hi there"

    def test_turn_with_tool_calls(self, conn):
        wid = db.new_wave(conn)
        tc = json.dumps([{"function": {"name": "bash", "arguments": {"command": "ls"}}}])
        db.add_turn(conn, wid, "assistant", "Let me check.", tool_calls=tc)

        turns = db.get_turns(conn, wid)
        assert turns[0]["content"] == "Let me check."
        assert turns[0]["tool_calls"] == tc

    def test_tool_calls_default_none(self, conn):
        wid = db.new_wave(conn)
        db.add_turn(conn, wid, "user", "hello")
        turns = db.get_turns(conn, wid)
        assert turns[0]["tool_calls"] is None

    def test_last_assistant_turn(self, conn):
        wid = db.new_wave(conn)
        db.add_turn(conn, wid, "user", "hello")
        db.add_turn(conn, wid, "assistant", "first")
        db.add_turn(conn, wid, "user", "again")
        db.add_turn(conn, wid, "assistant", "second")

        last = db.last_assistant_turn(conn, wid)
        assert last is not None
        assert last["content"] == "second"

    def test_last_assistant_turn_none(self, conn):
        wid = db.new_wave(conn)
        assert db.last_assistant_turn(conn, wid) is None

    def test_turns_ordered_chronologically(self, conn):
        wid = db.new_wave(conn)
        for i in range(5):
            db.add_turn(conn, wid, "user", f"msg-{i}")

        turns = db.get_turns(conn, wid)
        contents = [t["content"] for t in turns]
        assert contents == ["msg-0", "msg-1", "msg-2", "msg-3", "msg-4"]

    def test_turns_isolated_between_waves(self, conn):
        w1 = db.new_wave(conn)
        w2 = db.new_wave(conn)
        db.add_turn(conn, w1, "user", "wave one")
        db.add_turn(conn, w2, "user", "wave two")

        assert len(db.get_turns(conn, w1)) == 1
        assert len(db.get_turns(conn, w2)) == 1
        assert db.get_turns(conn, w1)[0]["content"] == "wave one"


# -- Messages for model ------------------------------------------------------

class TestMessagesForModel:
    def test_includes_system_message(self, conn):
        wid = db.new_wave(conn)
        messages = db.messages_for_model(conn, wid)
        assert len(messages) == 1
        assert messages[0]["role"] == "system"

    def test_reconstructs_plain_conversation(self, conn):
        wid = db.new_wave(conn)
        db.add_turn(conn, wid, "user", "hello")
        db.add_turn(conn, wid, "assistant", "hi")

        messages = db.messages_for_model(conn, wid)
        assert len(messages) == 3  # system + user + assistant
        assert messages[1] == {"role": "user", "content": "hello"}
        assert messages[2] == {"role": "assistant", "content": "hi"}

    def test_reconstructs_tool_calls(self, conn):
        wid = db.new_wave(conn)
        tc = [{"function": {"name": "bash", "arguments": {"command": "ls"}}}]
        db.add_turn(conn, wid, "assistant", "checking...", tool_calls=json.dumps(tc))

        messages = db.messages_for_model(conn, wid)
        assistant_msg = messages[1]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"] == "checking..."
        assert assistant_msg["tool_calls"] == tc
