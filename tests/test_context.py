"""Tests for context budgeting and emergency payload shaping."""

from puget import context


def _system() -> dict:
    return {"role": "system", "content": "You are helpful."}


def test_truncates_tool_output_to_max_tool_chars():
    turns = [
        {"role": "user", "content": "run it", "tool_calls": None},
        {"role": "tool", "content": "x" * 50, "tool_calls": None},
    ]
    cfg = context.ContextConfig(max_turns=10, max_chars=10_000, max_tool_chars=12, emergency_turns=8)

    messages = context.build_messages(_system(), turns, config=cfg)
    tool_msg = messages[2]

    assert tool_msg["role"] == "tool"
    assert len(tool_msg["content"]) <= 12


def test_prefers_dropping_tool_messages_before_user_or_assistant():
    turns = [
        {"role": "user", "content": "old user", "tool_calls": None},
        {"role": "tool", "content": "tool output", "tool_calls": None},
        {"role": "assistant", "content": "old assistant", "tool_calls": None},
        {"role": "user", "content": "latest user", "tool_calls": None},
    ]
    cfg = context.ContextConfig(max_turns=3, max_chars=10_000, max_tool_chars=8_000, emergency_turns=8)

    messages = context.build_messages(_system(), turns, config=cfg)
    history = messages[1:]

    assert [m["role"] for m in history] == ["user", "assistant", "user"]
    assert history[-1]["content"] == "latest user"


def test_emergency_mode_keeps_latest_user_and_adds_note():
    turns = [
        {"role": "user", "content": "latest request", "tool_calls": None},
        {"role": "assistant", "content": "planning", "tool_calls": None},
        {"role": "tool", "content": "result", "tool_calls": None},
        {"role": "assistant", "content": "more", "tool_calls": None},
    ]
    cfg = context.ContextConfig(max_turns=60, max_chars=10_000, max_tool_chars=8_000, emergency_turns=2)

    messages = context.build_messages(_system(), turns, config=cfg, emergency=True)

    assert messages[1]["role"] == "system"
    assert "request-size guard" in messages[1]["content"]
    history = messages[2:]
    assert any(m["role"] == "user" and m["content"] == "latest request" for m in history)
    assert len(history) <= 3  # emergency_turns + preserved latest user


def test_tiny_summary_skips_latest_user_request():
    turns = [
        {"role": "user", "content": "first request", "tool_calls": None},
        {"role": "assistant", "content": "first answer", "tool_calls": None},
        {"role": "user", "content": "latest request", "tool_calls": None},
    ]

    summary = context.build_tiny_summary(turns)

    assert "latest request" not in summary
    assert "first answer" in summary
