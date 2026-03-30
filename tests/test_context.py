"""Tests for context management, compaction, and payload shaping."""

import json

from puget import context


def _system() -> dict:
    return {"role": "system", "content": "You are helpful."}


# ---------------------------------------------------------------------------
# Context config
# ---------------------------------------------------------------------------

class TestConfigForContextWindow:
    def test_invalid_uses_default(self):
        assert context.config_for_context_window(None) == context.DEFAULT_CONFIG

    def test_large_window_preserves_defaults(self):
        cfg = context.config_for_context_window(262_144)
        assert cfg.context_window == 262_144
        assert cfg.reserve_tokens == context.DEFAULT_CONFIG.reserve_tokens
        assert cfg.keep_recent_tokens == context.DEFAULT_CONFIG.keep_recent_tokens

    def test_small_window_scales_down(self):
        cfg = context.config_for_context_window(4_096)
        assert cfg.context_window == 4_096
        assert cfg.reserve_tokens < context.DEFAULT_CONFIG.reserve_tokens
        assert cfg.keep_recent_tokens < context.DEFAULT_CONFIG.keep_recent_tokens


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

class TestTokenEstimation:
    def test_estimate_tokens_basic(self):
        assert context.estimate_tokens("hello world") >= 1
        # 11 chars // 4 = 2
        assert context.estimate_tokens("hello world") == 2

    def test_estimate_tokens_empty(self):
        assert context.estimate_tokens("") == 1  # min 1

    def test_estimate_turn_tokens_user(self):
        turn = {"role": "user", "content": "x" * 400}
        assert context.estimate_turn_tokens(turn) == 100

    def test_estimate_turn_tokens_with_tool_calls(self):
        tc = json.dumps([{"function": {"name": "bash", "arguments": {"command": "ls"}}}])
        turn = {"role": "assistant", "content": "ok", "tool_calls": tc}
        tokens = context.estimate_turn_tokens(turn)
        assert tokens > 1

    def test_estimate_context_tokens(self):
        turns = [
            {"role": "user", "content": "x" * 400},
            {"role": "assistant", "content": "y" * 400},
        ]
        total = context.estimate_context_tokens("system prompt", turns)
        # system(~3) + user(100) + assistant(100)
        assert total > 200


# ---------------------------------------------------------------------------
# Compaction trigger
# ---------------------------------------------------------------------------

class TestShouldCompact:
    def test_below_threshold(self):
        cfg = context.ContextConfig(context_window=1000, reserve_tokens=100)
        assert not context.should_compact(800, cfg)

    def test_above_threshold(self):
        cfg = context.ContextConfig(context_window=1000, reserve_tokens=100)
        assert context.should_compact(950, cfg)

    def test_exactly_at_threshold(self):
        cfg = context.ContextConfig(context_window=1000, reserve_tokens=100)
        # 900 is not > 900
        assert not context.should_compact(900, cfg)


# ---------------------------------------------------------------------------
# Cut point
# ---------------------------------------------------------------------------

class TestFindCutPoint:
    def test_no_cut_needed(self):
        turns = [
            {"role": "user", "content": "x" * 40},
            {"role": "assistant", "content": "y" * 40},
        ]
        cfg = context.ContextConfig(keep_recent_tokens=1000)
        assert context.find_cut_point(turns, cfg) == 0

    def test_cuts_at_user_boundary(self):
        turns = [
            {"role": "user", "content": "x" * 400},      # 100 tokens
            {"role": "assistant", "content": "y" * 400},  # 100 tokens
            {"role": "tool", "content": "z" * 400},       # 100 tokens
            {"role": "user", "content": "a" * 400},       # 100 tokens
            {"role": "assistant", "content": "b" * 400},  # 100 tokens
        ]
        # keep_recent_tokens=250 means we walk back ~250 tokens from end
        # Turns 4(100) + 3(100) + 2(100) = 300 >= 250 → cut near index 2
        # Walk forward from 2 to find user → index 3
        cfg = context.ContextConfig(keep_recent_tokens=250)
        cut = context.find_cut_point(turns, cfg)
        assert cut == 3  # First kept turn is the second user message

    def test_empty_turns(self):
        assert context.find_cut_point([], context.DEFAULT_CONFIG) == 0


# ---------------------------------------------------------------------------
# File operation tracking
# ---------------------------------------------------------------------------

class TestFileOperations:
    def test_extract_read_and_write(self):
        turns = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": json.dumps([
                    {"function": {"name": "read", "arguments": {"path": "foo.py"}}},
                    {"function": {"name": "write", "arguments": {"path": "bar.py", "content": "x"}}},
                    {"function": {"name": "read", "arguments": {"path": "bar.py"}}},
                ]),
            },
        ]
        read_only, modified = context.extract_file_operations(turns)
        assert read_only == ["foo.py"]  # bar.py was also written
        assert modified == ["bar.py"]

    def test_extract_edit(self):
        turns = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": json.dumps([
                    {"function": {"name": "edit", "arguments": {"path": "main.py"}}},
                ]),
            },
        ]
        read_only, modified = context.extract_file_operations(turns)
        assert read_only == []
        assert modified == ["main.py"]

    def test_format_file_operations(self):
        result = context.format_file_operations(["a.py"], ["b.py"])
        assert "<read-files>" in result
        assert "a.py" in result
        assert "<modified-files>" in result
        assert "b.py" in result

    def test_format_empty(self):
        assert context.format_file_operations([], []) == ""


# ---------------------------------------------------------------------------
# Conversation serialization
# ---------------------------------------------------------------------------

class TestSerializeConversation:
    def test_basic_conversation(self):
        turns = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        text = context.serialize_conversation(turns)
        assert "[User]: hello" in text
        assert "[Assistant]: hi there" in text

    def test_tool_calls_serialized(self):
        turns = [
            {
                "role": "assistant",
                "content": "checking",
                "tool_calls": json.dumps([
                    {"function": {"name": "bash", "arguments": {"command": "ls"}}},
                ]),
            },
        ]
        text = context.serialize_conversation(turns)
        assert "[Tool calls]:" in text
        assert "bash(" in text

    def test_tool_results_truncated(self):
        turns = [
            {"role": "tool", "content": "x" * 5000},
        ]
        text = context.serialize_conversation(turns)
        assert "[Tool result]:" in text
        assert len(text) < 5000  # Should be truncated


# ---------------------------------------------------------------------------
# Compaction preparation
# ---------------------------------------------------------------------------

class TestPrepareCompaction:
    def test_basic_preparation(self):
        turns = [
            {"id": 1, "role": "user", "content": "x" * 400},
            {"id": 2, "role": "assistant", "content": "y" * 400},
            {"id": 3, "role": "user", "content": "a" * 400},
            {"id": 4, "role": "assistant", "content": "b" * 400},
            {"id": 5, "role": "user", "content": "c" * 400},
            {"id": 6, "role": "assistant", "content": "d" * 400},
        ]
        cfg = context.ContextConfig(keep_recent_tokens=250)
        prep = context.prepare_compaction(turns, "system", None, cfg)
        assert prep is not None
        assert prep.first_kept_turn_id > 1
        assert len(prep.messages_to_summarize) > 0

    def test_returns_none_for_tiny_context(self):
        turns = [
            {"id": 1, "role": "user", "content": "hi"},
            {"id": 2, "role": "assistant", "content": "hello"},
        ]
        cfg = context.ContextConfig(keep_recent_tokens=10000)
        prep = context.prepare_compaction(turns, "system", None, cfg)
        assert prep is None

    def test_iterative_with_previous_compaction(self):
        turns = [
            {"id": 1, "role": "user", "content": "x" * 400},
            {"id": 2, "role": "assistant", "content": "y" * 400},
            {"id": 3, "role": "user", "content": "a" * 400},
            {"id": 4, "role": "assistant", "content": "b" * 400},
            {"id": 5, "role": "user", "content": "c" * 400},
            {"id": 6, "role": "assistant", "content": "d" * 400},
        ]
        prev_compaction = {
            "first_kept_turn_id": 3,
            "summary": "Previous summary here.",
            "details_json": json.dumps({
                "read_files": ["old.py"],
                "modified_files": ["changed.py"],
            }),
        }
        cfg = context.ContextConfig(keep_recent_tokens=250)
        prep = context.prepare_compaction(turns, "system", prev_compaction, cfg)
        assert prep is not None
        assert prep.previous_summary == "Previous summary here."
        # Previous file ops should be merged in.
        assert "changed.py" in prep.modified_files

    def test_returns_none_when_empty(self):
        assert context.prepare_compaction([], "sys", None) is None


# ---------------------------------------------------------------------------
# Summarization message building
# ---------------------------------------------------------------------------

class TestBuildSummarizationMessages:
    def test_initial_compaction(self):
        prep = context.CompactionPreparation(
            messages_to_summarize=[
                {"role": "user", "content": "do stuff"},
                {"role": "assistant", "content": "done"},
            ],
            first_kept_turn_id=3,
            tokens_before=1000,
            previous_summary=None,
            read_files=[],
            modified_files=[],
        )
        msgs = context.build_summarization_messages(prep)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "<conversation>" in msgs[1]["content"]
        assert "EXACT format" in msgs[1]["content"]

    def test_iterative_compaction_includes_previous(self):
        prep = context.CompactionPreparation(
            messages_to_summarize=[
                {"role": "user", "content": "more stuff"},
            ],
            first_kept_turn_id=5,
            tokens_before=2000,
            previous_summary="## Goal\nPrevious goal",
            read_files=[],
            modified_files=[],
        )
        msgs = context.build_summarization_messages(prep)
        assert "<previous-summary>" in msgs[1]["content"]
        assert "Previous goal" in msgs[1]["content"]
        assert "UPDATE" in msgs[1]["content"]


# ---------------------------------------------------------------------------
# Build messages for model calls
# ---------------------------------------------------------------------------

class TestBuildMessages:
    def test_basic_no_compaction(self):
        turns = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        cfg = context.ContextConfig(context_window=100000)
        messages = context.build_messages(_system(), turns, config=cfg)
        assert messages[0]["role"] == "system"
        assert messages[1] == {"role": "user", "content": "hello"}
        assert len(messages) == 3

    def test_with_compaction_summary(self):
        turns = [
            {"role": "user", "content": "latest question"},
        ]
        cfg = context.ContextConfig(context_window=100000)
        messages = context.build_messages(
            _system(), turns, config=cfg,
            compaction_summary="## Goal\nSome goal",
        )
        # system + summary + user
        assert len(messages) == 3
        assert "summary of the conversation" in messages[1]["content"]
        assert "Some goal" in messages[1]["content"]

    def test_truncates_tool_output(self):
        turns = [
            {"role": "user", "content": "run it"},
            {"role": "tool", "content": "x" * 50000},
        ]
        cfg = context.ContextConfig(
            context_window=100000,
            max_tool_result_chars=100,
        )
        messages = context.build_messages(_system(), turns, config=cfg)
        tool_msg = [m for m in messages if m["role"] == "tool"][0]
        assert len(tool_msg["content"]) <= 100

    def test_no_truncation_note_when_nothing_dropped(self):
        turns = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        cfg = context.ContextConfig(context_window=100000)
        messages = context.build_messages(_system(), turns, config=cfg)
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert len(messages) == 3

    def test_hard_budget_drops_oldest_when_huge(self):
        # Create enough turns to blow past a tiny context window.
        # Each turn is ~100 tokens (400 chars / 4), 20 turns = ~2000 tokens.
        turns = [
            {"role": "user", "content": "x" * 400}
            for _ in range(20)
        ]
        cfg = context.ContextConfig(
            context_window=500,   # ~500 tokens budget
            reserve_tokens=100,   # budget = 400 tokens
        )
        messages = context.build_messages(_system(), turns, config=cfg)
        # Should have dropped some turns and added a truncation note.
        has_note = any(
            m.get("role") == "system" and "truncated" in m.get("content", "")
            for m in messages
        )
        assert has_note
        assert len(messages) < 22  # system + 20 turns + note would be 22


# ---------------------------------------------------------------------------
# Emergency mode
# ---------------------------------------------------------------------------

class TestEmergencyMode:
    def test_emergency_keeps_latest_user_and_adds_note(self):
        turns = [
            {"role": "user", "content": "latest request"},
            {"role": "assistant", "content": "planning"},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "more"},
        ]
        cfg = context.ContextConfig(emergency_turns=2, context_window=100000)
        messages = context.build_messages(
            _system(), turns, config=cfg, emergency=True,
        )
        assert messages[1]["role"] == "system"
        assert "request-size guard" in messages[1]["content"]
        history = messages[2:]
        assert any(
            m["role"] == "user" and m["content"] == "latest request"
            for m in history
        )
        assert len(history) <= 3


# ---------------------------------------------------------------------------
# Fork helpers
# ---------------------------------------------------------------------------

class TestForkHelpers:
    def test_tiny_summary_skips_latest_user_request(self):
        turns = [
            {"role": "user", "content": "first request"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "latest request"},
        ]
        summary = context.build_tiny_summary(turns)
        assert "latest request" not in summary
        assert "first answer" in summary

    def test_latest_user_request(self):
        turns = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
        assert context.latest_user_request(turns) == "second"

    def test_fork_preamble(self):
        preamble = context.build_fork_preamble(42, "some summary")
        assert "wave #42" in preamble
        assert "some summary" in preamble
