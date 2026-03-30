"""Context management and compaction for puget.

This module implements LLM-powered context compaction inspired by pi's
approach. Instead of silently dropping old messages, puget generates
structured summaries via the model before trimming context.

Design approach (from pi):
- Proactive compaction when estimated tokens approach context window limit.
- Structured summaries preserving goals, progress, decisions, and file ops.
- Iterative summaries: previous summary is fed into the next compaction.
- Cumulative file operation tracking across compactions.
- Cut at turn boundaries (user messages) to keep conversation coherent.
- Emergency mode and auto-fork as fallback safety nets.

Key functions:
- should_compact()               Check if compaction is needed.
- prepare_compaction()           Find cut point, extract messages to summarize.
- build_summarization_messages() Build the prompt for the summarization call.
- finalize_summary()             Attach file operation lists to the summary.
- build_messages()               Build model messages with optional compaction.
"""

import json
from dataclasses import dataclass, replace
from typing import Any

from puget.output import err_console


# ---------------------------------------------------------------------------
# Summarization prompts (adapted from pi)
# ---------------------------------------------------------------------------

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Your task is to read a "
    "conversation between a user and an AI coding assistant, then produce "
    "a structured summary following the exact format specified.\n\n"
    "Do NOT continue the conversation. Do NOT respond to any questions "
    "in the conversation. ONLY output the structured summary."
)

SUMMARIZATION_PROMPT = """\
The messages above are a conversation to summarize. Create a structured \
context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, \
and error messages."""

UPDATE_SUMMARIZATION_PROMPT = """\
The messages above are NEW conversation messages to incorporate into the \
existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" \
when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, \
and error messages."""


# ---------------------------------------------------------------------------
# Context notes injected into model messages
# ---------------------------------------------------------------------------

COMPACTION_CONTEXT_NOTE = (
    "The following is a structured summary of the conversation so far:"
)
EMERGENCY_CONTEXT_NOTE = "Prior context truncated due to request-size guard."
TRUNCATION_NOTE = (
    "Note: {dropped} older turns were truncated from this conversation "
    "to stay within context limits. You may be missing prior context."
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContextConfig:
    """Token-aware limits for context management and compaction.

    Compaction triggers when estimated context tokens exceed
    ``context_window - reserve_tokens``.  The most recent
    ``keep_recent_tokens`` worth of turns are preserved; everything
    older is summarized.

    Emergency mode and auto-fork remain as safety nets for when
    compaction alone isn't enough (e.g. a single turn is enormous).
    """

    context_window: int = 128_000
    reserve_tokens: int = 16_384
    keep_recent_tokens: int = 20_000
    max_tool_result_chars: int = 8_000
    emergency_turns: int = 8
    max_400_retries: int = 1


DEFAULT_CONFIG = ContextConfig()


def config_for_context_window(context_window: int | None) -> ContextConfig:
    """Return DEFAULT_CONFIG adjusted for a model-specific context window."""
    if not isinstance(context_window, int) or context_window <= 0:
        return DEFAULT_CONFIG

    if context_window >= DEFAULT_CONFIG.context_window:
        return replace(DEFAULT_CONFIG, context_window=context_window)

    if context_window < 8_192:
        reserve = max(256, context_window // 8)
        keep_recent = max(1_024, context_window // 4)
    else:
        reserve = min(DEFAULT_CONFIG.reserve_tokens, max(1_024, context_window // 8))
        keep_recent = min(DEFAULT_CONFIG.keep_recent_tokens, max(2_048, context_window // 6))

    budget = max(1_024, context_window - reserve)
    keep_recent = min(keep_recent, max(1_024, budget // 2))

    return replace(
        DEFAULT_CONFIG,
        context_window=context_window,
        reserve_tokens=reserve,
        keep_recent_tokens=keep_recent,
    )


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Estimate token count using the chars/4 heuristic.

    Conservative (tends to overestimate), which is fine for compaction
    triggers — better to compact a little early than too late.
    """
    return max(1, len(text) // 4)


def estimate_turn_tokens(turn: dict[str, Any]) -> int:
    """Estimate tokens for a single stored turn dict."""
    chars = len(turn.get("content") or "")
    tc = turn.get("tool_calls")
    if tc:
        chars += len(tc) if isinstance(tc, str) else len(json.dumps(tc, ensure_ascii=False))
    return max(1, chars // 4)


def estimate_context_tokens(
    system_content: str,
    turns: list[dict[str, Any]],
    compaction_summary: str | None = None,
) -> int:
    """Estimate total context tokens from system prompt, turns, and summary."""
    total = estimate_tokens(system_content)
    if compaction_summary:
        total += estimate_tokens(compaction_summary)
    for t in turns:
        total += estimate_turn_tokens(t)
    return total


# ---------------------------------------------------------------------------
# Compaction trigger
# ---------------------------------------------------------------------------

def should_compact(
    estimated_tokens: int,
    config: ContextConfig = DEFAULT_CONFIG,
) -> bool:
    """Return True if estimated tokens exceed the compaction threshold."""
    return estimated_tokens > config.context_window - config.reserve_tokens


# ---------------------------------------------------------------------------
# Cut point finding
# ---------------------------------------------------------------------------

def find_cut_point(
    turns: list[dict[str, Any]],
    config: ContextConfig = DEFAULT_CONFIG,
) -> int:
    """Find the index of the first turn to *keep* (not summarize).

    Walks backwards from the newest turn, accumulating estimated tokens.
    When ``keep_recent_tokens`` is exceeded, finds the nearest user
    message at or after that point to cut at a clean turn boundary.

    Returns 0 if the budget is never exceeded (nothing to summarize).
    """
    if not turns:
        return 0

    accumulated = 0
    for i in range(len(turns) - 1, -1, -1):
        accumulated += estimate_turn_tokens(turns[i])
        if accumulated >= config.keep_recent_tokens:
            # Walk forward to the nearest user-message boundary.
            for j in range(i, len(turns)):
                if turns[j].get("role") == "user":
                    return j
            # No user message found after — cut here as a fallback.
            return i

    # Budget never exceeded.
    return 0


# ---------------------------------------------------------------------------
# File operation tracking
# ---------------------------------------------------------------------------

def extract_file_operations(
    turns: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Extract cumulative read/modified file lists from assistant tool calls.

    Returns (read_only_files, modified_files) where read_only excludes
    any file that was also written or edited.
    """
    read_files: set[str] = set()
    modified_files: set[str] = set()

    for turn in turns:
        if turn.get("role") != "assistant":
            continue
        tool_calls = _parse_tool_calls(turn.get("tool_calls"))
        if not tool_calls:
            continue
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    continue
            path = args.get("path") if isinstance(args, dict) else None
            if not path:
                continue
            if name == "read":
                read_files.add(path)
            elif name in ("write", "edit"):
                modified_files.add(path)

    read_only = sorted(read_files - modified_files)
    return read_only, sorted(modified_files)


def format_file_operations(
    read_files: list[str],
    modified_files: list[str],
) -> str:
    """Format file operation lists as XML tags appended to summaries."""
    sections: list[str] = []
    if read_files:
        sections.append("<read-files>\n" + "\n".join(read_files) + "\n</read-files>")
    if modified_files:
        sections.append("<modified-files>\n" + "\n".join(modified_files) + "\n</modified-files>")
    if not sections:
        return ""
    return "\n\n" + "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Conversation serialization (for summarization prompts)
# ---------------------------------------------------------------------------

TOOL_RESULT_MAX_CHARS = 2000


def serialize_conversation(turns: list[dict[str, Any]]) -> str:
    """Serialize turns to labelled text blocks for summarization.

    The output format prevents the summarization model from treating
    it as a live conversation to continue.  Tool results are truncated
    to ``TOOL_RESULT_MAX_CHARS`` since full output is rarely needed
    for a good summary.
    """
    parts: list[str] = []
    for t in turns:
        role = t.get("role", "")
        content = (t.get("content") or "").strip()

        if role == "user":
            if content:
                parts.append(f"[User]: {content}")

        elif role == "assistant":
            if content:
                parts.append(f"[Assistant]: {content}")
            tool_calls = _parse_tool_calls(t.get("tool_calls"))
            if tool_calls:
                calls: list[str] = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            pass
                    if isinstance(args, dict):
                        args_str = ", ".join(
                            f"{k}={json.dumps(v)}" for k, v in args.items()
                        )
                    else:
                        args_str = str(args)
                    calls.append(f"{name}({args_str})")
                parts.append(f"[Tool calls]: {'; '.join(calls)}")

        elif role == "tool":
            if content:
                truncated = _truncate(content, TOOL_RESULT_MAX_CHARS)
                parts.append(f"[Tool result]: {truncated}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Compaction preparation
# ---------------------------------------------------------------------------

@dataclass
class CompactionPreparation:
    """Pre-calculated data needed to generate a compaction summary."""

    messages_to_summarize: list[dict[str, Any]]
    first_kept_turn_id: int
    tokens_before: int
    previous_summary: str | None
    read_files: list[str]
    modified_files: list[str]


def prepare_compaction(
    turns: list[dict[str, Any]],
    system_content: str,
    compaction: dict[str, Any] | None,
    config: ContextConfig = DEFAULT_CONFIG,
) -> CompactionPreparation | None:
    """Prepare everything needed for a compaction summary.

    Finds the cut point, extracts messages to summarize, collects file
    operations (merging with the previous compaction's tracked files),
    and estimates pre-compaction token count.

    Returns None if compaction isn't possible or useful (too few turns,
    nothing to summarize).

    Args:
        turns: All turns in the wave (with ``id`` keys from the DB).
        system_content: The system prompt text (for token estimation).
        compaction: The latest compaction dict, or None.
        config: Context configuration.
    """
    if not turns:
        return None

    # Determine the starting boundary — only consider turns that haven't
    # already been summarized by a previous compaction.
    if compaction:
        first_kept_id = compaction["first_kept_turn_id"]
        start_idx = 0
        for i, t in enumerate(turns):
            if t.get("id") == first_kept_id:
                start_idx = i
                break
        relevant_turns = turns[start_idx:]
        previous_summary = compaction.get("summary")
        prev_details = json.loads(compaction.get("details_json") or "{}")
        prev_read: list[str] = prev_details.get("read_files", [])
        prev_modified: list[str] = prev_details.get("modified_files", [])
    else:
        relevant_turns = turns
        previous_summary = None
        prev_read = []
        prev_modified = []

    if len(relevant_turns) <= 2:
        return None

    cut_idx = find_cut_point(relevant_turns, config)
    if cut_idx == 0:
        return None  # Nothing to summarize.

    messages_to_summarize = relevant_turns[:cut_idx]
    kept_turns = relevant_turns[cut_idx:]

    if not kept_turns:
        return None

    first_kept_turn_id = kept_turns[0]["id"]

    tokens_before = estimate_context_tokens(
        system_content, relevant_turns, previous_summary,
    )

    # Extract file operations from the messages being summarized and
    # merge with the previous compaction's cumulative file lists.
    new_read, new_modified = extract_file_operations(messages_to_summarize)
    all_modified = sorted(set(prev_modified) | set(new_modified))
    all_modified_set = set(all_modified)
    all_read = sorted(
        (set(prev_read) | set(new_read)) - all_modified_set
    )

    return CompactionPreparation(
        messages_to_summarize=messages_to_summarize,
        first_kept_turn_id=first_kept_turn_id,
        tokens_before=tokens_before,
        previous_summary=previous_summary,
        read_files=all_read,
        modified_files=all_modified,
    )


# ---------------------------------------------------------------------------
# Summarization message building
# ---------------------------------------------------------------------------

def build_summarization_messages(
    preparation: CompactionPreparation,
) -> list[dict[str, Any]]:
    """Build the message list for the summarization model call.

    Wraps the serialized conversation (and optional previous summary)
    in XML tags, then appends the appropriate summarization prompt.
    """
    conversation_text = serialize_conversation(
        preparation.messages_to_summarize,
    )

    prompt_parts = [f"<conversation>\n{conversation_text}\n</conversation>"]

    if preparation.previous_summary:
        prompt_parts.append(
            f"<previous-summary>\n{preparation.previous_summary}"
            f"\n</previous-summary>"
        )
        prompt_parts.append(UPDATE_SUMMARIZATION_PROMPT)
    else:
        prompt_parts.append(SUMMARIZATION_PROMPT)

    return [
        {"role": "system", "content": SUMMARIZATION_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(prompt_parts)},
    ]


def finalize_summary(
    raw_summary: str,
    preparation: CompactionPreparation,
) -> str:
    """Append file-operation XML tags to the raw model summary."""
    return raw_summary + format_file_operations(
        preparation.read_files,
        preparation.modified_files,
    )


# ---------------------------------------------------------------------------
# Build messages for model calls
# ---------------------------------------------------------------------------

def build_messages(
    system: dict[str, Any],
    turns: list[dict[str, Any]],
    *,
    config: ContextConfig = DEFAULT_CONFIG,
    compaction_summary: str | None = None,
    emergency: bool = False,
) -> list[dict[str, Any]]:
    """Build the message list sent to model.chat().

    Args:
        system: The leading system message.
        turns: Wave turns from db.get_turns() or db.get_turns_from().
        config: Context budget configuration.
        compaction_summary: If present, injected as a system message
            right after the main system prompt.
        emergency: When True, build a minimal payload (last few turns
            plus the latest user request).

    Returns:
        Messages ready to send to model.chat().
    """
    history = turns_to_messages(turns)

    if emergency:
        history = _build_emergency_history(history, config)
        return [
            system,
            {"role": "system", "content": EMERGENCY_CONTEXT_NOTE},
            *history,
        ]

    # Truncate tool-result content to keep payloads reasonable.
    history = [
        _truncate_tool_message(m, config.max_tool_result_chars)
        for m in history
    ]

    # Safety-net hard budget — should rarely trigger when compaction
    # is active, but protects against edge cases.
    history, dropped = _apply_hard_budget(history, config)

    prefix: list[dict[str, Any]] = [system]
    if compaction_summary:
        prefix.append({
            "role": "system",
            "content": f"{COMPACTION_CONTEXT_NOTE}\n\n{compaction_summary}",
        })
    if dropped:
        prefix.append({
            "role": "system",
            "content": TRUNCATION_NOTE.format(dropped=dropped),
        })
        err_console.print(
            f"[yellow]Warning: {dropped} older turn(s) truncated to fit context limits[/yellow]"
        )

    return [*prefix, *history]


def turns_to_messages(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert stored DB turns into model API message dicts."""
    messages: list[dict[str, Any]] = []
    for t in turns:
        role = t.get("role", "")
        content = t.get("content") or ""
        raw_tool_calls = t.get("tool_calls")

        if role == "assistant" and raw_tool_calls:
            tool_calls = _parse_tool_calls(raw_tool_calls)
            if tool_calls is not None:
                messages.append({
                    "role": role,
                    "content": content,
                    "tool_calls": tool_calls,
                })
                continue

        messages.append({"role": role, "content": content})

    return messages


# ---------------------------------------------------------------------------
# Fork / emergency helpers (kept from the original design)
# ---------------------------------------------------------------------------

def latest_user_request(turns: list[dict[str, Any]]) -> str | None:
    """Return the latest non-empty user message from turns."""
    for t in reversed(turns):
        if t.get("role") != "user":
            continue
        content = (t.get("content") or "").strip()
        if content:
            return content
    return None


def build_tiny_summary(
    turns: list[dict[str, Any]],
    *,
    max_items: int = 3,
    max_chars: int = 500,
) -> str:
    """Build a tiny extractive summary from recent non-empty turns.

    Excludes the latest user request to avoid duplicating it when carrying
    both summary + latest request into a forked wave.
    """
    snippets: list[str] = []
    skipped_latest_user = False

    for t in reversed(turns):
        role = t.get("role")
        if role not in {"user", "assistant", "tool"}:
            continue

        content = (t.get("content") or "").strip()
        if not content:
            continue

        if role == "user" and not skipped_latest_user:
            skipped_latest_user = True
            continue

        single_line = " ".join(content.split())
        snippets.append(f"{role}: {_truncate(single_line, 160)}")
        if len(snippets) >= max_items:
            break

    if not snippets:
        return ""

    snippets.reverse()
    return _truncate(" | ".join(snippets), max_chars)


def build_fork_preamble(previous_wave_id: int, summary: str) -> str:
    """Build a compact context note for a wave fork after repeated 400s."""
    lines = [
        f"Context resumed from previous wave #{previous_wave_id}.",
        "Prior context was truncated due to request-size guard.",
    ]
    if summary:
        lines.append(f"Recent summary: {summary}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_emergency_history(
    history: list[dict[str, Any]],
    config: ContextConfig,
) -> list[dict[str, Any]]:
    """Build reduced history for one emergency retry after a 400."""
    if not history:
        return []

    normalized = [
        _truncate_tool_message(m, config.max_tool_result_chars)
        for m in history
    ]

    latest_user_idx = _latest_user_index(normalized)
    start = max(0, len(normalized) - max(1, config.emergency_turns))

    reduced = [dict(m) for m in normalized[start:]]
    if latest_user_idx is not None and latest_user_idx < start:
        reduced.insert(0, dict(normalized[latest_user_idx]))

    # Apply the token-based hard budget on the reduced set.
    reduced, _ = _apply_hard_budget(reduced, config)
    return reduced


def _apply_hard_budget(
    history: list[dict[str, Any]],
    config: ContextConfig,
) -> tuple[list[dict[str, Any]], int]:
    """Safety-net: drop oldest turns if estimated tokens exceed budget.

    With compaction active this should rarely trigger. It exists to
    prevent runaway payloads when compaction hasn't fired yet or when
    a single turn is unexpectedly large.
    """
    messages = list(history)
    original_count = len(messages)
    protected_user_idx = _latest_user_index(messages)
    budget = config.context_window - config.reserve_tokens

    while _estimate_history_tokens(messages) > budget:
        idx = _pick_drop_index(messages, protected_user_idx)
        if idx is None:
            break
        del messages[idx]
        if protected_user_idx is not None:
            if idx < protected_user_idx:
                protected_user_idx -= 1
            elif idx == protected_user_idx:
                protected_user_idx = None

    return messages, original_count - len(messages)


def _pick_drop_index(
    messages: list[dict[str, Any]],
    protected_user_idx: int | None,
) -> int | None:
    """Pick the oldest droppable message, preferring tool turns first."""
    idx = _oldest_droppable(messages, protected_user_idx, preferred_role="tool")
    if idx is not None:
        return idx
    return _oldest_droppable(messages, protected_user_idx, preferred_role=None)


def _oldest_droppable(
    messages: list[dict[str, Any]],
    protected_user_idx: int | None,
    preferred_role: str | None,
) -> int | None:
    for idx, msg in enumerate(messages):
        if (
            protected_user_idx is not None
            and idx == protected_user_idx
            and msg.get("role") == "user"
        ):
            continue
        if preferred_role is not None and msg.get("role") != preferred_role:
            continue
        return idx
    return None


def _latest_user_index(messages: list[dict[str, Any]]) -> int | None:
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "user":
            return idx
    return None


def _estimate_message_tokens(msg: dict[str, Any]) -> int:
    """Estimate tokens for a single API message dict."""
    chars = len(msg.get("content") or "")
    tc = msg.get("tool_calls")
    if tc is not None:
        if isinstance(tc, list):
            chars += len(json.dumps(tc, ensure_ascii=False))
        else:
            chars += len(str(tc))
    return max(1, chars // 4)


def _estimate_history_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(_estimate_message_tokens(m) for m in messages)


def _truncate_tool_message(
    message: dict[str, Any],
    max_chars: int,
) -> dict[str, Any]:
    if message.get("role") != "tool":
        return dict(message)
    out = dict(message)
    out["content"] = _truncate(message.get("content") or "", max_chars)
    return out


def _truncate(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    suffix = " \u2026[truncated]"
    if limit <= len(suffix):
        return text[:limit]
    return text[: limit - len(suffix)] + suffix


def _parse_tool_calls(value: Any) -> list[dict[str, Any]] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, list) else None
    return None
