"""Context management and payload budgeting for puget.

This module keeps model request payloads within conservative limits and
provides recovery helpers for request-size failures (HTTP 400).

Design goals:
- Hard preflight budget before every model call.
- Prefer dropping/truncating tool output over user/assistant turns.
- Always preserve the latest user request when possible.
- Provide an emergency context mode for a single reduced retry.
- Provide tiny carry-over helpers for auto-forking to a new wave.
"""

import json
from dataclasses import dataclass
from typing import Any


EMERGENCY_CONTEXT_NOTE = "Prior context truncated due to request-size guard."
TRUNCATION_NOTE = "Note: {dropped} older turns were truncated from this conversation to stay within context limits. You may be missing prior context."


@dataclass(frozen=True)
class ContextConfig:
    """Hard limits for model payload construction."""

    max_turns: int = 60
    max_chars: int = 120_000
    max_tool_chars: int = 8_000
    emergency_turns: int = 8
    max_400_retries: int = 1


DEFAULT_CONTEXT_CONFIG = ContextConfig()


def build_messages(
    system: dict[str, Any],
    turns: list[dict[str, Any]],
    *,
    config: ContextConfig = DEFAULT_CONTEXT_CONFIG,
    emergency: bool = False,
) -> list[dict[str, Any]]:
    """Build model messages from persisted turns with budget enforcement.

    Args:
        system: The leading system message.
        turns: Wave turns from db.get_turns().
        config: Context budget configuration.
        emergency: When True, build a reduced context payload and include
                   a short truncation note.

    Returns:
        Messages ready to send to model.chat().
    """
    history = turns_to_messages(turns)
    if emergency:
        history = _build_emergency_history(history, config)
        return [system, {"role": "system", "content": EMERGENCY_CONTEXT_NOTE}, *history]

    history, dropped = _apply_hard_budget(history, config)
    if dropped:
        note = {"role": "system", "content": TRUNCATION_NOTE.format(dropped=dropped)}
        return [system, note, *history]
    return [system, *history]


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
                messages.append({"role": role, "content": content, "tool_calls": tool_calls})
                continue

        messages.append({"role": role, "content": content})

    return messages


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


def _build_emergency_history(
    history: list[dict[str, Any]],
    config: ContextConfig,
) -> list[dict[str, Any]]:
    """Build reduced history for one emergency retry after a 400."""
    if not history:
        return []

    normalized = [_truncate_tool_message(m, config.max_tool_chars) for m in history]

    latest_user_idx = _latest_user_index(normalized)
    start = max(0, len(normalized) - max(1, config.emergency_turns))

    reduced = [dict(m) for m in normalized[start:]]
    if latest_user_idx is not None and latest_user_idx < start:
        reduced.insert(0, dict(normalized[latest_user_idx]))

    # Apply hard budget again with a tighter turn cap.
    emergency_cfg = ContextConfig(
        max_turns=max(1, config.emergency_turns + 1),
        max_chars=config.max_chars,
        max_tool_chars=config.max_tool_chars,
        emergency_turns=config.emergency_turns,
        max_400_retries=config.max_400_retries,
    )
    reduced, _ = _apply_hard_budget(reduced, emergency_cfg)
    return reduced


def _apply_hard_budget(
    history: list[dict[str, Any]],
    config: ContextConfig,
) -> tuple[list[dict[str, Any]], int]:
    """Apply tool truncation, turn cap, and total-char cap to history.

    Returns:
        A tuple of (messages, dropped_count) where dropped_count is the
        number of turns removed from history.
    """
    messages = [_truncate_tool_message(m, config.max_tool_chars) for m in history]
    original_count = len(messages)

    protected_user_idx = _latest_user_index(messages)

    # Hard cap by number of turns.
    while len(messages) > config.max_turns:
        idx = _pick_drop_index(messages, protected_user_idx)
        if idx is None:
            break
        del messages[idx]
        if protected_user_idx is not None:
            if idx < protected_user_idx:
                protected_user_idx -= 1
            elif idx == protected_user_idx:
                protected_user_idx = None

    # Hard cap by total payload chars.
    while _total_chars(messages) > config.max_chars:
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
        if protected_user_idx is not None and idx == protected_user_idx and msg.get("role") == "user":
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


def _truncate_tool_message(message: dict[str, Any], max_chars: int) -> dict[str, Any]:
    if message.get("role") != "tool":
        return dict(message)

    out = dict(message)
    out["content"] = _truncate(message.get("content") or "", max_chars)
    return out


def _message_chars(message: dict[str, Any]) -> int:
    total = len(message.get("role") or "")
    total += len(message.get("content") or "")

    tool_calls = message.get("tool_calls")
    if tool_calls is not None:
        total += len(json.dumps(tool_calls, ensure_ascii=False))

    return total


def _total_chars(messages: list[dict[str, Any]]) -> int:
    return sum(_message_chars(m) for m in messages)


def _truncate(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text

    suffix = " …[truncated]"
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
