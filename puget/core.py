"""Core turn execution engine for puget.

This module contains the two fundamental operations:

  turn()  — The primitive. One model call, one response. No tool execution,
             no looping, no rendering. Takes a wave forward by exactly one
             step and returns what the model said.

  run()   — The loop. Sends a message, auto-executes tool calls, and repeats
             until the model responds with plain text. Renders output as it
             goes (tool summaries to stderr, assistant text to stdout).

Everything else in puget — the CLI, the REPL, the one-shot mode — is built
on top of these two functions.

Context management:

  Before every model call, turn() runs proactive compaction. If the
  estimated context tokens approach the model's context window, puget
  generates a structured summary of older turns via the model and stores
  it in the compactions table. Subsequent model calls see the summary
  plus only the recent (kept) turns.

  Emergency mode and auto-forking remain as fallback safety nets for
  when compaction alone isn't enough (e.g. a single turn is enormous).
"""

import json
import sqlite3
from typing import Any

import httpx

from puget import context, db, model, tools
from puget.output import err_console, print_assistant, print_thinking


def turn(conn: sqlite3.Connection, wid: int, message: str | None = None) -> dict[str, Any]:
    """Execute a single turn in a wave.

    This is the primitive that everything else builds on. It does exactly
    four things:

      1. If message is provided, store it as a user turn.
      2. Run proactive compaction if context is approaching the limit.
      3. Send context-bounded wave history to the model.
      4. Store and return the model's response.

    If the model request fails with HTTP 400 (request too large), turn()
    retries once in emergency context mode. If that also fails with 400,
    it auto-forks to a new wave with a tiny carry-over summary and retries
    there.

    It does NOT execute tool calls, render output, or loop. The caller
    decides what to do with the response.

    Args:
        conn: SQLite connection.
        wid: Wave ID.
        message: Optional user message to add before calling the model.
                 None when continuing after a tool result.

    Returns:
        The model's response dict. Includes `wave_id`, which may differ
        from `wid` if an auto-fork occurred.
    """
    if message is not None:
        db.add_turn(conn, wid, "user", message)

    # Proactive compaction — summarize older turns before the model call
    # so the payload stays within the context window.
    _maybe_compact(conn, wid)

    response, active_wid = _chat_with_size_recovery(conn, wid)

    # Store the response. Content and tool_calls live in separate columns —
    # no JSON blob gymnastics, no sniffing on the way back out.
    db.add_turn(
        conn, active_wid, "assistant",
        content=response["content"],
        tool_calls=json.dumps(response["tool_calls"]) if response["tool_calls"] else None,
    )

    response["wave_id"] = active_wid
    return response


def run(conn: sqlite3.Connection, wid: int, message: str) -> dict[str, Any]:
    """Send a message and run to completion.

    This is the high-level operation for end users. It calls turn() to
    get the model's response, and if the model requests tool calls, it:

      1. Executes each tool call via the built-in tool registry.
      2. Stores the results as tool turns.
      3. Calls turn() again (with no message) for the model's next response.
      4. Repeats until the model responds with plain text.

    Output is rendered as it goes:
      - Tool call summaries go to stderr.
      - Assistant text responses are rendered as markdown to stdout.

    Args:
        conn: SQLite connection.
        wid: Wave ID.
        message: The user's message.

    Returns:
        The final text-only response dict. Includes `wave_id`, which may
        differ from `wid` if the run auto-forked after repeated 400s.
    """
    active_wid = wid
    response = turn(conn, active_wid, message)
    active_wid = response.get("wave_id", active_wid)

    # Show thinking if the model produced any.
    print_thinking(response.get("thinking"))

    # If the model responded with both text and tool calls, show the text
    # before we start executing tools.
    if response["content"] and response["tool_calls"]:
        print_assistant(response["content"])

    while response["tool_calls"]:
        for tc in response["tool_calls"]:
            name: str = tc["function"]["name"]
            arguments: dict[str, Any] = tc["function"]["arguments"]

            _print_tool_call(name, arguments)
            result_text = tools.execute(name, arguments)
            db.add_turn(conn, active_wid, "tool", result_text)

        # Continue the wave — no user message, just the tool results.
        response = turn(conn, active_wid)
        active_wid = response.get("wave_id", active_wid)

        # Show thinking if the model produced any.
        print_thinking(response.get("thinking"))

        # Show any text that accompanies further tool calls.
        if response["content"] and response["tool_calls"]:
            print_assistant(response["content"])

    # Final text-only response.
    print_assistant(response["content"])
    response["wave_id"] = active_wid
    return response


# ---------------------------------------------------------------------------
# Proactive compaction
# ---------------------------------------------------------------------------

def _maybe_compact(conn: sqlite3.Connection, wid: int) -> None:
    """Run compaction if the estimated context exceeds the threshold.

    This is called before every model call in turn(). It estimates the
    current context size and, if it's approaching the context window
    limit, generates a structured summary of older turns via the model.

    Compaction failure is non-fatal — if the summarization call fails,
    we log and continue. The emergency/fork mechanism in
    _chat_with_size_recovery will handle the situation if the payload
    is actually too large.
    """
    from puget.prompt import system_message

    config = context.config_for_context_window(model.get_context_window())
    compaction = db.latest_compaction(conn, wid)

    # Determine which turns the model would see.
    if compaction:
        turns = db.get_turns_from(conn, wid, compaction["first_kept_turn_id"])
        summary = compaction["summary"]
    else:
        turns = db.get_turns(conn, wid)
        summary = None

    sys_content = system_message()["content"]
    estimated = context.estimate_context_tokens(sys_content, turns, summary)

    if not context.should_compact(estimated, config):
        return

    # Need all turns for compaction preparation (to find file ops, etc.).
    all_turns = db.get_turns(conn, wid)
    preparation = context.prepare_compaction(
        all_turns, sys_content, compaction, config,
    )
    if preparation is None:
        return

    err_console.print("[dim]Compacting context\u2026[/dim]")

    try:
        messages = context.build_summarization_messages(preparation)
        raw_summary = model.complete(messages)
        final_summary = context.finalize_summary(raw_summary, preparation)

        details = json.dumps({
            "read_files": preparation.read_files,
            "modified_files": preparation.modified_files,
        })

        db.add_compaction(
            conn, wid,
            summary=final_summary,
            first_kept_turn_id=preparation.first_kept_turn_id,
            tokens_before=preparation.tokens_before,
            details_json=details,
        )
        err_console.print("[dim]Context compacted.[/dim]")

    except Exception as exc:
        err_console.print(
            f"[dim]Compaction failed: {exc}. Continuing with full context.[/dim]"
        )


# ---------------------------------------------------------------------------
# Model call with size recovery
# ---------------------------------------------------------------------------

def _chat_with_size_recovery(
    conn: sqlite3.Connection,
    wid: int,
) -> tuple[dict[str, Any], int]:
    """Call the model with context guards and 400 recovery."""
    messages = db.messages_for_model(conn, wid)
    try:
        return model.chat(messages), wid
    except httpx.HTTPStatusError as e:
        if e.response.status_code != 400:
            raise

    err_console.print("[dim]Request too large; retrying with reduced context.[/dim]")

    emergency_messages = db.messages_for_model(conn, wid, emergency=True)
    try:
        return model.chat(emergency_messages), wid
    except httpx.HTTPStatusError as e:
        if e.response.status_code != 400:
            raise

    new_wid = _fork_wave_after_400(conn, wid)
    err_console.print(f"[dim]Request still too large. Forked to wave #{new_wid}.[/dim]")

    fork_messages = db.messages_for_model(conn, new_wid, emergency=True)
    return model.chat(fork_messages), new_wid


def _fork_wave_after_400(conn: sqlite3.Connection, wid: int) -> int:
    """Create a new wave carrying only tiny context + latest request."""
    turns = db.get_turns(conn, wid)
    summary = context.build_tiny_summary(turns)
    latest_request = context.latest_user_request(turns)

    new_wid = db.new_wave(conn, label=f"fork of wave #{wid}")
    db.add_turn(conn, new_wid, "user", context.build_fork_preamble(wid, summary))

    if latest_request:
        db.add_turn(conn, new_wid, "user", latest_request)

    return new_wid


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _print_tool_call(name: str, arguments: dict[str, Any]) -> None:
    """Render a tool call to stderr with preserved line breaks."""
    from rich.syntax import Syntax

    err_console.print(f"[bold yellow]\u26a1 {name}[/bold yellow]")

    # Build a readable representation preserving real line breaks.
    parts: list[str] = []
    for key, val in arguments.items():
        if isinstance(val, str) and "\n" in val:
            # Multi-line string — show as a code block.
            parts.append(f"{key}:\n{val}")
        else:
            parts.append(f"{key}: {json.dumps(val)}")

    body = "\n".join(parts)
    err_console.print(Syntax(body, "yaml", theme="monokai", line_numbers=False))
    err_console.print()


