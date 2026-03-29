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
    three things:

      1. If message is provided, store it as a user turn.
      2. Send context-bounded wave history to the model.
      3. Store and return the model's response.

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
      - Tool call summaries (⚡ bash: ...) go to stderr.
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

            err_console.print(f"[yellow]⚡ {name}:[/yellow] {_summarize(name, arguments)}")
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


def _summarize(name: str, arguments: dict[str, Any]) -> str:
    """One-line summary of tool arguments for stderr display."""
    if name == "bash":
        cmd: str = arguments["command"]
        return cmd[:77] + "..." if len(cmd) > 80 else cmd
    elif name == "read":
        s = arguments["path"]
        if arguments.get("offset"):
            s += f" (from line {arguments['offset']})"
        return s
    elif name == "write":
        return arguments["path"]
    elif name == "edit":
        n = len(arguments["edits"]) if "edits" in arguments else 1
        plural = "edit" if n == 1 else "edits"
        return f"{arguments['path']} ({n} {plural})"
    elif name == "config":
        return "(snapshot)"
    return str(arguments)
