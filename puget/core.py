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

from puget import db, model, tools
from puget.output import err_console, print_assistant, print_thinking


def turn(conn: sqlite3.Connection, wid: int, message: str | None = None) -> dict[str, Any]:
    """Execute a single turn in a wave.

    This is the primitive that everything else builds on. It does exactly
    three things:

      1. If message is provided, store it as a user turn.
      2. Send the full wave history to the model.
      3. Store and return the model's response.

    It does NOT execute tool calls, render output, or loop. The caller
    decides what to do with the response.

    Args:
        conn: SQLite connection.
        wid: Wave ID.
        message: Optional user message to add before calling the model.
                 None when continuing after a tool result.

    Returns:
        The model's response dict:
          {"role": "assistant", "content": str, "tool_calls": list | None}
    """
    if message is not None:
        db.add_turn(conn, wid, "user", message)

    messages = db.messages_for_model(conn, wid)
    response = model.chat(messages)

    # Store the response. Content and tool_calls live in separate columns —
    # no JSON blob gymnastics, no sniffing on the way back out.
    db.add_turn(
        conn, wid, "assistant",
        content=response["content"],
        tool_calls=json.dumps(response["tool_calls"]) if response["tool_calls"] else None,
    )

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
        The final text-only response dict (tool_calls will be None).
    """
    response = turn(conn, wid, message)

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
            db.add_turn(conn, wid, "tool", result_text)

        # Continue the wave — no user message, just the tool results.
        response = turn(conn, wid)

        # Show thinking if the model produced any.
        print_thinking(response.get("thinking"))

        # Show any text that accompanies further tool calls.
        if response["content"] and response["tool_calls"]:
            print_assistant(response["content"])

    # Final text-only response.
    print_assistant(response["content"])
    return response


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
    return str(arguments)
