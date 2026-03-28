"""Ollama model interaction for puget."""

import os
from typing import Any

import ollama


DEFAULT_MODEL = "minimax-m2.7:cloud"


def get_model() -> str:
    """Return the model name from $PUGET_OLLAMA_MODEL or the default."""
    return os.environ.get("PUGET_OLLAMA_MODEL", DEFAULT_MODEL)


def _client() -> ollama.Client:
    """Return an Ollama client, respecting $PUGET_OLLAMA_HOST.

    If $PUGET_OLLAMA_HOST is set it takes precedence.  Otherwise the
    ollama library falls back to $OLLAMA_HOST, then its built-in
    default (http://localhost:11434).
    """
    host = os.environ.get("PUGET_OLLAMA_HOST")
    return ollama.Client(host=host)


def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Call the model and return the response message.

    If tools is not explicitly provided, the built-in tool definitions are used.

    Returns a dict with either:
      - {"role": "assistant", "content": "...", "tool_calls": None}
      - {"role": "assistant", "content": "", "tool_calls": [...]}
    """
    if tools is None:
        from puget.tools import TOOL_DEFINITIONS
        tools = TOOL_DEFINITIONS

    response = _client().chat(
        model=get_model(),
        messages=messages,
        tools=tools if tools else None,
    )
    msg = response.message

    content = msg.content or ""
    # Strip <think> blocks from reasoning models
    if "</think>" in content:
        content = content.split("</think>", 1)[-1].strip()

    tool_calls: list[dict[str, Any]] | None = None
    if msg.tool_calls:
        tool_calls = []
        for tc in msg.tool_calls:
            tool_calls.append({
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            })

    return {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
    }
