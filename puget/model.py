"""Ollama model interaction for puget.

Uses raw HTTP calls to the Ollama API instead of the ollama Python library.
This preserves opaque fields like tool_call ``id`` which Ollama uses to
round-trip provider-specific metadata (e.g. Gemini's ``thought_signature``).
The typed library silently drops these, causing 400 errors on the next turn.
"""

import os
import re
from typing import Any

import httpx


DEFAULT_MODEL = "minimax-m2.7:cloud"

# Timeout: None means no read timeout (large model responses can be slow).
_TIMEOUT = httpx.Timeout(connect=10, read=None, write=10, pool=10)

# Runtime model override. When set, takes precedence over the env var.
_model_override: str | None = None


def get_model() -> str:
    """Return the active model name.

    Priority: set_model() override > $PUGET_OLLAMA_MODEL > DEFAULT_MODEL.
    """
    if _model_override is not None:
        return _model_override
    return os.environ.get("PUGET_OLLAMA_MODEL", DEFAULT_MODEL)


def set_model(name: str | None) -> None:
    """Set or clear the runtime model override.

    Args:
        name: Model name to use, or None to revert to env/default.
    """
    global _model_override
    _model_override = name


def _base_url() -> str:
    """Resolve the Ollama base URL.

    Checks $PUGET_OLLAMA_HOST, then $OLLAMA_HOST, then defaults to
    http://localhost:11434.  Ensures the result has a scheme.
    """
    host = os.environ.get("PUGET_OLLAMA_HOST") or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return host.rstrip("/")


def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Call the model and return the response message.

    If tools is not explicitly provided, the built-in tool definitions are used.

    Returns a dict with either:
      - {"role": "assistant", "content": "...", "tool_calls": None}
      - {"role": "assistant", "content": "", "tool_calls": [...]}

    Tool-call dicts preserve all fields from the Ollama API (including
    ``id`` and ``function.index``) so they can be echoed back verbatim.
    """
    if tools is None:
        from puget.tools import TOOL_DEFINITIONS
        tools = TOOL_DEFINITIONS

    payload: dict[str, Any] = {
        "model": get_model(),
        "messages": messages,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
    # Always request thinking from the model — it produces better results
    # even when the thinking isn't displayed. Display is controlled separately
    # by show_thinking() in output.py.
    payload["think"] = True

    url = f"{_base_url()}/api/chat"
    resp = httpx.post(url, json=payload, timeout=_TIMEOUT)
    resp.raise_for_status()
    msg = resp.json()["message"]

    content: str = msg.get("content") or ""

    # Thinking can arrive two ways:
    #   1. Ollama's native "thinking" field (when think=true is supported).
    #   2. <think>...</think> tags embedded in content (older models).
    # Check both, preferring the native field.
    thinking: str | None = (msg.get("thinking") or "").strip() or None

    if thinking is None and "</think>" in content:
        match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if match:
            thinking = match.group(1).strip() or None

    # Always strip <think> tags from content regardless.
    if "</think>" in content:
        content = content.split("</think>", 1)[-1].strip()

    # Preserve tool_calls exactly as returned — including id, index, etc.
    raw_tool_calls: list[dict[str, Any]] | None = msg.get("tool_calls") or None

    return {
        "role": "assistant",
        "content": content,
        "tool_calls": raw_tool_calls,
        "thinking": thinking,
    }
