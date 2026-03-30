"""Ollama model interaction for puget.

Uses raw HTTP calls to the Ollama API instead of the ollama Python library.
This preserves opaque fields like tool_call ``id`` which Ollama uses to
round-trip provider-specific metadata (e.g. Gemini's ``thought_signature``).
The typed library silently drops these, causing 400 errors on the next turn.
"""

import os
import re
import time
from typing import Any, Literal

import httpx


DEFAULT_MODEL = "minimax-m2.7:cloud"
DEFAULT_CONTEXT_WINDOW = 128_000
DEFAULT_THINKING_MODE = "auto"
VALID_THINKING_MODES = {"off", "low", "on", "auto"}

# Chat template tokens that models sometimes leak into content.
# Qwen: <|im_start|>, <|im_end|>  Llama: <|start_header_id|>, <|end_header_id|>, <|eot_id|>
_CHAT_TEMPLATE_RE = re.compile(r"<\|[a-z_]+\|>")

# Timeout: None means no read timeout (large model responses can be slow).
_TIMEOUT = httpx.Timeout(connect=10, read=None, write=10, pool=10)
_METADATA_TIMEOUT = httpx.Timeout(connect=2, read=5, write=5, pool=5)
_RUNTIME_CONTEXT_TTL = 2.0

# Runtime model override. When set, takes precedence over the env var.
_model_override: str | None = None
_thinking_mode_override: str | None = None
_model_show_cache: dict[str, dict[str, Any]] = {}
_runtime_context_cache: dict[str, tuple[float, int | None]] = {}


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



def get_thinking_mode() -> str:
    """Return the active thinking policy.

    Priority: set_thinking_mode() override > $PUGET_OLLAMA_THINK > "auto".

    Modes:
      - off  -> send think=False for normal chat turns
      - low  -> send think="low"
      - on   -> send think=True
      - auto -> normal chat turns use think=False; internal summarization
                calls use think="low"
    """
    raw = _thinking_mode_override or os.environ.get("PUGET_OLLAMA_THINK", DEFAULT_THINKING_MODE)
    mode = raw.strip().lower()
    if mode in VALID_THINKING_MODES:
        return mode
    return DEFAULT_THINKING_MODE



def set_thinking_mode(mode: str | None) -> None:
    """Set or clear the runtime thinking-mode override."""
    global _thinking_mode_override
    if mode is None:
        _thinking_mode_override = None
        return

    normalized = mode.strip().lower()
    if normalized not in VALID_THINKING_MODES:
        valid = ", ".join(sorted(VALID_THINKING_MODES))
        raise ValueError(f"invalid thinking mode: {mode!r} (expected one of: {valid})")
    _thinking_mode_override = normalized



def list_available_models() -> list[str]:
    """Return locally available Ollama model names.

    Uses the official ollama Python SDK for model discovery. Returns an
    empty list if the API is unreachable or the SDK response format is
    unexpected.
    """
    try:
        import ollama

        client = ollama.Client(host=_base_url())
        response = client.list()
    except Exception:
        return []

    models = response.get("models") if isinstance(response, dict) else getattr(response, "models", None)
    if not models:
        return []

    names: list[str] = []
    for model in models:
        if isinstance(model, dict):
            name = model.get("model") or model.get("name")
        else:
            name = getattr(model, "model", None) or getattr(model, "name", None)

        if isinstance(name, str) and name:
            names.append(name)

    return sorted(set(names))



def get_model_capabilities(model_name: str | None = None) -> list[str]:
    """Return Ollama capabilities for a model, if available."""
    data = _show_model(model_name or get_model())
    caps = data.get("capabilities")
    if not isinstance(caps, list):
        return []
    return sorted({cap for cap in caps if isinstance(cap, str) and cap})



def get_context_window(model_name: str | None = None) -> int:
    """Return the best-known context window for a model.

    Prefers the runtime context length from `/api/ps` when the model is
    currently loaded, then falls back to the model's static metadata from
    `/api/show`, then finally to DEFAULT_CONTEXT_WINDOW.
    """
    name = model_name or get_model()

    runtime_window = _runtime_context_window(name)
    if runtime_window is not None:
        return runtime_window

    static_window = _extract_context_window(_show_model(name))
    if static_window is not None:
        return static_window

    return DEFAULT_CONTEXT_WINDOW



def get_model_info(model_name: str | None = None) -> dict[str, Any]:
    """Return model metadata useful to the agent and UI."""
    name = model_name or get_model()
    data = _show_model(name)
    caps = data.get("capabilities")
    capabilities = sorted({cap for cap in caps if isinstance(cap, str) and cap}) if isinstance(caps, list) else []

    return {
        "model": name,
        "capabilities": capabilities,
        "capabilities_known": isinstance(caps, list),
        "supports_tools": _supports_capability(name, "tools"),
        "supports_thinking": _supports_capability(name, "thinking"),
        "context_window": get_context_window(name),
        "thinking_mode": get_thinking_mode(),
    }



def _base_url() -> str:
    """Resolve the Ollama base URL.

    Checks $PUGET_OLLAMA_HOST, then $OLLAMA_HOST, then defaults to
    http://localhost:11434.  Ensures the result has a scheme.
    """
    host = os.environ.get("PUGET_OLLAMA_HOST") or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return host.rstrip("/")



def complete(messages: list[dict[str, Any]]) -> str:
    """Call the model without tools and return the text content.

    Used for summarization and other non-tool tasks. Thinking follows the
    active policy and model capability checks, then is stripped from the
    returned content.
    """
    model_name = get_model()
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "stream": False,
    }
    think = _resolve_think_value(model_name, task="summary")
    if think is not None:
        payload["think"] = think

    url = f"{_base_url()}/api/chat"
    resp = httpx.post(url, json=payload, timeout=_TIMEOUT)
    resp.raise_for_status()
    msg = resp.json()["message"]

    content: str = msg.get("content") or ""
    if "</think>" in content:
        content = content.split("</think>", 1)[-1].strip()
    return _strip_chat_template_tokens(content)



def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Call the model and return the response message.

    If tools is not explicitly provided, the built-in tool definitions are used.

    Returns a dict with either:
      - {"role": "assistant", "content": "...", "tool_calls": None}
      - {"role": "assistant", "content": "", "tool_calls": [...]}.

    Tool-call dicts preserve all fields from the Ollama API (including
    ``id`` and ``function.index``) so they can be echoed back verbatim.
    """
    if tools is None:
        from puget.tools import TOOL_DEFINITIONS
        tools = TOOL_DEFINITIONS

    model_name = get_model()
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "stream": False,
    }

    if tools and _supports_capability(model_name, "tools"):
        payload["tools"] = tools

    think = _resolve_think_value(model_name, task="chat")
    if think is not None:
        payload["think"] = think

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

    # Strip leaked chat template tokens (e.g. <|im_start|>, <|end_header_id|>).
    content = _strip_chat_template_tokens(content)
    if thinking:
        thinking = _strip_chat_template_tokens(thinking) or None

    # Preserve tool_calls exactly as returned — including id, index, etc.
    raw_tool_calls: list[dict[str, Any]] | None = msg.get("tool_calls") or None

    return {
        "role": "assistant",
        "content": content,
        "tool_calls": raw_tool_calls,
        "thinking": thinking,
    }



def _show_model(model_name: str) -> dict[str, Any]:
    """Fetch and cache `/api/show` metadata for a model."""
    cached = _model_show_cache.get(model_name)
    if cached is not None:
        return cached

    try:
        url = f"{_base_url()}/api/show"
        resp = httpx.post(url, json={"model": model_name}, timeout=_METADATA_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            _model_show_cache[model_name] = data
            return data
    except Exception:
        pass

    return {}



def _runtime_context_window(model_name: str) -> int | None:
    """Return the active runtime context length from `/api/ps`, if loaded."""
    now = time.monotonic()
    cached = _runtime_context_cache.get(model_name)
    if cached is not None and now - cached[0] < _RUNTIME_CONTEXT_TTL:
        return cached[1]

    value: int | None = None
    try:
        url = f"{_base_url()}/api/ps"
        resp = httpx.get(url, timeout=_METADATA_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("models") if isinstance(data, dict) else None
        if isinstance(models, list):
            for item in models:
                if not isinstance(item, dict):
                    continue
                current = item.get("model") or item.get("name")
                if current != model_name:
                    continue
                value = _positive_int(item.get("context_length"))
                if value is not None:
                    break
    except Exception:
        value = None

    _runtime_context_cache[model_name] = (now, value)
    return value



def _extract_context_window(data: dict[str, Any]) -> int | None:
    """Extract a context length from `/api/show` metadata."""
    model_info = data.get("model_info")
    if not isinstance(model_info, dict):
        return None

    for key, value in model_info.items():
        if isinstance(key, str) and key.endswith(".context_length"):
            parsed = _positive_int(value)
            if parsed is not None:
                return parsed
    return None



def _positive_int(value: Any) -> int | None:
    """Return value as a positive int, or None if it can't be parsed."""
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError:
            return None
        return parsed if parsed > 0 else None
    return None



def _supports_capability(model_name: str, capability: str) -> bool:
    """Return whether a model supports a capability.

    If the server cannot provide capability metadata, default to True so
    puget preserves its previous behavior rather than disabling features
    opportunistically.
    """
    data = _show_model(model_name)
    caps = data.get("capabilities")
    if isinstance(caps, list):
        return capability in caps
    return True



def _resolve_think_value(
    model_name: str,
    *,
    task: Literal["chat", "summary"],
) -> bool | str | None:
    """Map puget's thinking policy to the Ollama `think` parameter."""
    if not _supports_capability(model_name, "thinking"):
        return None

    mode = get_thinking_mode()
    if mode == "off":
        return False
    if mode == "low":
        return "low"
    if mode == "on":
        return True
    return "low" if task == "summary" else False



def _strip_chat_template_tokens(text: str) -> str:
    """Remove leaked chat template tokens from model output."""
    return _CHAT_TEMPLATE_RE.sub("", text).strip()
