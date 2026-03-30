"""Centralized runtime configuration for puget.

Every resolved path, every env-var-with-default, every runtime override —
one module, one snapshot. Other modules should import from here when they
need a resolved value rather than doing their own os.environ.get().

The snapshot() function returns the full resolved state as a plain dict,
suitable for JSON serialization. This is what the `config` tool exposes
to the model.
"""

import json
import os
from pathlib import Path
from typing import Any


def puget_home() -> Path:
    """Resolve the puget home directory.

    Uses $PUGET_HOME if set, otherwise ~/.puget.
    """
    env = os.environ.get("PUGET_HOME")
    if env:
        return Path(env)
    return Path.home() / ".puget"


def db_path() -> Path:
    """Resolve the database path.

    Priority: $PUGET_DB > $PUGET_HOME/puget.db > ~/.puget/puget.db.
    """
    env = os.environ.get("PUGET_DB")
    if env:
        return Path(env)
    return puget_home() / "puget.db"


def ollama_host() -> str:
    """Resolve the Ollama API base URL.

    Checks $PUGET_OLLAMA_HOST, then $OLLAMA_HOST, then defaults to
    http://localhost:11434.
    """
    host = (
        os.environ.get("PUGET_OLLAMA_HOST")
        or os.environ.get("OLLAMA_HOST")
        or "http://localhost:11434"
    )
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return host.rstrip("/")


def model_name() -> str:
    """Return the active model name.

    Defers to model.get_model() which handles runtime overrides.
    """
    from puget.model import get_model
    return get_model()


def show_thinking() -> bool:
    """Return whether thinking display is currently enabled."""
    from puget.output import show_thinking as _show_thinking
    return _show_thinking()


def thinking_mode() -> str:
    """Return puget's active Ollama thinking policy."""
    from puget.model import get_thinking_mode
    return get_thinking_mode()


def model_info() -> dict[str, Any]:
    """Return active-model metadata resolved from Ollama."""
    from puget.model import get_model_info
    return get_model_info()


def current_wave_id() -> int | None:
    """Return the current (most recent) wave ID, or None.

    Returns None without side effects if the DB doesn't exist yet.
    """
    if not db_path().is_file():
        return None
    from puget import db
    try:
        conn = db.connect()
        return db.current_wave_id(conn)
    except Exception:
        return None


def wave_turn_count(wave_id: int) -> int | None:
    """Return the number of turns in a wave, or None on error."""
    if not db_path().is_file():
        return None
    from puget import db
    try:
        conn = db.connect()
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM turns WHERE wave_id = ?",
            (wave_id,),
        ).fetchone()
        return row["n"] if row else 0
    except Exception:
        return None


def skill_dirs() -> list[dict[str, Any]]:
    """Return skill directories with existence flags."""
    from puget.skills import skill_dirs as _skill_dirs
    dirs = _skill_dirs()
    return [
        {"path": str(d), "exists": d.is_dir()}
        for d in dirs
    ]


def snapshot() -> dict[str, Any]:
    """Return the full resolved configuration as a plain dict.

    This is the canonical "tell me about yourself" payload. Every value
    is concrete and resolved — no env var names, no templates, no
    defaults-that-might-apply. The model gets facts.
    """
    home = puget_home()
    db = db_path()
    wid = current_wave_id()

    info = model_info()

    result: dict[str, Any] = {
        "puget_home": str(home),
        "db_path": str(db),
        "db_exists": db.is_file(),
        "model": model_name(),
        "model_capabilities": info["capabilities"],
        "model_capabilities_known": info["capabilities_known"],
        "context_window": info["context_window"],
        "ollama_host": ollama_host(),
        "show_thinking": show_thinking(),
        "thinking_mode": thinking_mode(),
        "cwd": os.getcwd(),
        "skill_dirs": skill_dirs(),
        "current_wave_id": wid,
    }

    if wid is not None:
        count = wave_turn_count(wid)
        if count is not None:
            result["current_wave_turn_count"] = count

    return result


def snapshot_json(indent: int = 2) -> str:
    """Return snapshot() as a formatted JSON string."""
    return json.dumps(snapshot(), indent=indent)
