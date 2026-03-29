"""SQLite storage layer for puget.

The database is the entire state of puget. No hidden config, no daemon,
no lock files. `sqlite3 $PUGET_HOME/puget.db` is your debugger.

Two tables:

  waves  — A wave is a unit of work or discussion. Named for the Puget
           Sound. Each wave has an optional label; if unlabeled, the first
           user message serves as a preview when listing waves.

  turns  — The ordered sequence of messages within a wave. Each turn has
           a role (user, assistant, tool), text content, and an optional
           tool_calls JSON column for assistant turns that request tools.
           Content is always plain text — tool call structure lives in its
           own column, never smuggled into content as a JSON blob.
"""

import os
import sqlite3
from pathlib import Path
from typing import Any


def _db_path() -> Path:
    """Resolve the database path.

    Uses $PUGET_DB if set, otherwise $PUGET_HOME/puget.db (defaulting
    to ~/.puget/puget.db). The parent directory is created automatically
    on connect().
    """
    env = os.environ.get("PUGET_DB")
    if env:
        return Path(env)
    from puget import puget_home
    return puget_home() / "puget.db"


def connect() -> sqlite3.Connection:
    """Open a connection to the puget database.

    Creates the database file and parent directories if they don't exist.
    Enables WAL mode for concurrent reads and foreign keys for integrity.
    Schema is created automatically on first connect.
    """
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist.

    Safe to call on every connection — uses IF NOT EXISTS throughout.
    """
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS waves (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
        CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wave_id INTEGER NOT NULL REFERENCES waves(id),
            role TEXT NOT NULL,
            content TEXT NOT NULL DEFAULT '',
            tool_calls TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
    """)


# -- Waves -------------------------------------------------------------------

def current_wave_id(conn: sqlite3.Connection) -> int | None:
    """Get the ID of the most recent wave, or None if no waves exist."""
    row = conn.execute(
        "SELECT id FROM waves ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return row["id"] if row else None


def new_wave(conn: sqlite3.Connection, label: str | None = None) -> int:
    """Create a new wave and return its ID.

    The optional label is a short human-readable description. If omitted,
    the wave can be identified later by previewing its first user message.
    """
    cur = conn.execute("INSERT INTO waves (label) VALUES (?)", (label,))
    conn.commit()
    assert cur.lastrowid is not None
    return cur.lastrowid


def ensure_wave(conn: sqlite3.Connection) -> int:
    """Return the current wave ID, creating one if none exist."""
    wid = current_wave_id(conn)
    if wid is None:
        wid = new_wave(conn)
    return wid


def wave_preview(conn: sqlite3.Connection, wave_id: int, max_chars: int = 250) -> str:
    """Get a preview string for a wave.

    Returns the wave's label if it has one, otherwise the first user
    message truncated to max_chars. Returns "(empty)" if the wave has
    no user turns.
    """
    # Check for a label first.
    row = conn.execute(
        "SELECT label FROM waves WHERE id = ?", (wave_id,)
    ).fetchone()
    if row and row["label"]:
        label: str = row["label"]
        if len(label) > max_chars:
            return label[:max_chars] + "…"
        return label

    # Fall back to first user message.
    row = conn.execute(
        "SELECT content FROM turns "
        "WHERE wave_id = ? AND role = 'user' ORDER BY id LIMIT 1",
        (wave_id,),
    ).fetchone()
    if row is None:
        return "(empty)"
    content: str = row["content"]
    if len(content) > max_chars:
        return content[:max_chars] + "…"
    return content


# -- Turns -------------------------------------------------------------------

def add_turn(
    conn: sqlite3.Connection,
    wave_id: int,
    role: str,
    content: str,
    tool_calls: str | None = None,
) -> None:
    """Append a turn to a wave.

    Args:
        conn: SQLite connection.
        wave_id: The wave this turn belongs to.
        role: One of 'user', 'assistant', 'tool'.
        content: Text content of the turn. Always a string, even if empty.
        tool_calls: Optional JSON string of tool calls. Only meaningful
                    for assistant turns. Pass json.dumps(list) or None.
    """
    conn.execute(
        "INSERT INTO turns (wave_id, role, content, tool_calls) VALUES (?, ?, ?, ?)",
        (wave_id, role, content, tool_calls),
    )
    conn.commit()


def get_turns(conn: sqlite3.Connection, wave_id: int) -> list[dict[str, Any]]:
    """Get all turns in a wave, ordered chronologically.

    Returns a list of dicts with keys: role, content, tool_calls, created_at.
    """
    rows = conn.execute(
        "SELECT role, content, tool_calls, created_at FROM turns "
        "WHERE wave_id = ? ORDER BY id",
        (wave_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def last_assistant_turn(conn: sqlite3.Connection, wave_id: int) -> dict[str, Any] | None:
    """Get the most recent assistant turn in a wave, or None.

    Returns a dict with keys: role, content, tool_calls.
    """
    row = conn.execute(
        "SELECT role, content, tool_calls FROM turns "
        "WHERE wave_id = ? AND role = 'assistant' ORDER BY id DESC LIMIT 1",
        (wave_id,),
    ).fetchone()
    return dict(row) if row else None


def messages_for_model(
    conn: sqlite3.Connection,
    wave_id: int,
    *,
    emergency: bool = False,
) -> list[dict[str, Any]]:
    """Build context-bounded model messages for a wave.

    Includes the dynamic system message, reconstructs turn messages, and
    enforces request-size guardrails via puget.context.

    Args:
        conn: SQLite connection.
        wave_id: Wave ID.
        emergency: Build a reduced emergency payload used after a 400.

    Returns:
        Message list ready for model.chat().
    """
    from puget.context import build_messages
    from puget.prompt import system_message

    turns = get_turns(conn, wave_id)
    return build_messages(system_message(), turns, emergency=emergency)
