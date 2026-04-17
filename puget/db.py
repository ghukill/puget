"""SQLite storage layer for puget.

The database is the entire state of puget. No hidden config, no daemon,
no lock files. `sqlite3 $PUGET_HOME/puget.db` is your debugger.

Three tables:

  waves  — A wave is a unit of work or discussion. Named for the Puget
           Sound. Each wave has an optional label; if unlabeled, the first
           user message serves as a preview when listing waves.

  turns  — The ordered sequence of messages within a wave. Each turn has
           a role (user, assistant, tool), text content, and an optional
           tool_calls JSON column for assistant turns that request tools.
           Content is always plain text — tool call structure lives in its
           own column, never smuggled into content as a JSON blob.

  compactions — Context compaction checkpoints. Each compaction stores a
           structured summary of older turns plus the ID of the first turn
           that was kept (not summarized). When building model context,
           the summary replaces all turns before first_kept_turn_id.
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
        CREATE TABLE IF NOT EXISTS compactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wave_id INTEGER NOT NULL REFERENCES waves(id),
            summary TEXT NOT NULL,
            first_kept_turn_id INTEGER NOT NULL REFERENCES turns(id),
            tokens_before INTEGER NOT NULL,
            details_json TEXT,
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


def list_waves(conn: sqlite3.Connection, max_chars: int = 200) -> list[dict[str, Any]]:
    """List all waves in reverse chronological order with previews.

    Returns a list of dicts with keys: id, label, created_at, preview.
    The preview is truncated to max_chars.
    """
    rows = conn.execute(
        "SELECT id, label, created_at FROM waves ORDER BY id DESC"
    ).fetchall()
    result = []
    for row in rows:
        preview = row["label"] or None
        if preview is None:
            # Fall back to first user message.
            msg = conn.execute(
                "SELECT content FROM turns "
                "WHERE wave_id = ? AND role = 'user' ORDER BY id LIMIT 1",
                (row["id"],),
            ).fetchone()
            preview = msg["content"] if msg else "(empty)"
        if len(preview) > max_chars:
            preview = preview[:max_chars] + "\u2026"
        result.append({
            "id": row["id"],
            "label": row["label"],
            "created_at": row["created_at"],
            "preview": preview,
        })
    return result


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
            return label[:max_chars] + "\u2026"
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
        return content[:max_chars] + "\u2026"
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

    Returns a list of dicts with keys: id, role, content, tool_calls,
    created_at. The id is included so compaction can reference specific
    turn boundaries.
    """
    rows = conn.execute(
        "SELECT id, role, content, tool_calls, created_at FROM turns "
        "WHERE wave_id = ? ORDER BY id",
        (wave_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_turns_from(
    conn: sqlite3.Connection,
    wave_id: int,
    from_turn_id: int,
) -> list[dict[str, Any]]:
    """Get turns in a wave starting from a specific turn ID.

    Used after compaction to load only the kept (non-summarized) turns.

    Returns a list of dicts with keys: id, role, content, tool_calls,
    created_at.
    """
    rows = conn.execute(
        "SELECT id, role, content, tool_calls, created_at FROM turns "
        "WHERE wave_id = ? AND id >= ? ORDER BY id",
        (wave_id, from_turn_id),
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


# -- Compactions -------------------------------------------------------------

def add_compaction(
    conn: sqlite3.Connection,
    wave_id: int,
    summary: str,
    first_kept_turn_id: int,
    tokens_before: int,
    details_json: str | None = None,
) -> None:
    """Store a compaction checkpoint for a wave.

    Args:
        conn: SQLite connection.
        wave_id: The wave this compaction belongs to.
        summary: Structured summary of the compacted turns.
        first_kept_turn_id: ID of the first turn NOT summarized.
        tokens_before: Estimated context tokens before compaction.
        details_json: Optional JSON with file operation details.
    """
    conn.execute(
        "INSERT INTO compactions "
        "(wave_id, summary, first_kept_turn_id, tokens_before, details_json) "
        "VALUES (?, ?, ?, ?, ?)",
        (wave_id, summary, first_kept_turn_id, tokens_before, details_json),
    )
    conn.commit()


def latest_compaction(
    conn: sqlite3.Connection,
    wave_id: int,
) -> dict[str, Any] | None:
    """Get the most recent compaction for a wave, or None.

    Returns a dict with keys: id, summary, first_kept_turn_id,
    tokens_before, details_json, created_at.
    """
    row = conn.execute(
        "SELECT id, summary, first_kept_turn_id, tokens_before, "
        "details_json, created_at "
        "FROM compactions WHERE wave_id = ? ORDER BY id DESC LIMIT 1",
        (wave_id,),
    ).fetchone()
    return dict(row) if row else None


# -- Messages for model ------------------------------------------------------

def messages_for_model(
    conn: sqlite3.Connection,
    wave_id: int,
    *,
    emergency: bool = False,
) -> list[dict[str, Any]]:
    """Build context-bounded model messages for a wave.

    When a compaction exists, only turns from first_kept_turn_id onwards
    are included, preceded by the compaction summary. This replaces the
    old approach of silently dropping turns.

    In emergency mode, compaction's kept turns are used but the summary
    is omitted to minimize payload size.

    Args:
        conn: SQLite connection.
        wave_id: Wave ID.
        emergency: Build a reduced emergency payload used after a 400.

    Returns:
        Message list ready for model.chat().
    """
    from puget import model
    from puget.context import build_messages, config_for_context_window
    from puget.prompt import system_message

    config = config_for_context_window(model.get_context_window())
    compaction = latest_compaction(conn, wave_id)

    if compaction:
        turns = get_turns_from(conn, wave_id, compaction["first_kept_turn_id"])
        if emergency:
            return build_messages(system_message(), turns, config=config, emergency=True)
        return build_messages(
            system_message(), turns,
            config=config,
            compaction_summary=compaction["summary"],
        )

    turns = get_turns(conn, wave_id)
    return build_messages(system_message(), turns, config=config, emergency=emergency)
