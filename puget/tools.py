"""Built-in tools for puget.

The foundational four: bash, read, write, edit.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

# -- Constants ---------------------------------------------------------------

MAX_LINES = 2000
MAX_BYTES = 50 * 1024  # 50KB
DEFAULT_TIMEOUT: float | None = None  # no timeout by default


# -- Tool definitions (Ollama format) ---------------------------------------

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Execute a bash command in the current working directory. "
                "Returns stdout and stderr. Output is truncated to the last "
                f"{MAX_LINES} lines or {MAX_BYTES // 1024}KB (whichever is hit first). "
                "If truncated, full output is saved to a temp file. "
                "Optionally provide a timeout in seconds."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Bash command to execute",
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds (optional, no default timeout)",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": (
                "Read the contents of a file. Output is truncated to "
                f"{MAX_LINES} lines or {MAX_BYTES // 1024}KB (whichever is hit first). "
                "Use offset and limit to page through large files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read (relative or absolute)",
                    },
                    "offset": {
                        "type": "number",
                        "description": "Line number to start reading from (1-indexed, optional)",
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of lines to read (optional)",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": (
                "Write content to a file. Creates the file if it doesn't exist, "
                "overwrites if it does. Automatically creates parent directories."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write (relative or absolute)",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": (
                "Edit a file using exact text replacement. Supports two modes: "
                "(1) Single replacement with oldText/newText. "
                "(2) Multiple disjoint replacements with an edits array, where each "
                "entry has oldText/newText. Each oldText must match exactly once in "
                "the file. All edits are matched against the original file content, "
                "not incrementally."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit (relative or absolute)",
                    },
                    "oldText": {
                        "type": "string",
                        "description": "Exact text to replace (single-edit mode)",
                    },
                    "newText": {
                        "type": "string",
                        "description": "Replacement text (single-edit mode)",
                    },
                    "edits": {
                        "type": "array",
                        "description": "Array of {oldText, newText} edits for multiple disjoint replacements",
                        "items": {
                            "type": "object",
                            "properties": {
                                "oldText": {
                                    "type": "string",
                                    "description": "Exact text to replace",
                                },
                                "newText": {
                                    "type": "string",
                                    "description": "Replacement text",
                                },
                            },
                            "required": ["oldText", "newText"],
                        },
                    },
                },
                "required": ["path"],
            },
        },
    },
]


# -- Dispatch ----------------------------------------------------------------

def execute(name: str, arguments: dict[str, Any]) -> str:
    """Execute a tool by name. Returns the result as a string."""
    if name == "bash":
        return _bash(
            command=arguments["command"],
            timeout=arguments.get("timeout"),
        )
    elif name == "read":
        return _read(
            path=arguments["path"],
            offset=arguments.get("offset"),
            limit=arguments.get("limit"),
        )
    elif name == "write":
        return _write(
            path=arguments["path"],
            content=arguments["content"],
        )
    elif name == "edit":
        return _edit(
            path=arguments["path"],
            old_text=arguments.get("oldText"),
            new_text=arguments.get("newText"),
            edits=arguments.get("edits"),
        )
    else:
        return f"Unknown tool: {name}"


# -- Bash tool ---------------------------------------------------------------

def _truncate(text: str) -> tuple[str, bool]:
    """Truncate to last MAX_LINES lines and MAX_BYTES bytes.

    Returns (text, was_truncated).
    """
    lines = text.splitlines(keepends=True)

    # Truncate by line count (keep the tail)
    if len(lines) > MAX_LINES:
        lines = lines[-MAX_LINES:]
        truncated = True
    else:
        truncated = False

    result = "".join(lines)

    # Truncate by byte size (keep the tail)
    encoded = result.encode("utf-8", errors="replace")
    if len(encoded) > MAX_BYTES:
        encoded = encoded[-MAX_BYTES:]
        result = encoded.decode("utf-8", errors="replace")
        truncated = True

    return result, truncated


def _bash(command: str, timeout: float | None = DEFAULT_TIMEOUT) -> str:
    """Execute a bash command and return the output."""
    try:
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout} seconds."
    except Exception as e:
        return f"Error executing command: {e}"

    parts: list[str] = []

    # Combine stdout and stderr
    output = ""
    if result.stdout:
        output += result.stdout
    if result.stderr:
        if output and not output.endswith("\n"):
            output += "\n"
        output += result.stderr

    if output:
        truncated_output, was_truncated = _truncate(output)

        if was_truncated:
            # Save full output to temp file
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", prefix="puget-bash-",
                delete=False,
            )
            tmp.write(output)
            tmp.close()
            parts.append(f"[Output truncated. Full output saved to {tmp.name}]")

        parts.append(truncated_output)
    else:
        parts.append("(no output)")

    if result.returncode != 0:
        parts.append(f"\nExit code: {result.returncode}")

    return "\n".join(parts)


# -- Read tool ---------------------------------------------------------------

def _read(
    path: str,
    offset: int | None = None,
    limit: int | None = None,
) -> str:
    """Read a file's contents, optionally slicing by line."""
    filepath = Path(os.path.expanduser(path))

    if not filepath.exists():
        return f"Error: file not found: {path}"
    if filepath.is_dir():
        return f"Error: path is a directory: {path}"

    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"

    lines = text.splitlines(keepends=True)
    total_lines = len(lines)

    # Apply offset (1-indexed) and limit.
    if offset is not None:
        start = max(0, offset - 1)  # convert to 0-indexed
    else:
        start = 0

    if limit is not None:
        end = start + limit
    else:
        end = total_lines

    lines = lines[start:end]
    sliced = "".join(lines)

    result, was_truncated = _truncate(sliced)

    parts: list[str] = []
    if was_truncated:
        parts.append(f"[Output truncated. Showing {MAX_LINES} lines / {MAX_BYTES // 1024}KB of file.]")
    parts.append(result)

    # Show a hint about total size for context.
    actual_start = start + 1
    actual_end = min(start + len(lines), total_lines)
    parts.append(f"\n[Lines {actual_start}-{actual_end} of {total_lines} total]")

    return "".join(parts)


# -- Write tool --------------------------------------------------------------

def _write(path: str, content: str) -> str:
    """Write content to a file, creating parent directories as needed."""
    filepath = Path(os.path.expanduser(path))

    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")
    except Exception as e:
        return f"Error writing file: {e}"

    num_lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
    size = len(content.encode("utf-8"))
    return f"Wrote {num_lines} lines ({size} bytes) to {path}"


# -- Edit tool ---------------------------------------------------------------

def _edit(
    path: str,
    old_text: str | None = None,
    new_text: str | None = None,
    edits: list[dict[str, str]] | None = None,
) -> str:
    """Edit a file using exact text replacement.

    Two modes:
      1. Single edit: old_text + new_text
      2. Multi edit:  edits = [{oldText, newText}, ...]

    All matches are done against the original file content. Edits must not
    overlap.
    """
    filepath = Path(os.path.expanduser(path))

    if not filepath.exists():
        return f"Error: file not found: {path}"

    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file: {e}"

    # Normalize to a list of (old, new) pairs.
    if edits is not None:
        pairs = [(e["oldText"], e["newText"]) for e in edits]
    elif old_text is not None and new_text is not None:
        pairs = [(old_text, new_text)]
    else:
        return "Error: provide either oldText/newText or an edits array."

    # Validate: each oldText must appear exactly once.
    errors: list[str] = []
    for i, (old, _new) in enumerate(pairs):
        count = content.count(old)
        if count == 0:
            snippet = old[:60] + "..." if len(old) > 60 else old
            errors.append(f"Edit {i + 1}: oldText not found: {snippet!r}")
        elif count > 1:
            snippet = old[:60] + "..." if len(old) > 60 else old
            errors.append(f"Edit {i + 1}: oldText matches {count} times (must be unique): {snippet!r}")
    if errors:
        return "Error:\n" + "\n".join(errors)

    # Validate: no overlapping edits.
    # Find (start, end) of each match in the original content.
    regions: list[tuple[int, int, int]] = []
    for i, (old, _new) in enumerate(pairs):
        start = content.index(old)
        end = start + len(old)
        regions.append((start, end, i))

    # Sort by start position and check for overlaps.
    regions.sort()
    for j in range(1, len(regions)):
        if regions[j][0] < regions[j - 1][1]:
            return (
                f"Error: edits {regions[j - 1][2] + 1} and {regions[j][2] + 1} overlap."
            )

    # Apply edits from back to front (so positions stay valid).
    regions.sort(reverse=True)
    result = content
    for start, end, i in regions:
        _old, new = pairs[i]
        result = result[:start] + new + result[end:]

    try:
        filepath.write_text(result, encoding="utf-8")
    except Exception as e:
        return f"Error writing file: {e}"

    n = len(pairs)
    plural = "edit" if n == 1 else "edits"
    return f"Applied {n} {plural} to {path}"
