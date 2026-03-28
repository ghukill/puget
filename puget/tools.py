"""Built-in tools for puget.

Minimal tool set. The bash tool does the heavy lifting.
"""

import subprocess
import tempfile
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
]


# -- Dispatch ----------------------------------------------------------------

def execute(name: str, arguments: dict[str, Any]) -> str:
    """Execute a tool by name. Returns the result as a string."""
    if name == "bash":
        return _bash(
            command=arguments["command"],
            timeout=arguments.get("timeout"),
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
