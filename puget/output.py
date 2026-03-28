"""Shared output rendering for puget.

All terminal output goes through this module. Rich handles the formatting:
assistant responses get markdown rendering, wave logs get role-colored
labels, and everything else gets sensible defaults.

Two consoles are exposed:
  - console: writes to stdout (for primary output).
  - err_console: writes to stderr (for status messages, tool summaries).
"""

import json
from typing import Any

from rich.console import Console
from rich.markdown import Markdown

console = Console()
err_console = Console(stderr=True)


def print_assistant(content: str) -> None:
    """Render an assistant response as markdown.

    The response is parsed as markdown and printed with a blank line
    above and below for breathing room. Empty or whitespace-only
    content is silently skipped.
    """
    if content and content.strip():
        console.print()
        console.print(Markdown(content))
        console.print()


def print_log(turns: list[dict[str, Any]]) -> None:
    """Render a list of wave turns as a readable log.

    Each turn is displayed with a role-colored label and timestamp.
    Assistant turns with tool_calls show their structured data from
    the dedicated column — no JSON sniffing needed.

    Args:
        turns: List of turn dicts from db.get_turns(), each with
               role, content, tool_calls, and created_at keys.
    """
    if not turns:
        console.print("[dim]Empty wave.[/dim]")
        return

    role_styles: dict[str, tuple[str, str]] = {
        "user": ("bold green", "▶ user"),
        "assistant": ("bold blue", "◀ assistant"),
        "tool": ("bold yellow", "⚡ tool"),
    }

    for t in turns:
        style, label = role_styles.get(t["role"], ("dim", t["role"]))
        console.print(f"\n[{style}]{label}[/{style}]  [dim]{t['created_at']}[/dim]")

        if t["role"] == "assistant":
            if t["content"]:
                console.print(Markdown(t["content"]))
            if t["tool_calls"]:
                for tc in json.loads(t["tool_calls"]):
                    console.print(f"  [yellow]tool_call:[/yellow] {json.dumps(tc, indent=2)}")
        else:
            console.print(t["content"])

    console.print()
