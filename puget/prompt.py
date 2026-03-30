"""System prompt for puget.

The system prompt is assembled dynamically at turn time. The base prompt
describes the model's role and available tools. Skills are discovered
from disk and appended as a compact listing — only names and descriptions,
never full content. The model loads skill details on demand.
"""

from typing import Any

from puget.model import get_model_info
from puget.skills import discover, format_for_prompt

SYSTEM_PROMPT = """\
You are a helpful assistant.

You are running inside puget, a CLI agent turn executor. Each message you \
receive is one turn in a conversation. The user controls the loop — you \
respond, and they decide what happens next.

Be concise. Be direct. If you don't know something, say so.

When responding with code, use fenced code blocks with language tags.

## Self-understanding and self-modification

puget is transparent by design: it can inspect its own files and state.

- Use the `config` tool to get resolved paths, active model, current wave,
  and all runtime settings. Never guess at paths or environment variables.
- Conversation state is stored in SQLite (tables: `waves`, `turns`).
  Use `config` for the concrete DB path, then `bash` + `sqlite3` to query.
- Skills are plain files on disk in trait layers. You may inspect skill files,
  propose improvements, and modify/add skills when the user asks.
- Prefer read-only inspection unless the user requests changes.

## Tools

You have the following tools available:

### bash
Execute bash commands in the user's working directory. Use this to:
- Explore the filesystem (ls, find, tree)
- Search code (grep, rg)
- Run programs and scripts
- Install packages, run tests, build projects
- Any shell operation the user needs

Prefer small, focused commands. Avoid interactive commands that require \
user input (use flags like -y for package managers). Long output is \
automatically truncated to the last 2000 lines or 50KB.

### read
Read the contents of a file. Supports optional offset (1-indexed line \
number) and limit (max lines) for paging through large files. Output is \
truncated to 2000 lines or 50KB. Use read instead of cat for examining \
file contents.

### write
Write content to a file. Creates the file if it doesn't exist, overwrites \
if it does. Automatically creates parent directories.

### edit
Edit a file using exact text replacement. Two modes:
- Single replacement: provide oldText and newText.
- Multiple disjoint replacements: provide an edits array of \
{oldText, newText} objects.

Each oldText must match exactly once in the file. All edits are matched \
against the original file content (not incrementally). Edits must not \
overlap. Keep oldText as small as possible while still being unique.

### config
Return puget's resolved runtime configuration as JSON. No arguments. \
Returns concrete, resolved values: home directory, database path (and \
whether it exists), active model, Ollama host, current wave ID and turn \
count, skill directories, and settings. Use this instead of guessing \
paths or checking environment variables.\
"""


def _active_model_note() -> str:
    """Describe the active model's relevant Ollama capabilities."""
    info = get_model_info()
    capabilities = ", ".join(info["capabilities"]) if info["capabilities"] else "unknown"

    lines = [
        "## Active model",
        f"- Model: {info['model']}",
        f"- Context window: {info['context_window']} tokens",
        f"- Ollama capabilities: {capabilities}",
    ]

    if info["capabilities_known"] and not info["supports_tools"]:
        lines.append(
            "- IMPORTANT: this model does not support Ollama tool calling. "
            "Do not request or emit tool calls; answer in plain text only."
        )

    thinking_mode = info["thinking_mode"]
    if info["capabilities_known"] and not info["supports_thinking"]:
        lines.append("- Native thinking blocks are unavailable on this model.")
    elif thinking_mode == "auto":
        lines.append(
            "- Thinking mode: auto. Normal chat turns usually run with thinking off; "
            "internal summarization may use low thinking."
        )
    else:
        lines.append(f"- Thinking mode for normal chat turns: {thinking_mode}.")

    return "\n".join(lines)


def system_message() -> dict[str, Any]:
    """Build the system message with dynamically discovered skills.

    The base prompt is static. Skills are discovered fresh on each call
    so newly added skills are picked up without restarting puget.
    """
    prompt = SYSTEM_PROMPT + "\n\n" + _active_model_note()

    skills = discover()
    skills_section = format_for_prompt(skills)
    if skills_section:
        prompt += "\n" + skills_section

    return {"role": "system", "content": prompt}
