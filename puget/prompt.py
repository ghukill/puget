"""System prompt for puget.

The system prompt is assembled dynamically at turn time. The base prompt
describes the model's role and available tools. Skills are discovered
from disk and appended as a compact listing — only names and descriptions,
never full content. The model loads skill details on demand.
"""

from typing import Any

from puget.skills import discover, format_for_prompt

SYSTEM_PROMPT = """\
You are a helpful assistant.

You are running inside puget, a CLI agent turn executor. Each message you \
receive is one turn in a conversation. The user controls the loop — you \
respond, and they decide what happens next.

Be concise. Be direct. If you don't know something, say so.

When responding with code, use fenced code blocks with language tags.

## Tools

You have the following tools available:

### bash
Execute bash commands in the user's working directory. Use this to:
- Explore the filesystem (ls, find, tree)
- Read files (cat, head, tail, grep)
- Run programs and scripts
- Install packages, run tests, build projects
- Any shell operation the user needs

Prefer small, focused commands. Avoid interactive commands that require \
user input (use flags like -y for package managers). Long output is \
automatically truncated to the last 2000 lines or 50KB.\
"""


def system_message() -> dict[str, Any]:
    """Build the system message with dynamically discovered skills.

    The base prompt is static. Skills are discovered fresh on each call
    so newly added skills are picked up without restarting puget.
    """
    prompt = SYSTEM_PROMPT

    skills = discover()
    skills_section = format_for_prompt(skills)
    if skills_section:
        prompt += "\n" + skills_section

    return {"role": "system", "content": prompt}
