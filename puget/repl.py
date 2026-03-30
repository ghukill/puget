"""Interactive REPL for puget.

A prompt loop that feeds user input directly to core.run(). No subprocess
orchestration, no shelling out to the CLI — just direct Python calls to
the core engine.

The REPL resumes the most recent wave if one exists, otherwise starts
fresh. Tool calls are auto-executed by the core; the REPL just waits
for run() to complete and prompts for the next input.
"""

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from rich.rule import Rule

from puget import core, db
from puget.model import (
    get_model,
    get_model_info,
    get_thinking_mode,
    list_available_models,
    set_model,
    set_thinking_mode,
)
from puget.output import console, print_log, set_show_thinking, show_thinking
from puget.skills import discover


def run_repl(*, resume: bool = False):
    """Run the interactive REPL.

    With resume=True, continues the most recent wave. Otherwise starts
    a new wave. Each user input is sent through core.run(), which
    handles tool execution and model interaction to completion before
    returning control to the prompt.

    Slash commands:
      /new            — start a new wave
      /log            — print wave history
      /thinking off|low|on|auto — set the thinking policy
      /help           — show available commands
      /quit           — exit
    """
    conn = db.connect()

    resuming = False
    if resume:
        wid = db.current_wave_id(conn)
        if wid is not None:
            resuming = True
        else:
            wid = db.new_wave(conn)
    else:
        wid = db.new_wave(conn)

    model_name = get_model()

    console.print()
    label = f"resuming wave #{wid}" if resuming else "new wave"
    console.print(Rule(f"[bold]puget[/bold]  [dim]model: {model_name} • {label}[/dim]"))
    console.print("[dim]  Enter sends • Esc+Enter for newline • /new /log /thinking /quit[/dim]")
    console.print(f"[dim]  {_thinking_status_text()}[/dim]")
    console.print()

    kb = _build_key_bindings()
    skills = discover()
    completer = _SlashCompleter(skills=skills)
    def _bottom_toolbar():
        return HTML(f'<style bg="" fg="gray"> model: {get_model()} </style>')

    session = PromptSession(
        key_bindings=kb,
        multiline=True,
        completer=completer,
        complete_while_typing=False,
        bottom_toolbar=_bottom_toolbar,
    )

    while True:
        try:
            text = session.prompt("❯ ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        text = text.strip()
        if not text:
            continue

        # Slash commands.
        if text.startswith("/"):
            lowered = text.lower()
            if lowered in ("/quit", "/exit", "/q"):
                console.print("[dim]Goodbye![/dim]")
                break
            elif lowered == "/new":
                wid = db.new_wave(conn)
                console.print(f"[dim]New wave (id: {wid})[/dim]")
                continue
            elif lowered == "/log":
                turns = db.get_turns(conn, wid)
                print_log(turns)
                continue
            elif lowered.startswith("/thinking"):
                _handle_thinking_cmd(text)
                continue
            elif lowered.startswith("/model"):
                _handle_model_cmd(text)
                continue
            elif lowered == "/help":
                _print_help()
                continue
            # Unrecognized slash command — treat as a normal message.

        console.print()

        try:
            response = core.run(conn, wid, text)
            wid = response.get("wave_id", wid)
        except KeyboardInterrupt:
            console.print("\n[dim](interrupted)[/dim]")
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")

        console.print()


def _build_key_bindings():
    """Build key bindings for the prompt.

    Enter sends the current input. Esc+Enter inserts a newline for
    multi-line messages.
    """
    kb = KeyBindings()

    @kb.add(Keys.Enter)
    def _(event):
        event.current_buffer.validate_and_handle()

    @kb.add(Keys.Escape, Keys.Enter)
    def _(event):
        event.current_buffer.insert_text("\n")

    return kb


# -- Autocomplete ------------------------------------------------------------

# Slash commands with descriptions for the completion menu.
_SLASH_COMMANDS = {
    "/help": "show available commands",
    "/log": "print wave history",
    "/model": "show models or switch the active model",
    "/new": "start a new wave",
    "/quit": "exit",
    "/thinking": "set the model thinking policy",
}

_THINKING_ARGS = {
    "off": "disable Ollama thinking for normal chat turns",
    "low": "request low thinking for normal chat turns",
    "on": "request full thinking for normal chat turns",
    "auto": "chat off, internal summarization low",
}


class _SlashCompleter(Completer):
    """Tab-completer for slash commands and skill names.

    Only offers completions when the input starts with '/'. Skills
    are included as slash commands — selecting one sends the skill
    name to the model as a regular message (falls through the
    unrecognized-command path).
    """

    def __init__(self, skills: list[dict[str, str]] | None = None):
        self._skills = skills or []
        self._model_names: list[str] | None = None

    def _get_model_names(self) -> list[str]:
        """Load and cache model names for /model completion."""
        if self._model_names is None:
            self._model_names = list_available_models()
        return self._model_names

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        if not text.startswith("/"):
            return

        parts = text.split()

        # Sub-argument completion: "/thinking on|off"
        if parts[0] == "/thinking":
            if len(parts) == 2:
                prefix = parts[1]
            elif len(parts) == 1 and text.endswith(" "):
                prefix = ""
            else:
                return
            for arg, desc in _THINKING_ARGS.items():
                if arg.startswith(prefix):
                    yield Completion(arg, start_position=-len(prefix), display_meta=desc)
            return

        # Sub-argument completion: "/model <name>"
        if parts[0] == "/model":
            if len(parts) == 2:
                prefix = parts[1]
            elif len(parts) == 1 and text.endswith(" "):
                prefix = ""
            else:
                return

            for name in self._get_model_names():
                if name.startswith(prefix):
                    yield Completion(name, start_position=-len(prefix), display_meta="available model")
            return

        # Top-level: complete the slash command itself.
        # Only when still typing the first word.
        if len(parts) > 1 or text.endswith(" "):
            return

        prefix = text
        for cmd, desc in _SLASH_COMMANDS.items():
            if cmd.startswith(prefix):
                yield Completion(cmd, start_position=-len(prefix), display_meta=desc)

        for skill in self._skills:
            skill_cmd = f"/{skill['name']}"
            if skill_cmd.startswith(prefix):
                yield Completion(
                    skill_cmd,
                    start_position=-len(prefix),
                    display_meta=skill["description"],
                )


# -- Slash command handlers --------------------------------------------------


def _thinking_status_text() -> str:
    """Return a compact status line for thinking policy + display."""
    display = "on" if show_thinking() else "off"
    return f"thinking: {get_thinking_mode()} • display: {display}"



def _handle_thinking_cmd(cmd: str) -> None:
    """Handle /thinking [off|low|on|auto] command."""
    parts = cmd.split()
    if len(parts) == 1:
        console.print(f"[dim]{_thinking_status_text()}[/dim]")
        return

    mode = parts[1].strip().lower()
    if mode not in _THINKING_ARGS:
        console.print("[dim]Usage: /thinking [off|low|on|auto][/dim]")
        return

    set_thinking_mode(mode)
    set_show_thinking(mode != "off")
    console.print(f"[dim]{_thinking_status_text()}[/dim]")


def _handle_model_cmd(cmd: str) -> None:
    """Handle /model [name|number] command."""
    parts = cmd.split(None, 1)
    available = list_available_models()

    if len(parts) == 1:
        # Just "/model" — show current model plus selectable list.
        current = get_model()
        info = get_model_info(current)
        caps = ", ".join(info["capabilities"]) if info["capabilities"] else "unknown"
        console.print(f"[dim]model: {current}[/dim]")
        console.print(f"[dim]capabilities: {caps} • context: {info['context_window']}[/dim]")

        if available:
            console.print("[dim]available models:[/dim]")
            for i, name in enumerate(available, start=1):
                active = " (active)" if name == current else ""
                console.print(f"[dim]  {i:>2}. {name}{active}[/dim]")
            console.print("[dim]use /model <name|number> to switch[/dim]")
        else:
            console.print("[dim]no models discovered from Ollama[/dim]")
        return

    selection = parts[1].strip()
    name = selection

    if selection.isdigit() and available:
        idx = int(selection)
        if 1 <= idx <= len(available):
            name = available[idx - 1]
        else:
            console.print(f"[dim]invalid model number: {selection}[/dim]")
            return

    set_model(name)
    info = get_model_info()
    caps = ", ".join(info["capabilities"]) if info["capabilities"] else "unknown"
    console.print(f"[dim]model: {get_model()}[/dim]")
    console.print(f"[dim]capabilities: {caps} • context: {info['context_window']}[/dim]")


def _print_help():
    """Print available REPL commands."""
    console.print()
    console.print("[bold]Commands:[/bold]")
    console.print("  /new              — start a new wave")
    console.print("  /log              — print wave history")
    console.print("  /model [name|#]   — show models or switch the active model")
    console.print("  /thinking [off|low|on|auto] — set the model thinking policy")
    console.print("  /help             — show this help")
    console.print("  /quit             — exit")
    console.print()
