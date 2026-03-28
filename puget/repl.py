"""Interactive REPL for puget.

A prompt loop that feeds user input directly to core.run(). No subprocess
orchestration, no shelling out to the CLI — just direct Python calls to
the core engine.

The REPL resumes the most recent wave if one exists, otherwise starts
fresh. Tool calls are auto-executed by the core; the REPL just waits
for run() to complete and prompts for the next input.
"""

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from rich.rule import Rule

from puget import core, db
from puget.model import get_model
from puget.output import console, print_log, set_show_thinking, show_thinking


def run_repl(*, resume: bool = False):
    """Run the interactive REPL.

    With resume=True, continues the most recent wave. Otherwise starts
    a new wave. Each user input is sent through core.run(), which
    handles tool execution and model interaction to completion before
    returning control to the prompt.

    Slash commands:
      /new            — start a new wave
      /log            — print wave history
      /thinking on|off — toggle display of model thinking
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
    thinking_status = "on" if show_thinking() else "off"
    console.print(f"[dim]  Enter sends • Esc+Enter for newline • /new /log /thinking /quit[/dim]")
    console.print(f"[dim]  thinking: {thinking_status}[/dim]")
    console.print()

    kb = _build_key_bindings()
    session = PromptSession(key_bindings=kb, multiline=True)

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
            cmd = text.lower()
            if cmd in ("/quit", "/exit", "/q"):
                console.print("[dim]Goodbye![/dim]")
                break
            elif cmd == "/new":
                wid = db.new_wave(conn)
                console.print(f"[dim]New wave (id: {wid})[/dim]")
                continue
            elif cmd == "/log":
                turns = db.get_turns(conn, wid)
                print_log(turns)
                continue
            elif cmd.startswith("/thinking"):
                _handle_thinking_cmd(cmd)
                continue
            elif cmd == "/help":
                _print_help()
                continue
            # Unrecognized slash command — treat as a normal message.

        console.print()

        try:
            core.run(conn, wid, text)
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


def _handle_thinking_cmd(cmd: str) -> None:
    """Handle /thinking [on|off] command."""
    parts = cmd.split()
    if len(parts) == 1:
        # Just "/thinking" — show current status.
        status = "on" if show_thinking() else "off"
        console.print(f"[dim]thinking: {status}[/dim]")
    elif parts[1] == "on":
        set_show_thinking(True)
        console.print("[dim]thinking: on[/dim]")
    elif parts[1] == "off":
        set_show_thinking(False)
        console.print("[dim]thinking: off[/dim]")
    else:
        console.print("[dim]Usage: /thinking [on|off][/dim]")


def _print_help():
    """Print available REPL commands."""
    console.print()
    console.print("[bold]Commands:[/bold]")
    console.print("  /new            — start a new wave")
    console.print("  /log            — print wave history")
    console.print("  /thinking on|off — toggle model thinking display")
    console.print("  /help           — show this help")
    console.print("  /quit           — exit")
    console.print()
