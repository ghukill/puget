"""puget CLI — a turn executor for LLM agents.

Three modes of operation:

  puget                    Interactive REPL. Start a new wave.
  puget "message"          One-shot: send a message, run to completion, exit.
  puget say "message"      Single-turn primitive (exit 0=text, 10=tool call).

Utility commands:

  puget resume [ID]        Resume a wave (most recent if no ID given).
  puget log                Print the current wave's history.
  puget new                Start a new wave.
  puget echo               Replay the last assistant response.
"""

import json
import sys
from pathlib import Path

import click

from puget import core, db
from puget import skills as skills_mod
from puget.output import console, err_console, print_assistant, print_log, print_thinking

EXIT_DONE = 0
EXIT_TOOL = 10
EXIT_ERROR = 1


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """puget — a CLI agent turn executor.

    Run with no arguments for the interactive REPL, or pass a message
    in quotes for one-shot mode.
    """
    if ctx.invoked_subcommand is None:
        from puget.repl import run_repl

        run_repl(resume=False)


@cli.command()
@click.argument("message")
def say(message):
    """Send a user message and get one model response.

    This is the single-turn primitive. It makes exactly one model call
    and exits. If the model responds with text, exit code is 0. If the
    model wants to call a tool, exit code is 10 and the tool call JSON
    is printed.

    Tool calls are NOT auto-executed. This is the raw building block
    for scripts and pipelines that want to control the loop themselves.
    For auto-execution, use one-shot mode: puget "message".
    """
    try:
        conn = db.connect()
        wid = db.new_wave(conn)

        with console.status("[dim]thinking…[/dim]", spinner="dots"):
            response = core.turn(conn, wid, message)

        print_thinking(response.get("thinking"))

        if response["tool_calls"]:
            if response["content"]:
                print_assistant(response["content"])
            for tc in response["tool_calls"]:
                click.echo(json.dumps(tc, indent=2))
            sys.exit(EXIT_TOOL)
        else:
            print_assistant(response["content"])
            sys.exit(EXIT_DONE)
    except SystemExit:
        raise
    except Exception as e:
        err_console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(EXIT_ERROR)


@cli.command(name="echo")
def echo_cmd():
    """Replay the last assistant response.

    Prints the most recent assistant response exactly as it originally
    appeared. Useful for reviewing what the model last said without
    scrolling through terminal history.

    Named for the Puget Sound — sound bouncing off the mountains.
    """
    try:
        conn = db.connect()
        wid = db.current_wave_id(conn)
        if wid is None:
            err_console.print("[dim]No wave yet.[/dim]")
            sys.exit(EXIT_ERROR)

        turn = db.last_assistant_turn(conn, wid)
        if turn is None:
            err_console.print("[dim]No assistant turn to echo.[/dim]")
            sys.exit(EXIT_ERROR)

        if turn["content"]:
            print_assistant(turn["content"])
        else:
            err_console.print("[dim]Last response had no text content.[/dim]")

        sys.exit(EXIT_DONE)
    except SystemExit:
        raise
    except Exception as e:
        err_console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(EXIT_ERROR)


@cli.command()
def log():
    """Print the current wave's history.

    Shows all turns in the most recent wave: user messages, assistant
    responses, and tool results, each with a timestamp.
    """
    try:
        conn = db.connect()
        wid = db.current_wave_id(conn)
        if wid is None:
            console.print("[dim]No wave yet.[/dim]")
            return

        turns = db.get_turns(conn, wid)
        print_log(turns)
    except Exception as e:
        err_console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(EXIT_ERROR)


@cli.command()
def new():
    """Start a new wave.

    Creates a fresh wave and makes it the current one.
    Subsequent commands will operate on this wave.
    """
    try:
        conn = db.connect()
        wid = db.new_wave(conn)
        console.print(f"[dim]New wave started (id: {wid})[/dim]")
    except Exception as e:
        err_console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(EXIT_ERROR)


@cli.command()
@click.argument("wave_id", required=False, type=int)
def resume(wave_id):
    """Resume a wave in the interactive REPL.

    With no WAVE_ID, resumes the most recent wave. With a specific ID,
    resumes that wave.

    \b
      puget resume         resume the most recent wave
      puget resume 42      resume wave 42
    """
    try:
        conn = db.connect()
        if wave_id is not None:
            # Validate the wave exists.
            row = conn.execute(
                "SELECT id FROM waves WHERE id = ?", (wave_id,)
            ).fetchone()
            if row is None:
                err_console.print(f"[bold red]Error:[/bold red] wave {wave_id} not found")
                sys.exit(EXIT_ERROR)
            wid = wave_id
        else:
            wid = db.current_wave_id(conn)
            if wid is None:
                err_console.print("[dim]No waves exist yet. Use 'puget new' to start one.[/dim]")
                sys.exit(EXIT_ERROR)

        from puget.repl import run_repl
        run_repl(resume=True, wave_id=wid)
    except SystemExit:
        raise
    except Exception as e:
        err_console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(EXIT_ERROR)


# -- Skill management --------------------------------------------------------

@cli.group()
def skill():
    """Manage puget skills.

    Install, list, and remove skills. Skills can live in the project
    (.puget/skills/) or globally ($PUGET_HOME/skills/).
    """


@skill.command()
@click.argument("source")
@click.option(
    "-g", "--global", "global_scope", is_flag=True,
    help="Install to $PUGET_HOME/skills/ (global, all projects).",
)
@click.option("--ref", help="Git ref (branch or tag) to checkout.")
@click.option("--path", "subpath", help="Path to skill directory within a git repository.")
def install(source, global_scope, ref, subpath):
    """Install a skill from a local path or git URL.

    SOURCE can be a local directory containing a SKILL.md, or a git URL.
    GitHub tree URLs are supported for pointing to a subdirectory:

    \b
      puget skill install ./path/to/my-skill
      puget skill install https://github.com/user/repo
      puget skill install https://github.com/user/repo/tree/main/skills/audit
      puget skill install https://github.com/user/repo --ref v1.0 --path skills/audit

    By default, skills are installed to the project (.puget/skills/).
    Use --global to install to $PUGET_HOME/skills/ for all projects.
    """
    try:
        target = skills_mod.install_target(global_scope=global_scope)
        name = skills_mod.install_skill(source, target, ref=ref, subpath=subpath)
        label = "$PUGET_HOME/skills/" if global_scope else ".puget/skills/"
        console.print(f"[green]✓[/green] Installed [bold]{name}[/bold] to {label}")
    except Exception as e:
        err_console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(EXIT_ERROR)


@skill.command(name="list")
@click.option(
    "-g", "--global", "global_scope", is_flag=True,
    help="Show only global skills.",
)
@click.option(
    "-p", "--project", "project_scope", is_flag=True,
    help="Show only project skills.",
)
def list_cmd(global_scope, project_scope):
    """List installed skills.

    Shows skills from all trait layers by default (project, global,
    system). Use --global or --project to filter to one layer.
    """
    try:
        if global_scope:
            layers = ["global"]
        elif project_scope:
            layers = ["project"]
        else:
            layers = None

        by_layer = skills_mod.list_by_layer(layers=layers)
        _print_skill_layers(by_layer)
    except Exception as e:
        err_console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(EXIT_ERROR)


@skill.command()
@click.argument("name")
@click.option(
    "-g", "--global", "global_scope", is_flag=True,
    help="Remove from $PUGET_HOME/skills/ (global).",
)
def remove(name, global_scope):
    """Remove an installed skill by NAME.

    By default, removes from the project (.puget/skills/).
    Use --global to remove from $PUGET_HOME/skills/.
    """
    try:
        target = skills_mod.install_target(global_scope=global_scope)
        skills_mod.remove_skill(name, target)
        label = "$PUGET_HOME/skills/" if global_scope else ".puget/skills/"
        console.print(f"[green]✓[/green] Removed [bold]{name}[/bold] from {label}")
    except Exception as e:
        err_console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(EXIT_ERROR)


def _print_skill_layers(by_layer: dict[str, list[dict[str, str]]]) -> None:
    """Pretty-print skills grouped by layer."""
    labels = {
        "project": "Project (.puget/skills/)",
        "global": f"Global ({skills_mod.install_target(global_scope=True)})",
        "system": "System (bundled)",
    }

    for layer, skills in by_layer.items():
        console.print(f"\n[bold]{labels.get(layer, layer)}[/bold]")
        if not skills:
            console.print("  [dim](none)[/dim]")
        else:
            for s in skills:
                desc = s["description"]
                if len(desc) > 60:
                    desc = desc[:57] + "..."
                console.print(f"  [cyan]{s['name']:<24}[/cyan] {desc}")
    console.print()


# -- Entry point -------------------------------------------------------------

# Known subcommand names. Used to distinguish one-shot messages from
# subcommand invocations in main().
_SUBCOMMANDS = {"say", "echo", "log", "new", "resume", "skill"}


def main():
    """Entry point for the puget CLI.

    Handles three modes of operation:

      puget                        → REPL (no arguments, handled by click group)
      puget "message"              → one-shot: new wave, run to completion
      puget <command>              → subcommand dispatch (say, log, new, resume, echo, skill)
      puget --skill <path> ...     → load an ephemeral skill for the session

    The --skill flag is consumed before click processing so it works in
    all modes (REPL, one-shot, and subcommands). Multiple --skill flags
    are supported.

    One-shot detection: if the first argument isn't a known subcommand
    or flag, everything after 'puget' is joined into a message and
    executed via core.run().
    """
    # Extract --skill flags from sys.argv before click/oneshot processing.
    # This ensures ephemeral skills work in all modes.
    raw_args = sys.argv[1:]
    remaining: list[str] = []
    skill_sources: list[str] = []
    i = 0
    while i < len(raw_args):
        if raw_args[i] == "--skill" and i + 1 < len(raw_args):
            skill_sources.append(raw_args[i + 1])
            i += 2
        else:
            remaining.append(raw_args[i])
            i += 1

    # Register ephemeral skills.
    for src in skill_sources:
        try:
            path = skills_mod.resolve_ephemeral_source(src)
            skills_mod.add_ephemeral_skill(path)
        except Exception as e:
            err_console.print(f"[bold red]Error:[/bold red] {e}")
            sys.exit(EXIT_ERROR)

    # Update sys.argv with --skill flags stripped.
    sys.argv = [sys.argv[0]] + remaining

    if remaining:
        args = remaining[:]

        if args and args[0] not in _SUBCOMMANDS and not args[0].startswith("-"):
            message = " ".join(args)
            _oneshot(message)
            return

    cli()


def _oneshot(message):
    """Run a one-shot message to completion.

    Sends the message, auto-executes any tool calls the model requests,
    and exits. Always starts a new wave.
    """
    try:
        conn = db.connect()
        wid = db.new_wave(conn)
        with console.status("[dim]thinking…[/dim]", spinner="dots"):
            core.run(conn, wid, message)
    except KeyboardInterrupt:
        err_console.print("\n[dim](interrupted)[/dim]")
        sys.exit(EXIT_ERROR)
    except Exception as e:
        err_console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(EXIT_ERROR)


if __name__ == "__main__":
    main()
