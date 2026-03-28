"""puget CLI — a turn executor for LLM agents.

Three modes of operation:

  puget                    Interactive REPL. Resume or start a wave.
  puget "message"          One-shot: send a message, run to completion, exit.
  puget say "message"      Single-turn primitive (exit 0=text, 10=tool call).

Utility commands:

  puget log                Print the current wave's history.
  puget new                Start a new wave.
  puget echo               Replay the last assistant response.
"""

import json
import sys

import click

from puget import core, db
from puget.output import console, err_console, print_assistant, print_log, print_thinking

EXIT_DONE = 0
EXIT_TOOL = 10
EXIT_ERROR = 1


@click.group(invoke_without_command=True)
@click.option("-r", "--resume", is_flag=True, help="Resume the most recent wave.")
@click.pass_context
def cli(ctx, resume):
    """puget — a CLI agent turn executor.

    Run with no arguments for the interactive REPL, or pass a message
    in quotes for one-shot mode.
    """
    if ctx.invoked_subcommand is None:
        from puget.repl import run_repl

        run_repl(resume=resume)


@cli.command()
@click.argument("message")
@click.option("-r", "--resume", is_flag=True, help="Resume the most recent wave.")
def say(message, resume):
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
        wid = db.ensure_wave(conn) if resume else db.new_wave(conn)

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


# -- Entry point -------------------------------------------------------------

# Known subcommand names. Used to distinguish one-shot messages from
# subcommand invocations in main().
_SUBCOMMANDS = {"say", "echo", "log", "new"}


def main():
    """Entry point for the puget CLI.

    Handles three modes of operation:

      puget                        → REPL (no arguments, handled by click group)
      puget "message"              → one-shot: new wave, run to completion
      puget -r "message"           → one-shot: resume most recent wave
      puget --resume "message"     → same as -r
      puget <command>              → subcommand dispatch (say, log, new, echo)

    One-shot detection: if the first argument isn't a known subcommand
    or flag, everything after 'puget' is joined into a message and
    executed via core.run(). The -r/--resume flag is consumed first.
    """
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        resume = False

        # Consume -r / --resume before one-shot detection.
        if args and args[0] in ("-r", "--resume"):
            resume = True
            args = args[1:]

        if args and args[0] not in _SUBCOMMANDS and not args[0].startswith("-"):
            message = " ".join(args)
            _oneshot(message, resume=resume)
            return

    cli()


def _oneshot(message, *, resume=False):
    """Run a one-shot message to completion.

    Sends the message, auto-executes any tool calls the model requests,
    and exits. With resume=True, continues the most recent wave instead
    of starting a new one.
    """
    try:
        conn = db.connect()
        wid = db.ensure_wave(conn) if resume else db.new_wave(conn)
        core.run(conn, wid, message)
    except KeyboardInterrupt:
        err_console.print("\n[dim](interrupted)[/dim]")
        sys.exit(EXIT_ERROR)
    except Exception as e:
        err_console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(EXIT_ERROR)


if __name__ == "__main__":
    main()
