---
name: puget-understanding-self
description: Understand puget itself: its tools, state model, skill layering, and reflective analysis workflow. Use when the user asks how puget works, why it behaved a certain way, how to improve it, or when performing puget self-management tasks (skills, waves, config).
---

# puget-understanding-self

This skill provides a simple mental model for how puget works and how to reason about its behavior.

## Core model

puget is intentionally small and transparent.

- It has five tools. Its capabilities come from combining them well.
- Its capabilities also come from skills, layered by trait scope.
- Its durable conversation state lives in SQLite.

## Tools

puget has exactly five tools. This is the complete set:

- **`bash`** — Execute shell commands. The general-purpose escape hatch:
  filesystem exploration, running programs, installing packages, `sqlite3`
  queries, anything the shell can do.
- **`read`** — Read file contents with optional offset/limit paging.
  Use for examining code, configs, skill files, and logs.
- **`write`** — Write content to a file (create or overwrite).
  Creates parent directories automatically.
- **`edit`** — Precise text replacement in existing files.
  Single or multiple disjoint edits, matched against original content.
- **`config`** — Return puget's resolved runtime state as JSON.
  Concrete paths, active model, current wave, skill dirs, settings.
  No arguments. Call this instead of guessing.

Tool definitions live in `puget/tools.py` (`TOOL_DEFINITIONS` list).
Human-readable descriptions live in `puget/prompt.py` (`SYSTEM_PROMPT`).

## Skills

Skills extend puget's capabilities without changing its core. They are
plain files on disk, layered by scope (first match wins):

1. `system/skills/` — built-in, versioned with puget
2. `~/.puget/skills/` — global user skills
3. `./.puget/skills/` — project-specific skills

Skill discovery, installation, and prompt injection are handled by
`puget/skills.py`.

## CLI

puget has its own CLI for managing state and skills. **Prefer these
commands (via `bash`) over raw file or database manipulation.**

This is the same principle as using `git` instead of editing `.git/`
internals — the CLI handles validation, cleanup, and consistency.

### Top-level commands

- `puget` — interactive REPL (the normal way to use puget)
- `puget say "<message>"` — one-shot: send a message, get one response
- `puget new` — start a new wave
- `puget log` — print the current wave's conversation history
- `puget echo` — replay the last assistant response
- `puget -r` / `puget --resume` — resume the most recent wave

### Skill management

- `puget skill list` — list skills across all trait layers
- `puget skill install <source>` — install from a local path or git URL
- `puget skill remove <name>` — remove a skill by name
- Add `--global` / `-g` to target `$PUGET_HOME/skills/` instead of `.puget/skills/`

When the user asks you to install, remove, or manage skills, use these
commands. Do not manually `rm -rf` skill directories or hand-edit the
database.

## State

Conversation state lives in SQLite (tables: `waves`, `turns`).
Use `config` for the concrete DB path.

## The `config` tool

The `config` tool is the starting point for any self-understanding task.
Call it first. It returns resolved, concrete values — no guessing needed:

- Database path and whether it exists
- Active model and Ollama host
- Current wave ID and turn count
- Skill directories and which exist
- All runtime settings

Use these concrete values in subsequent tool calls (e.g., pass the
resolved `db_path` to `sqlite3`).

## What "understanding self" means

When asked to explain behavior, debug decisions, or improve puget, you can:

1. Call `config` to ground yourself in concrete runtime state.
2. Inspect code and prompts to understand intended behavior.
3. Inspect skills to understand learned workflows.
4. Inspect SQLite wave/turn history to understand actual behavior.
5. Compare intent vs. reality, then suggest focused improvements.

## How to apply this skill

If a user says "you", they are likely referring to puget, the agent (you).

Use it when the user asks things like:

- "Why did puget do that?"
- "Is the agent aware of X?"
- "How does puget actually work?"
- "How can we improve puget's behavior?"
- "Should this be prompt guidance or a skill?"

## Minimal self-analysis workflow

When the user asks for wave/turn analysis:

1. Call `config` to get `db_path`, `current_wave_id`, and `current_wave_turn_count`.
2. If db exists, run a focused `sqlite3` query with `LIMIT`.
3. Summarize what you observed, and clearly note uncertainty.

Example (after getting db_path from config):

```bash
sqlite3 -header -column "/path/from/config/puget.db" \
  "SELECT id, wave_id, role, substr(content,1,120) AS preview, created_at FROM turns ORDER BY id DESC LIMIT 10;"
```

## Guardrails

- Start with read-only introspection.
- Call `config` before guessing at paths or env vars.
- Be explicit about what is observed vs inferred.
- Prefer small, composable changes over heavy framework additions.
- Keep puget's philosophy intact: simple tools, transparent state, layered skills.
