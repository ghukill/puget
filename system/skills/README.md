# System Skills

Skills in this directory ship with puget and are always available.

This is puget's **system trait layer** — the foundational layer in the trait
layer stack:

1. `system/skills/` — built-in, versioned with puget, read-only at runtime
2. `~/.puget/skills/` — global user skills
3. `./.puget/skills/` — project-specific skills

Each trait layer can add skills. Project overrides global, global overrides system.

Bundled system skills:

- `puget-understanding-self/` — high-level self-understanding guide for puget's tools, state, skills, and reflective analysis.
