"""Skill discovery and prompt injection for puget.

Skills are self-contained capability packages that teach the model
specialized workflows. A skill is a directory containing a SKILL.md
file with YAML frontmatter (name, description) followed by markdown
instructions.

The design follows pi's progressive disclosure approach:
  1. At startup, scan skill directories for SKILL.md files.
  2. Parse only the frontmatter (name + description).
  3. Inject a compact listing into the system prompt.
  4. The model loads full skill content on demand via bash (cat).

This means skill instructions are never in context until the model
decides they're relevant. Only the name and one-line description
occupy prompt space.

Skill locations (trait layer order):
  - <puget package>/system/skills/  System skills (bundled with puget, always present).
  - $PUGET_HOME/skills/              Global (user-level) skills (default: ~/.puget/skills/).
  - .puget/skills/                   Project-level skills (relative to cwd).

A skill directory looks like:
  my-skill/
  ├── SKILL.md          # Required: frontmatter + instructions
  ├── scripts/          # Optional: helper scripts
  └── references/       # Optional: docs the model can read
"""

from pathlib import Path


# -- Discovery ---------------------------------------------------------------

def _system_skills_dir() -> Path:
    """Return the system trait layer's skills directory.

    This is the system/skills/ directory bundled with the puget package.
    Resolved relative to this file's location in the installed package.
    """
    return Path(__file__).resolve().parent.parent / "system" / "skills"


def skill_dirs() -> list[Path]:
    """Return the directories to scan for skills.

    Three trait layers are checked, project-first so narrower scopes
    shadow broader ones (first skill found with a given name wins):
      - .puget/skills/                  Project-specific skills (relative to cwd).
      - $PUGET_HOME/skills/             Global user skills (default: ~/.puget/skills/).
      - <puget package>/system/skills/  System skills (bundled, always present).

    Directories that don't exist are included in the list but skipped
    during scanning. This keeps the logic simple and the locations
    predictable.
    """
    from puget import puget_home
    return [
        Path.cwd() / ".puget" / "skills",
        puget_home() / "skills",
        _system_skills_dir(),
    ]


def discover(search_dirs: list[Path] | None = None) -> list[dict[str, str]]:
    """Scan skill directories and return all discovered skills.

    Looks for subdirectories containing a SKILL.md file, parses the
    frontmatter for name and description, and returns a list of skill
    metadata dicts. Skills without a description are skipped (the
    description is what tells the model when to use the skill).

    First skill found with a given name wins — duplicates are silently
    skipped. Trait layers are scanned project → global → system, so
    project skills shadow global, and global shadows system.

    Args:
        search_dirs: Override the default skill directories. Useful for
                     testing without touching the real filesystem.

    Returns:
        List of dicts, each with:
          - name: Skill identifier (from frontmatter or directory name).
          - description: What the skill does and when to use it.
          - file_path: Absolute path to the SKILL.md file.
          - base_dir: Absolute path to the skill directory.
    """
    skills: list[dict[str, str]] = []
    seen: set[str] = set()

    for search_dir in (search_dirs or skill_dirs()):
        if not search_dir.is_dir():
            continue

        for entry in sorted(search_dir.iterdir()):
            if not entry.is_dir():
                continue

            skill_file = entry / "SKILL.md"
            if not skill_file.is_file():
                continue

            try:
                content = skill_file.read_text()
            except OSError:
                continue

            frontmatter = parse_frontmatter(content)
            name = frontmatter.get("name", entry.name)
            description = frontmatter.get("description", "")

            if not description.strip():
                continue

            if name in seen:
                continue

            seen.add(name)
            skills.append({
                "name": name,
                "description": description,
                "file_path": str(skill_file),
                "base_dir": str(entry),
            })

    return skills


# -- Prompt formatting -------------------------------------------------------

def format_for_prompt(skills: list[dict[str, str]]) -> str:
    """Format discovered skills as an XML block for the system prompt.

    Produces an <available_skills> block compatible with pi's Agent Skills
    convention. Each skill entry includes its name, description, and file
    path so the model can load the full content when needed.

    The preamble instructs the model to use `cat` (via bash) to read
    skill files — puget doesn't have a dedicated read tool, but bash
    handles it fine.

    Returns an empty string if no skills are available.
    """
    if not skills:
        return ""

    lines = [
        "",
        "The following skills provide specialized instructions for specific tasks.",
        "Use `bash` with `cat` to load a skill's file when the task matches its description.",
        "When a skill file references a relative path, resolve it against the skill's",
        "base directory (the parent of SKILL.md).",
        "",
        "<available_skills>",
    ]

    for skill in skills:
        lines.append("  <skill>")
        lines.append(f"    <name>{skill['name']}</name>")
        lines.append(f"    <description>{skill['description']}</description>")
        lines.append(f"    <location>{skill['file_path']}</location>")
        lines.append("  </skill>")

    lines.append("</available_skills>")
    return "\n".join(lines)


# -- Frontmatter parsing ----------------------------------------------------

def parse_frontmatter(text: str) -> dict[str, str]:
    """Parse YAML-like frontmatter from a SKILL.md file.

    Expects the file to start with a --- line, followed by key: value
    pairs, closed by another --- line. Handles simple single-line values
    only (which is all skill frontmatter needs).

    This is intentionally minimal — no YAML library dependency for what
    amounts to parsing 2-3 key-value pairs.

    Returns:
        Dict of frontmatter fields. Empty dict if no valid frontmatter found.
    """
    if not text.startswith("---"):
        return {}

    end = text.find("\n---", 3)
    if end == -1:
        return {}

    block = text[4:end].strip()
    result: dict[str, str] = {}

    for line in block.split("\n"):
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if key and value:
            result[key] = value

    return result
