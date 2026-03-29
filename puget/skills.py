"""Skill discovery, prompt injection, and management for puget.

Skills are self-contained capability packages that teach the model
specialized workflows. A skill is a directory containing a SKILL.md
file with YAML frontmatter (name, description) followed by markdown
instructions.

Discovery (prompt injection):
  1. At startup, scan skill directories for SKILL.md files.
  2. Parse only the frontmatter (name + description).
  3. Inject a compact listing into the system prompt.
  4. The model loads full skill content on demand via bash (cat).

Management:
  - install_skill()       Install from local path or git URL.
  - remove_skill()        Remove an installed skill.
  - list_by_layer()       List skills grouped by trait layer.
  - add_ephemeral_skill() Load a skill for the current session only.

Skill locations (trait layer order, project-first):
  - (ephemeral)                      --skill flag (session only, highest priority).
  - .puget/skills/                   Project-level skills (relative to cwd).
  - $PUGET_HOME/skills/              Global (user-level) skills.
  - <puget package>/system/skills/   System skills (bundled with puget).

A skill directory looks like:
  my-skill/
  ├── SKILL.md          # Required: frontmatter + instructions
  ├── scripts/          # Optional: helper scripts
  └── references/       # Optional: docs the model can read
"""

import atexit
import re
import shutil
import subprocess as _subprocess
import tempfile
from pathlib import Path


# -- Ephemeral skill management -----------------------------------------------

_ephemeral_skill_paths: list[Path] = []
_temp_dirs: list[Path] = []


def _cleanup_temp_dirs() -> None:
    """Remove temp directories created for ephemeral git skills."""
    for d in _temp_dirs:
        shutil.rmtree(d, ignore_errors=True)


atexit.register(_cleanup_temp_dirs)


def add_ephemeral_skill(path: Path) -> None:
    """Register a skill directory for ephemeral (session-only) loading.

    The path must point to a directory containing a SKILL.md file.
    Ephemeral skills take highest priority in discovery — they shadow
    installed skills with the same name.
    """
    _ephemeral_skill_paths.append(path.resolve())


def clear_ephemeral_skills() -> None:
    """Clear all ephemeral skills. Useful for testing."""
    _ephemeral_skill_paths.clear()


def resolve_ephemeral_source(source: str) -> Path:
    """Resolve a --skill source (local path or URL) to a skill directory.

    For local paths, validates that SKILL.md exists and returns the
    resolved path. For git URLs, clones to a temp directory that is
    cleaned up automatically when puget exits.

    Args:
        source: Local directory path or git URL.

    Returns:
        Path to the skill directory containing SKILL.md.

    Raises:
        FileNotFoundError: If SKILL.md is not found.
        RuntimeError: If git clone fails.
    """
    if source.startswith(("https://", "http://", "git@")):
        clone_url, ref, subpath = _parse_git_source(source)

        tmp = Path(tempfile.mkdtemp(prefix="puget-skill-"))
        _temp_dirs.append(tmp)

        try:
            _git_clone(clone_url, tmp / "repo", ref=ref)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            _temp_dirs.remove(tmp)
            raise

        skill_dir = tmp / "repo"
        if subpath:
            skill_dir = skill_dir / subpath

        if not (skill_dir / "SKILL.md").is_file():
            shutil.rmtree(tmp, ignore_errors=True)
            _temp_dirs.remove(tmp)
            location = subpath or "repository root"
            raise FileNotFoundError(f"No SKILL.md found at {location}")

        return skill_dir

    path = Path(source).resolve()
    if not (path / "SKILL.md").is_file():
        raise FileNotFoundError(f"No SKILL.md found in {path}")
    return path


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


def _load_skill(skill_dir: Path) -> dict[str, str] | None:
    """Load a single skill from a directory.

    Reads the SKILL.md file, parses frontmatter, and returns a metadata
    dict. Returns None if the directory doesn't contain a valid skill
    (no SKILL.md, unreadable, or missing description).
    """
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.is_file():
        return None

    try:
        content = skill_file.read_text()
    except OSError:
        return None

    frontmatter = parse_frontmatter(content)
    name = frontmatter.get("name", skill_dir.name)
    description = frontmatter.get("description", "")

    if not description.strip():
        return None

    return {
        "name": name,
        "description": description,
        "file_path": str(skill_file),
        "base_dir": str(skill_dir),
    }


def discover(
    search_dirs: list[Path] | None = None,
    *,
    include_ephemeral: bool = True,
) -> list[dict[str, str]]:
    """Scan skill directories and return all discovered skills.

    Looks for subdirectories containing a SKILL.md file, parses the
    frontmatter for name and description, and returns a list of skill
    metadata dicts. Skills without a description are skipped (the
    description is what tells the model when to use the skill).

    First skill found with a given name wins — duplicates are silently
    skipped. Priority order:
      1. Ephemeral skills (--skill flag, if include_ephemeral=True)
      2. Project skills (.puget/skills/)
      3. Global skills ($PUGET_HOME/skills/)
      4. System skills (bundled)

    Args:
        search_dirs: Override the default skill directories. Useful for
                     testing without touching the real filesystem.
        include_ephemeral: Whether to include ephemeral skills loaded
                          via --skill. Default True.

    Returns:
        List of dicts, each with:
          - name: Skill identifier (from frontmatter or directory name).
          - description: What the skill does and when to use it.
          - file_path: Absolute path to the SKILL.md file.
          - base_dir: Absolute path to the skill directory.
    """
    skills: list[dict[str, str]] = []
    seen: set[str] = set()

    # Ephemeral skills first (highest priority).
    if include_ephemeral:
        for path in _ephemeral_skill_paths:
            skill = _load_skill(path)
            if skill and skill["name"] not in seen:
                seen.add(skill["name"])
                skills.append(skill)

    # Then scan search directories.
    for search_dir in (skill_dirs() if search_dirs is None else search_dirs):
        if not search_dir.is_dir():
            continue

        for entry in sorted(search_dir.iterdir()):
            if not entry.is_dir():
                continue

            skill = _load_skill(entry)
            if skill and skill["name"] not in seen:
                seen.add(skill["name"])
                skills.append(skill)

    return skills


# -- Prompt formatting -------------------------------------------------------

def format_for_prompt(skills: list[dict[str, str]]) -> str:
    """Format discovered skills as an XML block for the system prompt.

    Produces an <available_skills> block compatible with pi's Agent Skills
    convention. Each skill entry includes its name, description, and file
    path so the model can load the full content when needed.

    The preamble instructs the model to use the read tool to load skill
    files directly.

    Returns an empty string if no skills are available.
    """
    if not skills:
        return ""

    lines = [
        "",
        "The following skills provide specialized instructions for specific tasks.",
        "Use the `read` tool to load a skill's file when the task matches its description.",
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


# -- Skill management --------------------------------------------------------

def install_target(*, global_scope: bool = False) -> Path:
    """Return the target directory for skill installation.

    Args:
        global_scope: If True, return $PUGET_HOME/skills/.
                      If False, return .puget/skills/ (project-local).
    """
    if global_scope:
        from puget import puget_home
        return puget_home() / "skills"
    return Path.cwd() / ".puget" / "skills"


def install_skill(
    source: str,
    target_dir: Path,
    *,
    ref: str | None = None,
    subpath: str | None = None,
) -> str:
    """Install a skill from a local path or git URL.

    For local paths, copies the skill directory into the target.
    For git URLs, clones the repo, extracts the skill, and copies it.
    The .git directory is never copied.

    Args:
        source: Local directory path or git URL.
        target_dir: Directory to install into (e.g., .puget/skills/).
        ref: Git ref (branch/tag) to checkout. Only used for git sources.
        subpath: Path within repo to the skill directory. Only for git.

    Returns:
        The installed skill name.

    Raises:
        FileNotFoundError: If SKILL.md is not found in source.
        FileExistsError: If a skill with the same name already exists.
        RuntimeError: If git clone fails.
    """
    if source.startswith(("https://", "http://", "git@")):
        return _install_from_git(source, target_dir, ref=ref, subpath=subpath)

    source_path = Path(source).resolve()
    if not source_path.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source}")
    return _install_from_local(source_path, target_dir)


def remove_skill(name: str, search_dir: Path) -> None:
    """Remove an installed skill by name.

    Tries a direct directory match first, then falls back to scanning
    frontmatter names (in case the directory name differs from the
    frontmatter name).

    Args:
        name: The skill name.
        search_dir: The skills directory to remove from.

    Raises:
        FileNotFoundError: If the skill doesn't exist.
    """
    # Direct directory match.
    skill_path = search_dir / name
    if skill_path.is_dir():
        shutil.rmtree(skill_path)
        return

    # Fall back to scanning frontmatter names.
    if search_dir.is_dir():
        for entry in sorted(search_dir.iterdir()):
            if not entry.is_dir():
                continue
            skill = _load_skill(entry)
            if skill and skill["name"] == name:
                shutil.rmtree(entry)
                return

    raise FileNotFoundError(f"Skill '{name}' not found in {search_dir}")


def list_by_layer(
    layers: list[str] | None = None,
) -> dict[str, list[dict[str, str]]]:
    """Return skills grouped by trait layer.

    Scans each requested layer independently and returns the results
    keyed by layer name.

    Args:
        layers: Which layers to include ("project", "global", "system").
                None means all layers.

    Returns:
        Ordered dict mapping layer name to list of skill metadata dicts.
    """
    from puget import puget_home

    all_layers: dict[str, Path] = {
        "project": Path.cwd() / ".puget" / "skills",
        "global": puget_home() / "skills",
        "system": _system_skills_dir(),
    }

    if layers:
        selected = {k: v for k, v in all_layers.items() if k in layers}
    else:
        selected = all_layers

    result: dict[str, list[dict[str, str]]] = {}
    for layer_name, search_dir in selected.items():
        result[layer_name] = discover(
            search_dirs=[search_dir],
            include_ephemeral=False,
        )

    return result


# -- Internal helpers ---------------------------------------------------------

def _install_from_local(source_path: Path, target_dir: Path) -> str:
    """Install a skill from a local directory."""
    if not (source_path / "SKILL.md").is_file():
        raise FileNotFoundError(f"No SKILL.md found in {source_path}")

    content = (source_path / "SKILL.md").read_text()
    frontmatter = parse_frontmatter(content)
    name = frontmatter.get("name", source_path.name)

    dest = target_dir / name
    if dest.exists():
        raise FileExistsError(
            f"Skill '{name}' already exists at {dest}. "
            f"Remove it first with: puget skill remove {name}"
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_path, dest, ignore=shutil.ignore_patterns(".git"))
    return name


def _install_from_git(
    url: str,
    target_dir: Path,
    *,
    ref: str | None = None,
    subpath: str | None = None,
) -> str:
    """Install a skill from a git URL."""
    parsed_url, parsed_ref, parsed_subpath = _parse_git_source(url)

    # Explicit CLI args override values parsed from the URL.
    clone_url = parsed_url
    effective_ref = ref or parsed_ref
    effective_subpath = subpath or parsed_subpath

    tmp = Path(tempfile.mkdtemp(prefix="puget-skill-"))
    try:
        _git_clone(clone_url, tmp / "repo", ref=effective_ref)

        skill_dir = tmp / "repo"
        if effective_subpath:
            skill_dir = skill_dir / effective_subpath

        return _install_from_local(skill_dir, target_dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _parse_git_source(url: str) -> tuple[str, str | None, str | None]:
    """Parse a git URL into (clone_url, ref, subpath).

    Supports GitHub/GitLab tree URLs:
      https://github.com/user/repo/tree/branch/path/to/skill
      → (https://github.com/user/repo, branch, path/to/skill)

    And plain git URLs:
      https://github.com/user/repo
      git@github.com:user/repo.git
      → (url, None, None)
    """
    # GitHub/GitLab tree URL pattern.
    m = re.match(
        r"(https?://[^/]+/[^/]+/[^/]+)/tree/([^/]+)(?:/(.+))?$",
        url,
    )
    if m:
        return m.group(1), m.group(2), m.group(3)

    # Plain URL — pass through as-is.
    return url, None, None


def _git_clone(clone_url: str, dest: Path, *, ref: str | None = None) -> None:
    """Clone a git repository. Raises RuntimeError on failure."""
    cmd = ["git", "clone", "--depth", "1"]
    if ref:
        cmd += ["--branch", ref]
    cmd += [clone_url, str(dest)]

    result = _subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr.strip()}")
