"""Tests for skill discovery, prompt injection, and management."""

import pytest

from puget.skills import (
    _load_skill,
    _parse_git_source,
    add_ephemeral_skill,
    clear_ephemeral_skills,
    discover,
    format_for_prompt,
    install_skill,
    list_by_layer,
    parse_frontmatter,
    remove_skill,
)


# -- Frontmatter parsing ----------------------------------------------------

class TestFrontmatter:
    def test_basic_parsing(self):
        text = "---\nname: my-skill\ndescription: Does things.\n---\n# Body"
        result = parse_frontmatter(text)
        assert result == {"name": "my-skill", "description": "Does things."}

    def test_no_frontmatter(self):
        assert parse_frontmatter("# Just markdown") == {}

    def test_unclosed_frontmatter(self):
        assert parse_frontmatter("---\nname: broken\n# no closing") == {}

    def test_empty_frontmatter(self):
        assert parse_frontmatter("---\n---\n# Body") == {}

    def test_ignores_lines_without_colon(self):
        text = "---\nname: ok\nthis has no key value\n---"
        result = parse_frontmatter(text)
        assert result == {"name": "ok"}

    def test_colon_in_value(self):
        text = "---\nname: my-skill\ndescription: Does things: well.\n---"
        result = parse_frontmatter(text)
        assert result["description"] == "Does things: well."


# -- _load_skill -------------------------------------------------------------

class TestLoadSkill:
    def test_loads_valid_skill(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: Test skill.\n---\n# Hello"
        )

        result = _load_skill(skill_dir)
        assert result is not None
        assert result["name"] == "my-skill"
        assert result["description"] == "Test skill."
        assert result["file_path"] == str(skill_dir / "SKILL.md")
        assert result["base_dir"] == str(skill_dir)

    def test_returns_none_for_missing_skill_md(self, tmp_path):
        skill_dir = tmp_path / "no-file"
        skill_dir.mkdir()
        assert _load_skill(skill_dir) is None

    def test_returns_none_for_no_description(self, tmp_path):
        skill_dir = tmp_path / "no-desc"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: no-desc\n---\n# Hello")
        assert _load_skill(skill_dir) is None

    def test_falls_back_to_directory_name(self, tmp_path):
        skill_dir = tmp_path / "dir-name-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\ndescription: Uses dir name.\n---\n# Hello"
        )

        result = _load_skill(skill_dir)
        assert result is not None
        assert result["name"] == "dir-name-skill"

    def test_returns_none_for_nonexistent_dir(self, tmp_path):
        assert _load_skill(tmp_path / "nope") is None


# -- Discovery ---------------------------------------------------------------

class TestDiscovery:
    def test_discovers_skill_from_directory(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: Test skill.\n---\n# Hello"
        )

        skills = discover(search_dirs=[tmp_path])
        assert len(skills) == 1
        assert skills[0]["name"] == "my-skill"
        assert skills[0]["description"] == "Test skill."

    def test_skips_skill_without_description(self, tmp_path):
        skill_dir = tmp_path / "no-desc"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: no-desc\n---\n# Hello")

        skills = discover(search_dirs=[tmp_path])
        assert len(skills) == 0

    def test_falls_back_to_directory_name(self, tmp_path):
        skill_dir = tmp_path / "dir-name-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\ndescription: Uses dir name.\n---\n# Hello"
        )

        skills = discover(search_dirs=[tmp_path])
        assert skills[0]["name"] == "dir-name-skill"

    def test_deduplicates_by_name(self, tmp_path):
        dir1 = tmp_path / "search1"
        dir2 = tmp_path / "search2"
        for d in [dir1, dir2]:
            skill = d / "dupe"
            skill.mkdir(parents=True)
            (skill / "SKILL.md").write_text(
                f"---\nname: dupe\ndescription: From {d.name}.\n---"
            )

        skills = discover(search_dirs=[dir1, dir2])
        assert len(skills) == 1
        assert skills[0]["description"] == "From search1."

    def test_skips_nonexistent_directories(self, tmp_path):
        fake = tmp_path / "nonexistent"
        skills = discover(search_dirs=[fake])
        assert skills == []

    def test_skips_files_in_root(self, tmp_path):
        (tmp_path / "not-a-skill.md").write_text(
            "---\nname: nope\ndescription: I'm a file.\n---"
        )
        skills = discover(search_dirs=[tmp_path])
        assert len(skills) == 0


# -- Ephemeral skills --------------------------------------------------------

class TestEphemeralSkills:
    def teardown_method(self):
        clear_ephemeral_skills()

    def test_ephemeral_skills_discovered(self, tmp_path):
        skill_dir = tmp_path / "ephemeral-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: ephemeral\ndescription: Session only.\n---\n# Hi"
        )

        add_ephemeral_skill(skill_dir)
        # Use empty search_dirs so only ephemeral skills are found.
        skills = discover(search_dirs=[])
        assert len(skills) == 1
        assert skills[0]["name"] == "ephemeral"

    def test_ephemeral_skills_shadow_installed(self, tmp_path):
        # Installed version.
        installed_dir = tmp_path / "search" / "my-skill"
        installed_dir.mkdir(parents=True)
        (installed_dir / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: Installed version.\n---"
        )

        # Ephemeral version with same name.
        eph_dir = tmp_path / "ephemeral" / "my-skill"
        eph_dir.mkdir(parents=True)
        (eph_dir / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: Ephemeral version.\n---"
        )

        add_ephemeral_skill(eph_dir)
        skills = discover(search_dirs=[tmp_path / "search"])
        assert len(skills) == 1
        assert skills[0]["description"] == "Ephemeral version."

    def test_ephemeral_excluded_when_flag_false(self, tmp_path):
        skill_dir = tmp_path / "eph"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: eph\ndescription: Should not appear.\n---"
        )

        add_ephemeral_skill(skill_dir)
        skills = discover(search_dirs=[], include_ephemeral=False)
        assert skills == []


# -- Prompt formatting -------------------------------------------------------

class TestPromptFormatting:
    def test_empty_skills(self):
        assert format_for_prompt([]) == ""

    def test_formats_xml_block(self):
        skills = [
            {
                "name": "test-skill",
                "description": "Does testing.",
                "file_path": "/path/to/SKILL.md",
                "base_dir": "/path/to",
            }
        ]
        result = format_for_prompt(skills)
        assert "<available_skills>" in result
        assert "<name>test-skill</name>" in result
        assert "<description>Does testing.</description>" in result
        assert "<location>/path/to/SKILL.md</location>" in result
        assert "</available_skills>" in result

    def test_multiple_skills(self):
        skills = [
            {"name": "a", "description": "A.", "file_path": "/a", "base_dir": "/"},
            {"name": "b", "description": "B.", "file_path": "/b", "base_dir": "/"},
        ]
        result = format_for_prompt(skills)
        assert result.count("<skill>") == 2


# -- Git URL parsing ---------------------------------------------------------

class TestParseGitSource:
    def test_github_tree_url_with_path(self):
        url = "https://github.com/user/repo/tree/main/skills/my-skill"
        clone_url, ref, subpath = _parse_git_source(url)
        assert clone_url == "https://github.com/user/repo"
        assert ref == "main"
        assert subpath == "skills/my-skill"

    def test_github_tree_url_branch_only(self):
        url = "https://github.com/user/repo/tree/develop"
        clone_url, ref, subpath = _parse_git_source(url)
        assert clone_url == "https://github.com/user/repo"
        assert ref == "develop"
        assert subpath is None

    def test_plain_https_url(self):
        url = "https://github.com/user/repo"
        clone_url, ref, subpath = _parse_git_source(url)
        assert clone_url == "https://github.com/user/repo"
        assert ref is None
        assert subpath is None

    def test_plain_https_url_with_git_suffix(self):
        url = "https://github.com/user/repo.git"
        clone_url, ref, subpath = _parse_git_source(url)
        assert clone_url == "https://github.com/user/repo.git"
        assert ref is None
        assert subpath is None

    def test_git_ssh_url(self):
        url = "git@github.com:user/repo.git"
        clone_url, ref, subpath = _parse_git_source(url)
        assert clone_url == "git@github.com:user/repo.git"
        assert ref is None
        assert subpath is None

    def test_gitlab_tree_url(self):
        url = "https://gitlab.com/org/project/tree/v2.0/skills/lint"
        clone_url, ref, subpath = _parse_git_source(url)
        assert clone_url == "https://gitlab.com/org/project"
        assert ref == "v2.0"
        assert subpath == "skills/lint"


# -- Install from local path ------------------------------------------------

class TestInstallFromLocal:
    def _make_skill(self, path, name="test-skill", desc="A test skill."):
        """Create a minimal skill directory."""
        path.mkdir(parents=True, exist_ok=True)
        (path / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {desc}\n---\n# Hello"
        )
        (path / "helper.sh").write_text("echo hi")
        return path

    def test_installs_skill(self, tmp_path):
        source = self._make_skill(tmp_path / "source" / "my-skill")
        target = tmp_path / "target"

        name = install_skill(str(source), target)
        assert name == "test-skill"
        assert (target / "test-skill" / "SKILL.md").is_file()
        assert (target / "test-skill" / "helper.sh").is_file()

    def test_uses_frontmatter_name(self, tmp_path):
        source = self._make_skill(
            tmp_path / "source" / "dir-name",
            name="frontmatter-name",
        )
        target = tmp_path / "target"

        name = install_skill(str(source), target)
        assert name == "frontmatter-name"
        assert (target / "frontmatter-name" / "SKILL.md").is_file()

    def test_rejects_missing_skill_md(self, tmp_path):
        source = tmp_path / "no-skill"
        source.mkdir()

        with pytest.raises(FileNotFoundError, match="No SKILL.md"):
            install_skill(str(source), tmp_path / "target")

    def test_rejects_nonexistent_source(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            install_skill(str(tmp_path / "nope"), tmp_path / "target")

    def test_rejects_duplicate(self, tmp_path):
        source = self._make_skill(tmp_path / "source" / "s")
        target = tmp_path / "target"

        install_skill(str(source), target)
        with pytest.raises(FileExistsError, match="already exists"):
            install_skill(str(source), target)

    def test_creates_target_dir(self, tmp_path):
        source = self._make_skill(tmp_path / "source" / "s")
        target = tmp_path / "deep" / "nested" / "target"

        install_skill(str(source), target)
        assert (target / "test-skill" / "SKILL.md").is_file()

    def test_excludes_git_directory(self, tmp_path):
        source = self._make_skill(tmp_path / "source" / "s")
        (source / ".git").mkdir()
        (source / ".git" / "config").write_text("gitconfig")
        target = tmp_path / "target"

        install_skill(str(source), target)
        assert not (target / "test-skill" / ".git").exists()


# -- Remove skill ------------------------------------------------------------

class TestRemoveSkill:
    def test_removes_by_directory_name(self, tmp_path):
        skill_dir = tmp_path / "skills" / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: Test.\n---"
        )

        remove_skill("my-skill", tmp_path / "skills")
        assert not skill_dir.exists()

    def test_removes_by_frontmatter_name(self, tmp_path):
        # Directory name differs from frontmatter name.
        skill_dir = tmp_path / "skills" / "dir-name"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: fm-name\ndescription: Test.\n---"
        )

        remove_skill("fm-name", tmp_path / "skills")
        assert not skill_dir.exists()

    def test_raises_for_missing_skill(self, tmp_path):
        search_dir = tmp_path / "skills"
        search_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="not found"):
            remove_skill("nonexistent", search_dir)

    def test_raises_for_missing_search_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            remove_skill("anything", tmp_path / "nonexistent")


# -- List by layer -----------------------------------------------------------

class TestListByLayer:
    def test_lists_skills_from_specific_layer(self, tmp_path, monkeypatch):
        # Set up a project skills directory.
        project_skills = tmp_path / ".puget" / "skills" / "my-skill"
        project_skills.mkdir(parents=True)
        (project_skills / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: Project skill.\n---"
        )

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("PUGET_HOME", str(tmp_path / "home"))

        result = list_by_layer(layers=["project"])
        assert "project" in result
        assert len(result["project"]) == 1
        assert result["project"][0]["name"] == "my-skill"

    def test_all_layers_returned(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("PUGET_HOME", str(tmp_path / "home"))

        result = list_by_layer()
        assert "project" in result
        assert "global" in result
        assert "system" in result

    def test_empty_layers(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("PUGET_HOME", str(tmp_path / "home"))

        result = list_by_layer(layers=["global"])
        assert result["global"] == []
