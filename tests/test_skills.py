"""Tests for skill discovery and prompt injection."""


from puget.skills import discover, format_for_prompt, parse_frontmatter


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
