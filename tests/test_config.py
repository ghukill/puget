"""Tests for the centralized config module."""

import json
import os

import pytest

from puget import config, db


@pytest.fixture()
def conn(tmp_path, monkeypatch):
    """Create a test database."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("PUGET_DB", str(db_path))
    monkeypatch.setenv("PUGET_HOME", str(tmp_path / "home"))
    return db.connect()


class TestResolvers:
    def test_puget_home_default(self, monkeypatch):
        monkeypatch.delenv("PUGET_HOME", raising=False)
        assert config.puget_home() == config.Path.home() / ".puget"

    def test_puget_home_from_env(self, monkeypatch):
        monkeypatch.setenv("PUGET_HOME", "/custom/home")
        assert config.puget_home() == config.Path("/custom/home")

    def test_db_path_default(self, monkeypatch):
        monkeypatch.delenv("PUGET_DB", raising=False)
        monkeypatch.delenv("PUGET_HOME", raising=False)
        assert config.db_path() == config.Path.home() / ".puget" / "puget.db"

    def test_db_path_from_env(self, monkeypatch):
        monkeypatch.setenv("PUGET_DB", "/custom/db.sqlite")
        assert config.db_path() == config.Path("/custom/db.sqlite")

    def test_db_path_from_puget_home(self, monkeypatch):
        monkeypatch.delenv("PUGET_DB", raising=False)
        monkeypatch.setenv("PUGET_HOME", "/custom/home")
        assert config.db_path() == config.Path("/custom/home/puget.db")

    def test_ollama_host_default(self, monkeypatch):
        monkeypatch.delenv("PUGET_OLLAMA_HOST", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        assert config.ollama_host() == "http://localhost:11434"

    def test_ollama_host_from_env(self, monkeypatch):
        monkeypatch.setenv("PUGET_OLLAMA_HOST", "http://myhost:1234")
        assert config.ollama_host() == "http://myhost:1234"

    def test_ollama_host_adds_scheme(self, monkeypatch):
        monkeypatch.setenv("PUGET_OLLAMA_HOST", "myhost:1234")
        assert config.ollama_host() == "http://myhost:1234"


class TestSnapshot:
    def test_snapshot_has_required_keys(self, conn, tmp_path, monkeypatch):
        snap = config.snapshot()
        assert "puget_home" in snap
        assert "db_path" in snap
        assert "db_exists" in snap
        assert "model" in snap
        assert "model_capabilities" in snap
        assert "model_capabilities_known" in snap
        assert "context_window" in snap
        assert "ollama_host" in snap
        assert "show_thinking" in snap
        assert "thinking_mode" in snap
        assert "cwd" in snap
        assert "skill_dirs" in snap
        assert "current_wave_id" in snap

    def test_snapshot_db_exists_true(self, conn, tmp_path, monkeypatch):
        # conn fixture already created the db file.
        snap = config.snapshot()
        assert snap["db_exists"] is True

    def test_snapshot_db_exists_false(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PUGET_DB", str(tmp_path / "nonexistent.db"))
        monkeypatch.setenv("PUGET_HOME", str(tmp_path / "home"))
        snap = config.snapshot()
        assert snap["db_exists"] is False

    def test_snapshot_includes_wave_id(self, conn, tmp_path, monkeypatch):
        wid = db.new_wave(conn)
        snap = config.snapshot()
        assert snap["current_wave_id"] == wid

    def test_snapshot_includes_turn_count(self, conn, tmp_path, monkeypatch):
        wid = db.new_wave(conn)
        db.add_turn(conn, wid, "user", "hello")
        db.add_turn(conn, wid, "assistant", "hi")
        snap = config.snapshot()
        assert snap["current_wave_turn_count"] == 2

    def test_snapshot_no_turn_count_without_wave(self, conn, tmp_path, monkeypatch):
        snap = config.snapshot()
        assert snap["current_wave_id"] is None
        assert "current_wave_turn_count" not in snap

    def test_snapshot_json_is_valid(self, conn, tmp_path, monkeypatch):
        result = config.snapshot_json()
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "db_path" in parsed

    def test_snapshot_cwd(self, conn, tmp_path, monkeypatch):
        snap = config.snapshot()
        assert snap["cwd"] == os.getcwd()

    def test_skill_dirs_have_exists_flag(self, conn, tmp_path, monkeypatch):
        snap = config.snapshot()
        for sd in snap["skill_dirs"]:
            assert "path" in sd
            assert "exists" in sd
            assert isinstance(sd["exists"], bool)
