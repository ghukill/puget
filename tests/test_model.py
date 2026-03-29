"""Tests for model selection and runtime override."""

from puget.model import DEFAULT_MODEL, get_model, set_model


class TestGetModel:
    def teardown_method(self):
        set_model(None)

    def test_default_model(self, monkeypatch):
        monkeypatch.delenv("PUGET_OLLAMA_MODEL", raising=False)
        assert get_model() == DEFAULT_MODEL

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("PUGET_OLLAMA_MODEL", "llama3:8b")
        assert get_model() == "llama3:8b"

    def test_set_model_overrides_env(self, monkeypatch):
        monkeypatch.setenv("PUGET_OLLAMA_MODEL", "llama3:8b")
        set_model("gemma2:27b")
        assert get_model() == "gemma2:27b"

    def test_set_model_overrides_default(self, monkeypatch):
        monkeypatch.delenv("PUGET_OLLAMA_MODEL", raising=False)
        set_model("qwen2:7b")
        assert get_model() == "qwen2:7b"

    def test_clear_override(self, monkeypatch):
        monkeypatch.setenv("PUGET_OLLAMA_MODEL", "llama3:8b")
        set_model("gemma2:27b")
        assert get_model() == "gemma2:27b"

        set_model(None)
        assert get_model() == "llama3:8b"
