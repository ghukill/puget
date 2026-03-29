"""Tests for model selection and runtime override."""

from puget.model import DEFAULT_MODEL, _strip_chat_template_tokens, get_model, set_model


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


class TestStripChatTemplateTokens:
    def test_strips_qwen_tokens(self):
        assert _strip_chat_template_tokens("hello <|im_start|>user") == "hello user"
        assert _strip_chat_template_tokens("<|im_end|>") == ""

    def test_strips_llama_tokens(self):
        assert _strip_chat_template_tokens("<|start_header_id|>assistant<|end_header_id|>") == "assistant"
        assert _strip_chat_template_tokens("done<|eot_id|>") == "done"

    def test_leaves_clean_text_alone(self):
        assert _strip_chat_template_tokens("just normal text") == "just normal text"
        assert _strip_chat_template_tokens("") == ""
