"""Tests for model selection, capability detection, and thinking policy."""

from types import SimpleNamespace

import puget.model as model_module
from puget.model import (
    DEFAULT_MODEL,
    _strip_chat_template_tokens,
    chat,
    complete,
    get_context_window,
    get_model,
    get_model_info,
    get_thinking_mode,
    list_available_models,
    set_model,
    set_thinking_mode,
)


class _FakeHTTPResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


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


class TestThinkingMode:
    def teardown_method(self):
        set_thinking_mode(None)

    def test_default_thinking_mode(self, monkeypatch):
        monkeypatch.delenv("PUGET_OLLAMA_THINK", raising=False)
        assert get_thinking_mode() == "auto"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("PUGET_OLLAMA_THINK", "low")
        assert get_thinking_mode() == "low"

    def test_runtime_override(self, monkeypatch):
        monkeypatch.setenv("PUGET_OLLAMA_THINK", "off")
        set_thinking_mode("on")
        assert get_thinking_mode() == "on"

    def test_invalid_runtime_override(self):
        try:
            set_thinking_mode("weird")
        except ValueError as exc:
            assert "invalid thinking mode" in str(exc)
        else:
            raise AssertionError("expected ValueError")


class TestListAvailableModels:
    def test_parses_dict_response(self, monkeypatch):
        class FakeClient:
            def __init__(self, host):
                self.host = host

            def list(self):
                return {
                    "models": [
                        {"model": "llama3:8b"},
                        {"name": "qwen3:14b"},
                        {"model": "llama3:8b"},
                    ]
                }

        fake_ollama = SimpleNamespace(Client=FakeClient)
        monkeypatch.setitem(__import__("sys").modules, "ollama", fake_ollama)

        assert list_available_models() == ["llama3:8b", "qwen3:14b"]

    def test_parses_object_response(self, monkeypatch):
        class FakeModel:
            def __init__(self, name):
                self.name = name

        class FakeResponse:
            def __init__(self):
                self.models = [FakeModel("gemma3:27b")]

        class FakeClient:
            def __init__(self, host):
                self.host = host

            def list(self):
                return FakeResponse()

        fake_ollama = SimpleNamespace(Client=FakeClient)
        monkeypatch.setitem(__import__("sys").modules, "ollama", fake_ollama)

        assert list_available_models() == ["gemma3:27b"]

    def test_returns_empty_list_on_error(self, monkeypatch):
        class FakeClient:
            def __init__(self, host):
                self.host = host

            def list(self):
                raise RuntimeError("boom")

        fake_ollama = SimpleNamespace(Client=FakeClient)
        monkeypatch.setitem(__import__("sys").modules, "ollama", fake_ollama)

        assert list_available_models() == []


class TestOllamaMetadata:
    def setup_method(self):
        model_module._model_show_cache.clear()
        model_module._runtime_context_cache.clear()
        set_model(None)
        set_thinking_mode(None)

    def teardown_method(self):
        model_module._model_show_cache.clear()
        model_module._runtime_context_cache.clear()
        set_model(None)
        set_thinking_mode(None)

    def test_get_model_info_uses_show_and_ps(self, monkeypatch):
        set_model("qwen3:14b")

        def fake_post(url, json, timeout):
            assert url.endswith("/api/show")
            assert json == {"model": "qwen3:14b"}
            return _FakeHTTPResponse(
                {
                    "capabilities": ["completion", "tools", "thinking"],
                    "model_info": {"qwen.context_length": 262144},
                }
            )

        def fake_get(url, timeout):
            assert url.endswith("/api/ps")
            return _FakeHTTPResponse(
                {"models": [{"model": "qwen3:14b", "context_length": 131072}]}
            )

        monkeypatch.setattr(model_module.httpx, "post", fake_post)
        monkeypatch.setattr(model_module.httpx, "get", fake_get)

        info = get_model_info()
        assert info["capabilities"] == ["completion", "thinking", "tools"]
        assert info["capabilities_known"] is True
        assert info["supports_tools"] is True
        assert info["supports_thinking"] is True
        assert info["context_window"] == 131072
        assert get_context_window() == 131072

    def test_chat_omits_tools_and_think_when_unsupported(self, monkeypatch):
        set_model("dolphin3:latest")
        set_thinking_mode("on")
        payloads: list[dict] = []

        def fake_post(url, json, timeout):
            if url.endswith("/api/show"):
                return _FakeHTTPResponse({"capabilities": ["completion"]})
            if url.endswith("/api/chat"):
                payloads.append(json)
                return _FakeHTTPResponse({"message": {"content": "hello"}})
            raise AssertionError(f"unexpected URL: {url}")

        monkeypatch.setattr(model_module.httpx, "post", fake_post)

        response = chat(
            [{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "noop", "parameters": {}}}],
        )

        assert response["content"] == "hello"
        assert payloads[0]["model"] == "dolphin3:latest"
        assert "tools" not in payloads[0]
        assert "think" not in payloads[0]

    def test_auto_thinking_is_off_for_chat_and_low_for_summary(self, monkeypatch):
        set_model("qwen3:14b")
        set_thinking_mode("auto")
        payloads: list[dict] = []

        def fake_post(url, json, timeout):
            if url.endswith("/api/show"):
                return _FakeHTTPResponse({
                    "capabilities": ["completion", "thinking", "tools"],
                    "model_info": {"qwen.context_length": 262144},
                })
            if url.endswith("/api/chat"):
                payloads.append(json)
                return _FakeHTTPResponse({"message": {"content": "OK"}})
            raise AssertionError(f"unexpected URL: {url}")

        monkeypatch.setattr(model_module.httpx, "post", fake_post)

        chat([{"role": "user", "content": "hi"}], tools=[])
        complete([{"role": "user", "content": "summarize"}])

        assert payloads[0]["think"] is False
        assert payloads[1]["think"] == "low"


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
