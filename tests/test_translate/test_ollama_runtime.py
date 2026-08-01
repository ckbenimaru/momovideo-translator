from pathlib import Path

import pytest

from videocaptioner.core.translate import ollama_runtime


def test_local_ollama_url_detection():
    assert ollama_runtime.is_local_ollama_url("http://127.0.0.1:11434/v1")
    assert ollama_runtime.is_local_ollama_url("http://localhost:11434/v1")
    assert not ollama_runtime.is_local_ollama_url("https://api.openai.com/v1")


def test_prepare_checks_server_then_model(monkeypatch):
    calls = []
    monkeypatch.setattr(
        ollama_runtime, "ensure_ollama_server", lambda: calls.append("server")
    )
    monkeypatch.setattr(
        ollama_runtime,
        "ensure_ollama_model",
        lambda model: calls.append(("model", model)),
    )
    ollama_runtime.prepare_local_ollama("qwen-test")
    assert calls == ["server", ("model", "qwen-test")]


def test_missing_runtime_has_actionable_error(monkeypatch):
    monkeypatch.setattr(ollama_runtime, "ollama_is_ready", lambda: False)
    monkeypatch.setattr(ollama_runtime, "DEFAULT_OLLAMA_EXE", Path("missing.exe"))
    with pytest.raises(RuntimeError, match="離線翻譯環境不存在"):
        ollama_runtime.ensure_ollama_server()


def test_missing_model_has_actionable_error(monkeypatch):
    monkeypatch.setattr(
        ollama_runtime,
        "_request",
        lambda *args, **kwargs: {"models": [{"name": "another:latest"}]},
    )
    with pytest.raises(RuntimeError, match="離線翻譯模型尚未下載"):
        ollama_runtime.ensure_ollama_model("qwen-test")
