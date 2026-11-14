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
        ollama_runtime, "ensure_ollama_server", lambda *, parallelism: calls.append(("server", parallelism))
    )
    monkeypatch.setattr(
        ollama_runtime,
        "ensure_ollama_model",
        lambda model: calls.append(("model", model)),
    )
    ollama_runtime.prepare_local_ollama("qwen-test")
    assert calls == [("server", 1), ("model", "qwen-test")]


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



@pytest.mark.parametrize("custom_models", [False, True])
def test_macos_uses_local_models_unless_overridden(monkeypatch, tmp_path, custom_models):
    model_root = tmp_path / "models"
    shared_models = tmp_path / "shared"
    (shared_models / "ollama").mkdir(parents=True)
    model_root.symlink_to(shared_models, target_is_directory=True)
    custom_path = tmp_path / "custom"
    monkeypatch.setattr(ollama_runtime.sys, "platform", "darwin")
    monkeypatch.setattr(ollama_runtime, "MODEL_PATH", model_root)
    monkeypatch.setattr(ollama_runtime, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(ollama_runtime.shutil, "which", lambda name: "/test/ollama")
    readiness = iter((False, True))
    monkeypatch.setattr(ollama_runtime, "ollama_is_ready", lambda: next(readiness))
    if custom_models:
        monkeypatch.setenv("OLLAMA_MODELS", str(custom_path))
    else:
        monkeypatch.delenv("OLLAMA_MODELS", raising=False)

    launches = []
    monkeypatch.setattr(
        ollama_runtime.subprocess, "Popen", lambda *args, **kwargs: launches.append(kwargs["env"])
    )
    ollama_runtime.ensure_ollama_server()
    expected = custom_path if custom_models else (shared_models / "ollama").resolve()
    assert launches[0]["OLLAMA_MODELS"] == str(expected)
