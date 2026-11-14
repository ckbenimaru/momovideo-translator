import json
from concurrent.futures import CancelledError
from pathlib import Path
from threading import Event

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
        lambda model, **kwargs: calls.append(("model", model, kwargs)),
    )
    ollama_runtime.prepare_local_ollama("qwen-test")
    assert calls == [("server", 1), ("model", "qwen-test", {"on_download": None, "cancel_event": None})]


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



def test_installed_default_skips_download(monkeypatch):
    model = ollama_runtime.DEFAULT_OLLAMA_MODEL
    monkeypatch.setattr(ollama_runtime, "_request", lambda *args, **kwargs: {"models": [{"name": model}]})
    ollama_runtime.ensure_ollama_model(model, on_download=lambda: pytest.fail("already installed"))


@pytest.mark.parametrize(
    ("events", "message"),
    [
        ([b'{"status":"pulling manifest"}', b'{"status":"success"}'], None),
        ([b'{"status":"pulling manifest"}'], "未回報下載完成"),
        ([b'{"error":"not enough space"}'], "not enough space"),
    ],
)
def test_missing_default_downloads_via_ollama(monkeypatch, events, message):
    import io
    from types import SimpleNamespace

    calls = []
    monkeypatch.setattr(ollama_runtime, "_request", lambda *args, **kwargs: {"models": []})

    def open_pull(request, timeout):
        calls.append((request.full_url, json.loads(request.data), timeout))
        return io.BytesIO(b"\n".join(events))

    monkeypatch.setattr(
        ollama_runtime.urllib.request, "build_opener", lambda *args: SimpleNamespace(open=open_pull)
    )
    model = ollama_runtime.DEFAULT_OLLAMA_MODEL
    if message:
        with pytest.raises(RuntimeError, match=message):
            ollama_runtime.ensure_ollama_model(model, on_download=lambda: calls.append("download"))
    else:
        ollama_runtime.ensure_ollama_model(model, on_download=lambda: calls.append("download"))
    assert calls == ["download", ("http://127.0.0.1:11434/api/pull", {"model": model, "stream": True}, 30)]


def test_cancelled_task_does_not_start_download(monkeypatch):
    monkeypatch.setattr(ollama_runtime, "_request", lambda *args, **kwargs: {"models": []})
    event = Event()
    event.set()
    with pytest.raises(CancelledError):
        ollama_runtime.ensure_ollama_model(
            cancel_event=event, on_download=lambda: pytest.fail("cancelled task requested a download")
        )
