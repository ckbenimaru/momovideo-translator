"""Helpers for the bundled/local Ollama translation runtime."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_OLLAMA_DIR = Path(r"D:\AI\Ollama")
DEFAULT_OLLAMA_EXE = DEFAULT_OLLAMA_DIR / "ollama.exe"
DEFAULT_OLLAMA_MODELS = Path(r"D:\AI\OllamaModels")
DEFAULT_OLLAMA_MODEL = "qwen3:4b-instruct-2507-q4_K_M"
DEFAULT_OLLAMA_API = "http://127.0.0.1:11434"


def is_local_ollama_url(base_url: str | None) -> bool:
    value = (base_url or "").lower()
    return "127.0.0.1:11434" in value or "localhost:11434" in value


def _request(path: str, data: dict | None = None, timeout: float = 3.0):
    body = None if data is None else json.dumps(data).encode("utf-8")
    request = urllib.request.Request(
        f"{DEFAULT_OLLAMA_API}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ollama_is_ready() -> bool:
    try:
        _request("/api/version")
        return True
    except (OSError, ValueError, urllib.error.URLError):
        return False


def ensure_ollama_server(timeout: float = 20.0) -> None:
    """Start the D-drive portable Ollama runtime when it is not running."""
    if ollama_is_ready():
        return
    if not DEFAULT_OLLAMA_EXE.is_file():
        raise RuntimeError(
            f"離線翻譯環境不存在：{DEFAULT_OLLAMA_EXE}。請先安裝 Qwen 翻譯環境。"
        )

    DEFAULT_OLLAMA_MODELS.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = str(DEFAULT_OLLAMA_MODELS)
    env["OLLAMA_HOST"] = "127.0.0.1:11434"
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen(
        [str(DEFAULT_OLLAMA_EXE), "serve"],
        cwd=str(DEFAULT_OLLAMA_DIR),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ollama_is_ready():
            return
        time.sleep(0.25)
    raise RuntimeError("Ollama 離線翻譯服務啟動逾時，請檢查 D:\\AI\\Ollama。")


def ensure_ollama_model(model: str = DEFAULT_OLLAMA_MODEL) -> None:
    tags = _request("/api/tags", timeout=5.0).get("models", [])
    available = {item.get("name", "") for item in tags}
    aliases = {name.removesuffix(":latest") for name in available}
    if model not in available and model not in aliases:
        raise RuntimeError(
            f"離線翻譯模型尚未下載：{model}。模型目錄：{DEFAULT_OLLAMA_MODELS}"
        )


def prepare_local_ollama(model: str = DEFAULT_OLLAMA_MODEL) -> None:
    ensure_ollama_server()
    ensure_ollama_model(model)


def release_ollama_model(model: str = DEFAULT_OLLAMA_MODEL) -> None:
    """Unload the translation model so ASR/video work can reclaim VRAM."""
    if not ollama_is_ready():
        return
    try:
        _request(
            "/api/generate",
            {"model": model, "prompt": "", "stream": False, "keep_alive": 0},
            timeout=15.0,
        )
    except Exception:
        # Releasing VRAM is best-effort and must not turn a completed task into
        # a failure.
        return
