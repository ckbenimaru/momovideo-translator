"""Helpers for the bundled/local Ollama translation runtime."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

import psutil

from videocaptioner.config import MODEL_PATH
from videocaptioner.core.utils.logger import setup_logger
from videocaptioner.core.utils.platform_utils import is_apple_silicon

DEFAULT_OLLAMA_DIR = Path(r"D:\AI\Ollama")
DEFAULT_OLLAMA_EXE = DEFAULT_OLLAMA_DIR / "ollama.exe"
DEFAULT_OLLAMA_MODELS = Path(r"D:\AI\OllamaModels")
DEFAULT_OLLAMA_MODEL = "qwen3:4b-instruct-2507-q4_K_M"
DEFAULT_OLLAMA_API = "http://127.0.0.1:11434"
logger = setup_logger("ollama_runtime")


def is_local_ollama_url(base_url: str | None) -> bool:
    try:
        url = urlsplit(base_url or "")
        return (
            url.scheme == "http"
            and url.hostname in {"127.0.0.1", "localhost"}
            and url.port == 11434
        )
    except ValueError:
        return False


def apple_silicon_ollama_parallelism() -> int:
    if is_apple_silicon() and psutil.virtual_memory().total >= 24 * 1024**3:
        return 4  # Verified with Qwen3 4B at a 4K context on a 24 GiB Mac.
    return 1


def _request(path: str, data: dict | None = None, timeout: float = 3.0):
    body = None if data is None else json.dumps(data).encode("utf-8")
    request = urllib.request.Request(
        f"{DEFAULT_OLLAMA_API}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ollama_is_ready() -> bool:
    try:
        _request("/api/version")
        return True
    except (OSError, ValueError, urllib.error.URLError):
        return False


def ensure_ollama_server(timeout: float = 20.0, *, parallelism: int = 1) -> None:
    """Start the local Ollama runtime when it is not running."""
    if ollama_is_ready():
        return

    env = os.environ.copy()
    cwd = None
    if os.name == "nt":
        if not DEFAULT_OLLAMA_EXE.is_file():
            raise RuntimeError(
                f"離線翻譯環境不存在：{DEFAULT_OLLAMA_EXE}。請先安裝 Qwen 翻譯環境。"
            )
        executable = str(DEFAULT_OLLAMA_EXE)
        cwd = str(DEFAULT_OLLAMA_DIR)
        DEFAULT_OLLAMA_MODELS.mkdir(parents=True, exist_ok=True)
        env["OLLAMA_MODELS"] = str(DEFAULT_OLLAMA_MODELS)
    else:
        executable = shutil.which("ollama")
        if executable is None:
            raise RuntimeError("離線翻譯環境不存在。請安裝 Ollama，並將 ollama 加入 PATH。")
        local_models = MODEL_PATH / "ollama"
        if sys.platform == "darwin" and local_models.is_dir():
            env.setdefault("OLLAMA_MODELS", str(local_models.resolve()))

    env["OLLAMA_HOST"] = "127.0.0.1:11434"
    if parallelism > 1:
        env.setdefault("OLLAMA_NUM_PARALLEL", str(parallelism))
        env.setdefault("OLLAMA_CONTEXT_LENGTH", "4096")
    command = [executable, "serve"]
    if is_apple_silicon():
        command = ["/usr/bin/arch", "-arm64", *command]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        command,
        cwd=cwd,
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
        if process.poll() is not None:
            raise RuntimeError(f"Ollama 離線翻譯服務無法啟動：{executable}。")
        time.sleep(0.25)
    raise RuntimeError(f"Ollama 離線翻譯服務啟動逾時，請檢查 {executable}。")


def ensure_ollama_model(model: str = DEFAULT_OLLAMA_MODEL) -> None:
    tags = _request("/api/tags", timeout=5.0).get("models", [])
    available = {item.get("name", "") for item in tags}
    aliases = {name.removesuffix(":latest") for name in available}
    if model not in available and model not in aliases:
        if os.name != "nt":
            raise RuntimeError(f"離線翻譯模型尚未下載：{model}。請執行 ollama pull {model}。")
        raise RuntimeError(
            f"離線翻譯模型尚未下載：{model}。模型目錄：{DEFAULT_OLLAMA_MODELS}"
        )


def prepare_local_ollama(
    model: str = DEFAULT_OLLAMA_MODEL, *, parallelism: int = 1
) -> None:
    ensure_ollama_server(parallelism=parallelism)
    ensure_ollama_model(model)


def log_ollama_model_placement(model: str) -> None:
    try:
        models = _request("/api/ps").get("models", [])
    except (OSError, ValueError):
        return
    name = model.removesuffix(":latest")
    for loaded in models:
        if (loaded.get("name") or "").removesuffix(":latest") != name:
            continue
        size, gpu_size = loaded.get("size"), loaded.get("size_vram")
        if size and gpu_size is not None:
            log = logger.info if gpu_size >= size else logger.warning
            log(
                "Ollama %s: GPU %.0f%%, context %s",
                name,
                100 * gpu_size / size,
                loaded.get("context_length", "unknown"),
            )
        break


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
