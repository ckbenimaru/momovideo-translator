import json
import threading
import time
from datetime import datetime
from typing import Any, Dict

import httpx

from videocaptioner.config import LOG_PATH
from videocaptioner.core.llm.context import get_task_context

LLM_LOG_FILE = LOG_PATH / "llm_requests.jsonl"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB


_log_lock = threading.Lock()


# ==================== 日志写入 ====================


def _rotate_if_needed() -> None:
    """日志文件过大时轮转"""
    if not LLM_LOG_FILE.exists():
        return
    if LLM_LOG_FILE.stat().st_size < MAX_LOG_SIZE:
        return

    backup = LLM_LOG_FILE.with_suffix(".jsonl.old")
    if backup.exists():
        backup.unlink()
    LLM_LOG_FILE.rename(backup)


def _write_log(entry: Dict[str, Any]) -> None:
    """写入日志"""
    try:
        LOG_PATH.mkdir(parents=True, exist_ok=True)
        with _log_lock:
            _rotate_if_needed()
            with open(LLM_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ==================== HTTPX Hooks ====================


def _on_request(request: httpx.Request) -> None:
    if request.url.path.endswith("/chat/completions"):
        request.extensions["momo_started_at"] = time.monotonic()
        request.extensions["momo_task_context"] = get_task_context()


def _on_response(response: httpx.Response) -> None:
    request = response.request
    started = request.extensions.get("momo_started_at")
    if started is None or not response.is_success:
        return
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        return
    response.read()
    duration_ms = int((time.monotonic() - started) * 1000)
    try:
        request_data = json.loads(request.content)
        response_data = response.json()
    except ValueError:
        return
    ctx = request.extensions.get("momo_task_context")
    _write_log(
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_id": ctx.task_id if ctx else "",
            "file_name": ctx.file_name if ctx else "",
            "stage": ctx.stage if ctx else "",
            "url": str(request.url),
            "status": response.status_code,
            "duration_ms": duration_ms,
            "request": request_data,
            "response": response_data,
        }
    )


# ==================== 公开 API ====================


def create_logging_http_client() -> httpx.Client:
    """创建带日志记录的 HTTPX 客户端"""
    return httpx.Client(
        event_hooks={
            "request": [_on_request],
            "response": [_on_response],
        }
    )
