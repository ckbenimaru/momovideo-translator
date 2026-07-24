import os
from pathlib import Path

from huggingface_hub import snapshot_download
from PyQt5.QtCore import QThread, pyqtSignal


def validate_qwen_model(model_dir: str | Path) -> Path:
    path = Path(model_dir)
    if not (path / "config.json").is_file():
        raise RuntimeError("模型下载不完整，缺少 config.json")
    if not any(path.glob("*.safetensors")):
        raise RuntimeError("模型下载不完整，缺少 safetensors 权重")
    return path


def download_qwen_model(repo_id: str, model_dir: str | Path) -> Path:
    target = Path(model_dir)
    target.mkdir(parents=True, exist_ok=True)
    cache_dir = target.parent / ".huggingface-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Keep all temporary weight data beside the destination (normally D:),
    # rather than silently filling the user's C: profile cache.
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_XET_CACHE"] = str(cache_dir / "xet")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        cache_dir=str(cache_dir / "hub"),
        max_workers=4,
    )
    return validate_qwen_model(target)


class QwenModelDownloadThread(QThread):
    progress = pyqtSignal(int, str)
    succeeded = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, repo_id: str, model_dir: str):
        super().__init__()
        self.repo_id = repo_id
        self.model_dir = model_dir

    def run(self):
        try:
            self.progress.emit(5, f"正在下载 {self.repo_id}")
            path = download_qwen_model(self.repo_id, self.model_dir)
            self.progress.emit(100, "模型下载完成")
            self.succeeded.emit(str(path))
        except Exception as error:
            self.error.emit(f"{type(error).__name__}: {error}")
