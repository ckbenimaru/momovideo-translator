from pathlib import Path

from modelscope.hub.snapshot_download import snapshot_download
from PyQt5.QtCore import QThread, pyqtSignal

try:
    from huggingface_hub import snapshot_download as hf_snapshot_download
except ImportError:  # Kept for compatibility with older portable installations.
    hf_snapshot_download = None


REQUIRED_MODEL_FILES = ("model.bin", "config.json", "tokenizer.json")


def validate_model_directory(model_path: str | Path) -> Path:
    """Validate that a downloaded CTranslate2 Whisper model is usable."""
    path = Path(model_path)
    missing = [name for name in REQUIRED_MODEL_FILES if not (path / name).is_file()]
    if missing:
        raise RuntimeError("模型下载不完整，缺少文件: " + ", ".join(missing))
    if (path / "model.bin").stat().st_size == 0:
        raise RuntimeError("模型下载不完整: model.bin 为空文件")
    return path


def download_model_snapshot(
    model_id: str,
    save_path: str | Path,
    huggingface_repo: str | None = None,
) -> Path:
    """Download a model snapshot and verify all runtime files exist.

    Hugging Face is preferred because it reliably streams the multi-gigabyte
    model file. ModelScope remains a fallback for networks where Hugging Face
    is unavailable. ModelScope 1.38 removed the old ``progress_callbacks``
    argument, so only version-stable parameters are used here.
    """
    target = Path(save_path)
    target.mkdir(parents=True, exist_ok=True)

    hf_error = None
    if huggingface_repo and hf_snapshot_download is not None:
        try:
            hf_snapshot_download(
                repo_id=huggingface_repo,
                local_dir=str(target),
                max_workers=4,
            )
        except Exception as error:
            hf_error = error
        else:
            return validate_model_directory(target)

    try:
        snapshot_download(model_id=model_id, local_dir=str(target), max_workers=4)
    except Exception as modelscope_error:
        if hf_error is not None:
            raise RuntimeError(
                f"Hugging Face 下载失败: {hf_error}; "
                f"ModelScope 下载失败: {modelscope_error}"
            ) from modelscope_error
        raise
    return validate_model_directory(target)


class ModelscopeDownloadThread(QThread):
    progress = pyqtSignal(int, str)
    succeeded = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(
        self,
        model_id: str,
        save_path: str,
        huggingface_repo: str | None = None,
    ):
        super().__init__()
        self.model_id = model_id
        self.save_path = save_path
        self.huggingface_repo = huggingface_repo

    def run(self):
        try:
            self.progress.emit(5, self.tr("正在连接模型仓库并下载模型..."))
            model_path = download_model_snapshot(
                self.model_id,
                self.save_path,
                self.huggingface_repo,
            )
            self.progress.emit(100, self.tr("下载完成"))
            self.succeeded.emit(str(model_path))

        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    import sys

    from PyQt5.QtCore import QCoreApplication

    app = QCoreApplication(sys.argv)
    model_id = "pengzhendong/faster-whisper-tiny"
    save_path = r"models/faster-whisper-tiny"
    downloader = ModelscopeDownloadThread(model_id, save_path)

    def on_progress(percentage, message):
        print(f"进度: {message}")

    def on_error(error_msg):
        print(f"错误: {error_msg}")
        app.quit()

    def on_finished():
        print("下载完成！")
        app.quit()

    downloader.progress.connect(on_progress)
    downloader.error.connect(on_error)
    downloader.finished.connect(on_finished)

    print(f"开始下载模型 {model_id}")
    downloader.start()

    sys.exit(app.exec_())
