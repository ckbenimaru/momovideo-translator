import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal

from videocaptioner.core.asr.qwen3_asr import qwen_runtime_python
from videocaptioner.core.utils.platform_utils import get_subprocess_kwargs, is_apple_silicon

QWEN_RUNTIME_PACKAGES = [
    # The downloaded *-hf checkpoints use the native Qwen3-ASR classes.  They
    # are not available in the old 4.57.x API even though importing the base
    # transformers package succeeds.
    "transformers==5.14.1",
    "accelerate==1.12.0",
    "librosa>=0.11.0",
    "soundfile>=0.13.1",
    "huggingface-hub>=0.34.0",
    "safetensors>=0.6.0",
]
QWEN_MLX_RUNTIME_PACKAGES = [
    "mlx-audio==0.6.5",
    "librosa>=0.11.0",
    "soundfile>=0.13.1",
    "nagisa==0.3.1",
    "soynlp==0.0.493",
]


def qwen_runtime_commands(runtime_dir: str | Path) -> list[list[str]]:
    runtime_dir = str(runtime_dir)
    runtime_python = str(qwen_runtime_python(runtime_dir))
    commands = [
        [sys.executable, "-m", "venv", runtime_dir],
        [
            runtime_python,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--upgrade",
            "pip",
        ],
    ]
    if is_apple_silicon():
        commands.append(
            [runtime_python, "-m", "pip", "install", "--no-cache-dir", *QWEN_MLX_RUNTIME_PACKAGES]
        )
        return commands

    torch_command = [
        runtime_python,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "torch>=2.7.0",
    ]
    if os.name == "nt":
        torch_command.extend(
            ["--index-url", "https://download.pytorch.org/whl/cu128"]
        )
    commands.append(torch_command)
    commands.append(
        [
            runtime_python,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            *QWEN_RUNTIME_PACKAGES,
        ]
    )
    return commands


def validate_qwen_runtime(runtime_dir: str | Path) -> tuple[bool, str]:
    runtime_python = qwen_runtime_python(runtime_dir)
    if not runtime_python.is_file():
        return False, "Python 环境不存在"
    use_mlx = is_apple_silicon()
    command = [
        str(runtime_python),
        "-c",
        (
            "from importlib.metadata import version; import librosa, soundfile; "
            "from mlx_audio.stt import load; "
            "import nagisa; from soynlp.tokenizer import LTokenizer; "
            "print('MLX Audio ' + version('mlx-audio'))"
            if use_mlx else
            "import torch, transformers, librosa, soundfile; "
            "from transformers import AutoModelForMultimodalLM, "
            "AutoModelForTokenClassification, AutoProcessor; "
            "print(torch.__version__); print(torch.cuda.is_available()); "
            "print(transformers.__version__)"
        ),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        **get_subprocess_kwargs(),
    )
    if result.returncode != 0:
        return False, result.stderr.strip() or "运行依赖不完整"
    if use_mlx:
        return True, result.stdout.strip()
    values = result.stdout.strip().splitlines()
    torch_version = values[0] if values else ""
    cuda_state = values[1] if len(values) > 1 else "False"
    transformers_version = values[2] if len(values) > 2 else ""
    return True, (
        f"PyTorch {torch_version}, CUDA={cuda_state}, "
        f"Transformers {transformers_version}"
    )


class QwenRuntimeInstallThread(QThread):
    progress = pyqtSignal(int, str)
    succeeded = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, runtime_dir: str):
        super().__init__()
        self.runtime_dir = runtime_dir
        self.process = None

    def run(self):
        log_path = Path(self.runtime_dir).parent / "qwen-runtime-install.log"
        try:
            commands = qwen_runtime_commands(self.runtime_dir)
            messages = [
                "建立 Qwen 独立 Python 环境",
                "更新安装工具",
                "安装 CUDA PyTorch",
                "安装 Qwen 推理依赖",
            ]
            if is_apple_silicon():
                messages = [*messages[:2], "安装 Apple Silicon MLX 推理依赖"]
            for index, (command, message) in enumerate(zip(commands, messages)):
                self.progress.emit(index * 23, message)
                self.process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    **get_subprocess_kwargs(),
                )
                output = []
                assert self.process.stdout is not None
                for line in self.process.stdout:
                    if line.strip():
                        output.append(line.strip())
                return_code = self.process.wait()
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write(
                        f"\n[{datetime.now().isoformat(timespec='seconds')}] {message}\n"
                    )
                    log_file.write("\n".join(output[-200:]))
                    log_file.write("\n")
                if return_code != 0:
                    raise RuntimeError("\n".join(output[-20:]))
            valid, details = validate_qwen_runtime(self.runtime_dir)
            if not valid:
                raise RuntimeError(details)
            self.progress.emit(100, "Qwen3-ASR 运行环境安装完成")
            self.succeeded.emit(details)
        except Exception as error:
            self.error.emit(
                f"{type(error).__name__}: {error}\n安装日志: {log_path}"
            )
        finally:
            self.process = None

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()


class QwenRuntimeCheckThread(QThread):
    """Validate the isolated Qwen runtime without blocking the Qt UI thread."""

    completed = pyqtSignal(bool, str)

    def __init__(self, runtime_dir: str):
        super().__init__()
        self.runtime_dir = runtime_dir

    def run(self):
        try:
            valid, details = validate_qwen_runtime(self.runtime_dir)
            self.completed.emit(valid, details)
        except subprocess.TimeoutExpired:
            self.completed.emit(False, "運行環境檢查逾時")
        except Exception as error:
            self.completed.emit(False, f"{type(error).__name__}: {error}")
