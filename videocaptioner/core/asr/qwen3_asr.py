import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional, Union

from videocaptioner.config import MODEL_PATH, ROOT_PATH
from .asr_data import ASRDataSeg
from .base import BaseASR


QWEN_ALIGNER_DIR_NAME = "Qwen3-ForcedAligner-0.6B-hf"


def default_qwen_runtime_dir() -> Path:
    return Path(ROOT_PATH) / "qwen3-asr-runtime"


def default_qwen_model_dir() -> Path:
    return Path(MODEL_PATH) / "Qwen3-ASR"


def resolve_qwen_paths(
    runtime_dir: str = "",
    model_dir: str = "",
    faster_whisper_model_dir: str = "",
) -> tuple[Path, Path]:
    """Resolve Qwen paths, reusing an existing external AI drive when set."""
    external_ai_root = None
    if faster_whisper_model_dir:
        external_ai_root = Path(faster_whisper_model_dir).expanduser().parent
    runtime = (
        Path(runtime_dir).expanduser()
        if runtime_dir
        else (
            external_ai_root / "Qwen3-ASR-Runtime"
            if external_ai_root
            else default_qwen_runtime_dir()
        )
    )
    models = (
        Path(model_dir).expanduser()
        if model_dir
        else (
            external_ai_root / "QwenModels"
            if external_ai_root
            else default_qwen_model_dir()
        )
    )
    return runtime, models


def qwen_runtime_python(runtime_dir: str | Path) -> Path:
    root = Path(runtime_dir)
    if os.name == "nt":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


class Qwen3ASR(BaseASR):
    """Run Qwen3-ASR in an isolated Python environment.

    Keeping PyTorch and Transformers outside momo's application environment
    avoids dependency conflicts. The worker loads ASR and forced aligner one
    after another so both do not occupy the 8 GB GPU at the same time.
    """

    def __init__(
        self,
        audio_input: Union[str, bytes],
        runtime_dir: str = "",
        model_dir: str = "",
        asr_model: str = "Qwen3-ASR-1.7B-hf",
        use_forced_aligner: bool = True,
        language: str = "",
        device: str = "cuda",
        chunk_seconds: int = 30,
        use_cache: bool = True,
    ):
        super().__init__(audio_input, use_cache)
        self.runtime_dir = Path(runtime_dir) if runtime_dir else default_qwen_runtime_dir()
        self.model_dir = Path(model_dir) if model_dir else default_qwen_model_dir()
        self.asr_model = asr_model
        self.use_forced_aligner = use_forced_aligner
        self.language = language
        self.device = device
        self.chunk_seconds = max(30, min(int(chunk_seconds), 240))
        self.process: Optional[subprocess.Popen] = None

    @property
    def asr_model_path(self) -> Path:
        return self.model_dir / self.asr_model

    @property
    def aligner_model_path(self) -> Path:
        return self.model_dir / QWEN_ALIGNER_DIR_NAME

    def _validate_installation(self) -> Path:
        runtime_python = qwen_runtime_python(self.runtime_dir)
        if not runtime_python.is_file():
            raise RuntimeError("Qwen3-ASR 运行环境未安装，请先打开模型管理完成安装。")
        if not (self.asr_model_path / "config.json").is_file() or not any(
            self.asr_model_path.glob("*.safetensors")
        ):
            raise RuntimeError(f"Qwen3-ASR 模型未下载: {self.asr_model_path}")
        if self.use_forced_aligner and (
            not (self.aligner_model_path / "config.json").is_file()
            or not any(self.aligner_model_path.glob("*.safetensors"))
        ):
            raise RuntimeError(f"Qwen3 Forced Aligner 未下载: {self.aligner_model_path}")
        return runtime_python

    def _run(
        self,
        callback: Optional[Callable[[int, str], None]] = None,
        **_kwargs,
    ) -> dict:
        callback = callback or (lambda _progress, _message: None)
        runtime_python = self._validate_installation()

        with tempfile.TemporaryDirectory(prefix="momo-qwen3-") as temp_dir:
            temp_root = Path(temp_dir)
            audio_path = temp_root / "input.wav"
            output_path = temp_root / "result.json"
            audio_path.write_bytes(self.file_binary or b"")

            worker = Path(__file__).with_name("qwen3_worker.py")
            cmd = [
                str(runtime_python),
                str(worker),
                "--audio",
                str(audio_path),
                "--asr-model",
                str(self.asr_model_path),
                "--output",
                str(output_path),
                "--device",
                self.device,
                "--chunk-seconds",
                str(self.chunk_seconds),
            ]
            if self.language:
                cmd.extend(["--language", self.language])
            if self.use_forced_aligner:
                cmd.extend(["--aligner-model", str(self.aligner_model_path)])

            creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=creationflags,
            )

            log_lines = []
            assert self.process.stdout is not None
            for line in self.process.stdout:
                line = line.rstrip()
                if line.startswith("PROGRESS\t"):
                    parts = line.split("\t", 2)
                    if len(parts) == 3:
                        try:
                            callback(int(parts[1]), parts[2])
                        except ValueError:
                            pass
                elif line:
                    log_lines.append(line)

            return_code = self.process.wait()
            self.process = None
            if return_code != 0:
                details = "\n".join(log_lines[-20:])
                raise RuntimeError(f"Qwen3-ASR 执行失败 (code={return_code})\n{details}")
            if not output_path.is_file():
                raise RuntimeError("Qwen3-ASR 未生成结果文件")
            return json.loads(output_path.read_text(encoding="utf-8"))

    def _make_segments(self, resp_data: dict) -> list[ASRDataSeg]:
        return [
            ASRDataSeg(
                text=str(item.get("text", "")).strip(),
                start_time=int(round(float(item.get("start_time", 0)) * 1000)),
                end_time=int(round(float(item.get("end_time", 0)) * 1000)),
                language=str(item.get("language", "")).strip(),
            )
            for item in resp_data.get("segments", [])
            if str(item.get("text", "")).strip()
        ]

    def _get_key(self) -> str:
        return ":".join(
            [
                self.crc32_hex,
                self.asr_model,
                str(self.use_forced_aligner),
                self.language,
                self.device,
                str(self.chunk_seconds),
                "mixed-language-v1",
            ]
        )

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
