"""Predictable quality-tier, macOS-compatible FFmpeg encoding policy."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple


@dataclass(frozen=True)
class SourceVideoInfo:
    codec: str
    bit_rate: int
    width: int
    height: int
    fps: float


def _positive_int(value: object) -> int:
    try:
        parsed = int(str(value))
        return parsed if parsed > 0 else 0
    except (TypeError, ValueError):
        return 0


def _parse_fps(value: object) -> float:
    try:
        fps = float(Fraction(str(value)))
        return fps if fps > 0 else 30.0
    except (TypeError, ValueError, ZeroDivisionError):
        return 30.0


def probe_source_video(video_path: str) -> SourceVideoInfo:
    """Read the first video stream without relying on localized FFmpeg text."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,bit_rate,width,height,avg_frame_rate:format=bit_rate",
        "-of",
        "json",
        video_path,
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=(
            getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        ),
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")

    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError("No video stream found")

    stream = streams[0]
    width = _positive_int(stream.get("width")) or 1920
    height = _positive_int(stream.get("height")) or 1080
    fps = _parse_fps(stream.get("avg_frame_rate"))
    bit_rate = _positive_int(stream.get("bit_rate"))
    if not bit_rate:
        # Container bitrate includes audio. Reserve 256 kbps for a conservative
        # video estimate when the stream itself does not publish a bitrate.
        bit_rate = max(0, _positive_int((payload.get("format") or {}).get("bit_rate")) - 256_000)

    return SourceVideoInfo(
        codec=str(stream.get("codec_name") or "unknown").lower(),
        bit_rate=bit_rate,
        width=width,
        height=height,
        fps=fps,
    )


@lru_cache(maxsize=1)
def is_nvenc_available() -> bool:
    """Return whether FFmpeg can actually initialize the NVIDIA H.264 encoder."""
    null_output = "NUL" if os.name == "nt" else "/dev/null"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        # Recent NVENC drivers reject dimensions below their minimum surface
        # size, so use a small but universally accepted probe frame.
        "color=size=256x256:rate=1:duration=0.1",
        "-frames:v",
        "1",
        "-an",
        "-c:v",
        "h264_nvenc",
        "-f",
        "null",
        "-y",
        null_output,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            creationflags=(
                getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            ),
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


_TARGET_BITRATE_BY_CRF = {
    18: 8000,
    23: 5000,
    28: 3000,
    32: 1500,
}


def get_target_bitrate_kbps(crf: int) -> int:
    """Map the legacy quality/CRF value to a predictable VBR target."""
    if crf in _TARGET_BITRATE_BY_CRF:
        return _TARGET_BITRATE_BY_CRF[crf]
    nearest_crf = min(_TARGET_BITRATE_BY_CRF, key=lambda value: abs(value - crf))
    return _TARGET_BITRATE_BY_CRF[nearest_crf]


def _common_compatibility_args(max_rate_k: int, buffer_k: int) -> List[str]:
    return [
        "-maxrate",
        f"{max_rate_k}k",
        "-bufsize",
        f"{buffer_k}k",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-tag:v",
        "avc1",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
    ]


def build_mac_encoding_args(
    video_path: str,
    output_path: str,
    crf: int,
    preset: str,
    prefer_hardware: bool = True,
) -> Tuple[List[str], SourceVideoInfo, int]:
    """Return macOS-compatible H.264/AAC VBR args for the selected quality."""
    info = probe_source_video(video_path)

    # ``crf`` is retained in the public signature for compatibility with the
    # existing task/rendering interfaces, but now identifies one of four
    # explicit target-bitrate tiers.
    target_rate_k = get_target_bitrate_kbps(crf)
    max_rate_k = target_rate_k * 5 // 4
    buffer_k = target_rate_k * 2
    max_rate = max_rate_k * 1000

    suffix = Path(output_path).suffix.lower()
    if suffix not in {".mp4", ".mov", ".m4v"}:
        raise ValueError("Mac-compatible policy requires MP4, MOV, or M4V output")

    if prefer_hardware and is_nvenc_available():
        args = [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p5",
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-multipass",
            "fullres",
            "-rc-lookahead",
            "32",
            "-b:v",
            f"{target_rate_k}k",
            *_common_compatibility_args(max_rate_k, buffer_k),
        ]
    else:
        args = [
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-b:v",
            f"{target_rate_k}k",
            *_common_compatibility_args(max_rate_k, buffer_k),
        ]
    return args, info, max_rate
