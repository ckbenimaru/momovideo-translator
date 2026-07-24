"""Display geometry and orientation-aware subtitle sizing."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass

from videocaptioner.core.utils.platform_utils import get_subprocess_kwargs


@dataclass(frozen=True)
class SubtitleGeometry:
    width: int
    height: int
    is_portrait: bool
    typography_scale: float
    horizontal_margin: int
    bottom_margin: int
    max_text_width: int
    max_chars_cjk: int
    max_chars_english: int


def display_dimensions(width: int, height: int, rotation: float = 0) -> tuple[int, int]:
    """Return displayed dimensions after applying container rotation metadata."""
    normalized_rotation = int(round(rotation)) % 360
    if normalized_rotation in (90, 270):
        return height, width
    return width, height


def calculate_subtitle_geometry(
    width: int,
    height: int,
    reference_width: int = 1280,
    reference_height: int = 720,
) -> SubtitleGeometry:
    """Calculate typography and safe areas without changing landscape sizing."""
    width = max(int(width), 1)
    height = max(int(height), 1)
    is_portrait = height > width

    # Landscape styles historically use a 720p-height reference. Portrait
    # video is width-constrained, but a 1280px width reference makes captions
    # too small when watched on a phone. A 900px portrait reference at 720p
    # keeps 18 CJK characters inside the 90% safe width while producing a
    # readable visual size (34px at 720x1280, 50px at 1080x1920).
    portrait_reference_width = reference_height * 1.25
    typography_scale = (
        width / portrait_reference_width
        if is_portrait
        else height / reference_height
    )
    horizontal_margin = max(1, int(round(width * 0.05)))
    bottom_ratio = 0.10 if is_portrait else 0.08
    bottom_margin = max(1, int(round(height * bottom_ratio)))

    return SubtitleGeometry(
        width=width,
        height=height,
        is_portrait=is_portrait,
        typography_scale=typography_scale,
        horizontal_margin=horizontal_margin,
        bottom_margin=bottom_margin,
        max_text_width=max(1, width - horizontal_margin * 2),
        max_chars_cjk=18 if is_portrait else 20,
        max_chars_english=34 if is_portrait else 42,
    )


def _rotation_from_stream(stream: dict) -> float:
    for side_data in stream.get("side_data_list") or []:
        if side_data.get("rotation") is not None:
            try:
                return float(side_data["rotation"])
            except (TypeError, ValueError):
                pass
    try:
        return float((stream.get("tags") or {}).get("rotate", 0))
    except (TypeError, ValueError):
        return 0.0


def probe_display_geometry(video_path: str) -> tuple[int, int]:
    """Probe display dimensions, including phone rotation metadata."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:stream_tags=rotate:stream_side_data=rotation",
            "-of",
            "json",
            video_path,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        **get_subprocess_kwargs(),
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")
    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError("No video stream found")
    stream = streams[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError("Invalid video dimensions")
    return display_dimensions(width, height, _rotation_from_stream(stream))
