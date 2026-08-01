"""Chinese script normalization and ASS font selection helpers."""

import re
from functools import lru_cache
from typing import Optional

from opencc import OpenCC

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_ASS_TAG_RE = re.compile(r"\{[^}]*\}")


@lru_cache(maxsize=2)
def _converter(config: str) -> OpenCC:
    return OpenCC(config)


def normalize_translated_script(
    asr_data, target_language
):
    """Normalize translated Chinese text to the explicitly selected script."""
    language_name = getattr(target_language, "name", "")
    if language_name in {"TRADITIONAL_CHINESE", "CANTONESE"}:
        converter = _converter("s2t")
    elif language_name == "SIMPLIFIED_CHINESE":
        converter = _converter("t2s")
    else:
        return asr_data

    for seg in asr_data.segments:
        if seg.translated_text:
            seg.translated_text = converter.convert(seg.translated_text)
    return asr_data


def font_name_for_subtitle_text(text: str) -> Optional[str]:
    """Choose a Noto font only when the text gives a reliable script signal."""
    plain = _ASS_TAG_RE.sub("", text).replace("\\N", "")
    if not _CJK_RE.search(plain):
        return "Noto Sans" if _LATIN_RE.search(plain) else None

    has_simplified = _converter("s2t").convert(plain) != plain
    has_traditional = _converter("t2s").convert(plain) != plain

    # Noto Sans SC has the broadest coverage for mixed Chinese lines. This
    # prevents a single simplified-only glyph from falling back to YaHei Bold.
    if has_simplified:
        return "Noto Sans SC"
    if has_traditional:
        return "Noto Sans TC"
    return None


def apply_ass_font_override(text: str) -> str:
    """Apply a per-dialogue font override while preserving ASS line breaks."""
    font_name = font_name_for_subtitle_text(text)
    return f"{{\\fn{font_name}}}{text}" if font_name else text
