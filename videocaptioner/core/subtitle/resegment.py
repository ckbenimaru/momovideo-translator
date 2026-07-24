"""Deterministic subtitle re-segmentation based on word timestamps.

The ASR engine is responsible only for recognising words.  This module owns
the presentation-oriented decisions: cue boundaries, reading speed and line
wrapping.  It deliberately has no dependency on the UI or translation
providers so it can be used by both the desktop and CLI pipelines.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from videocaptioner.core.asr.asr_data import ASRData, ASRDataSeg
from videocaptioner.core.utils.text_utils import is_mainly_cjk


SENTENCE_END = frozenset(".!?。！？…")
WEAK_END = frozenset(",;:，；：、")
NO_SPACE_BEFORE = SENTENCE_END | WEAK_END | frozenset(")]}>〉》」』】〕〗〙〛")
NO_SPACE_AFTER = frozenset("([{<〈《「『【〔〖〘〚")
TOKEN_PATTERN = re.compile(
    r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*|"
    r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]|"
    r"[^\s]",
    re.UNICODE,
)


@dataclass(frozen=True)
class SubtitleSegmentationConfig:
    ideal_duration: float = 3.5
    max_duration: float = 6.0
    min_duration: float = 0.8
    pause_threshold: float = 0.45
    # Zero means use the language-aware defaults documented below.
    max_chars_per_line: int = 0
    max_lines: int = 2
    max_chars_per_subtitle: int = 0
    max_chars_per_second: float = 0.0

    def limits_for(self, text: str) -> tuple[int, int, float]:
        cjk = is_mainly_cjk(text)
        per_line = self.max_chars_per_line or (20 if cjk else 42)
        per_subtitle = self.max_chars_per_subtitle or (36 if cjk else 76)
        chars_per_second = self.max_chars_per_second or (12.0 if cjk else 20.0)
        per_subtitle = min(per_subtitle, per_line * max(self.max_lines, 1))
        return per_line, per_subtitle, chars_per_second


@dataclass(frozen=True)
class WordTimestamp:
    text: str
    start_ms: int
    end_ms: int
    language: str = ""

    def __post_init__(self):
        if self.end_ms < self.start_ms:
            object.__setattr__(self, "end_ms", self.start_ms)


def config_from_object(value) -> SubtitleSegmentationConfig:
    """Build a segmentation config from TranscribeConfig/SubtitleConfig."""

    return SubtitleSegmentationConfig(
        ideal_duration=float(getattr(value, "ideal_duration", 3.5)),
        max_duration=float(getattr(value, "max_duration", 6.0)),
        min_duration=float(getattr(value, "min_duration", 0.8)),
        pause_threshold=float(getattr(value, "pause_threshold", 0.45)),
        max_chars_per_line=int(getattr(value, "max_chars_per_line", 0)),
        max_lines=int(getattr(value, "max_lines", 2)),
        max_chars_per_subtitle=int(getattr(value, "max_chars_per_subtitle", 0)),
        max_chars_per_second=float(getattr(value, "max_chars_per_second", 0.0)),
    )


def parse_faster_whisper_json(payload: str | dict) -> List[ASRDataSeg]:
    """Return word-level segments from Faster-Whisper JSON.

    If ``words`` is absent, segment text is converted to approximate word
    timings.  This makes malformed/old cached responses safe and deterministic.
    """

    data = json.loads(payload) if isinstance(payload, str) else payload
    words: List[WordTimestamp] = []
    for segment in (data or {}).get("segments", []):
        segment_words = []
        for word in segment.get("words") or []:
            text = str(word.get("word") or "")
            start = word.get("start")
            end = word.get("end")
            if not text.strip() or start is None or end is None:
                continue
            segment_words.append(
                WordTimestamp(text, int(round(float(start) * 1000)), int(round(float(end) * 1000)))
            )
        if segment_words:
            words.extend(segment_words)
        else:
            text = str(segment.get("text") or "")
            start_ms = int(round(float(segment.get("start") or 0) * 1000))
            end_ms = int(round(float(segment.get("end") or segment.get("start") or 0) * 1000))
            words.extend(_approximate_word_timings(text, start_ms, end_ms))
    words.sort(key=lambda word: (word.start_ms, word.end_ms))
    return [ASRDataSeg(word.text, word.start_ms, word.end_ms) for word in words]


def resegment_asr_data(
    data: ASRData,
    config: SubtitleSegmentationConfig | None = None,
) -> ASRData:
    """Flatten word timestamps and produce display-ready subtitle cues."""

    cfg = config or SubtitleSegmentationConfig()
    words = _flatten_asr_words(data)
    if not words:
        return data
    groups = _segment_words(words, cfg)
    segments = [
        ASRDataSeg(
            wrap_text(_join_words(group), cfg),
            group[0].start_ms,
            max(group[-1].end_ms, group[0].start_ms + 1),
            language=_language_for_words(group),
        )
        for group in groups
        if group
    ]
    return ASRData(segments)


def resegment_translations(
    data: ASRData,
    config: SubtitleSegmentationConfig | None = None,
) -> ASRData:
    """Validate translated cues and split overlong translations without truncation."""

    cfg = config or SubtitleSegmentationConfig()
    result: List[ASRDataSeg] = []
    for segment in data.segments:
        original = segment.text.replace("\n", " ").strip()
        translated = segment.translated_text.replace("\n", " ").strip()
        if not translated:
            result.append(
                ASRDataSeg(
                    wrap_text(original, cfg),
                    segment.start_time,
                    segment.end_time,
                    language=segment.language,
                )
            )
            continue

        duration = max((segment.end_time - segment.start_time) / 1000.0, 0.001)
        if not _text_exceeds_limits(translated, duration, cfg):
            result.append(
                ASRDataSeg(
                    wrap_text(original, cfg),
                    segment.start_time,
                    segment.end_time,
                    wrap_text(translated, cfg),
                    language=segment.language,
                )
            )
            continue

        translated_words = _approximate_word_timings(
            translated, segment.start_time, segment.end_time
        )
        translated_groups = _segment_words(translated_words, cfg)
        if len(translated_groups) <= 1:
            translated_groups = _split_words_by_capacity(translated_words, cfg)
        original_parts = _split_text_into_parts(original, len(translated_groups))
        for index, group in enumerate(translated_groups):
            if not group:
                continue
            result.append(
                ASRDataSeg(
                    wrap_text(original_parts[index], cfg),
                    group[0].start_ms,
                    max(group[-1].end_ms, group[0].start_ms + 1),
                    wrap_text(_join_words(group), cfg),
                    language=segment.language,
                )
            )
    return ASRData(result)


def wrap_text(text: str, config: SubtitleSegmentationConfig | None = None) -> str:
    """Wrap a cue into at most ``max_lines`` without discarding characters."""

    cfg = config or SubtitleSegmentationConfig()
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return ""
    per_line, _, _ = cfg.limits_for(clean)
    max_lines = max(cfg.max_lines, 1)
    if _visible_length(clean) <= per_line:
        return clean

    if is_mainly_cjk(clean):
        lines = _wrap_cjk(clean, per_line)
    else:
        lines = _wrap_latin(clean, per_line)

    if len(lines) <= max_lines:
        return "\n".join(lines)

    # Segmentation normally prevents this.  Preserve text if a single token or
    # caller-provided override still produces extra lines; never truncate.
    head = lines[: max_lines - 1]
    tail = " ".join(lines[max_lines - 1 :]) if not is_mainly_cjk(clean) else "".join(lines[max_lines - 1 :])
    return "\n".join(head + [tail])


def _flatten_asr_words(data: ASRData) -> List[WordTimestamp]:
    words: List[WordTimestamp] = []
    for segment in data.segments:
        if data._is_word_level_segment(segment):
            words.append(
                WordTimestamp(
                    segment.text,
                    segment.start_time,
                    segment.end_time,
                    segment.language,
                )
            )
        else:
            words.extend(
                _approximate_word_timings(
                    segment.text,
                    segment.start_time,
                    segment.end_time,
                    segment.language,
                )
            )
    return sorted(words, key=lambda word: (word.start_ms, word.end_ms))


def _approximate_word_timings(
    text: str,
    start_ms: int,
    end_ms: int,
    language: str = "",
) -> List[WordTimestamp]:
    tokens = TOKEN_PATTERN.findall(text or "")
    if not tokens:
        return []
    duration = max(end_ms - start_ms, len(tokens))
    weights = [max(_visible_length(token), 1) for token in tokens]
    total = sum(weights)
    result: List[WordTimestamp] = []
    cursor = start_ms
    for index, (token, weight) in enumerate(zip(tokens, weights)):
        token_end = end_ms if index == len(tokens) - 1 else cursor + max(1, round(duration * weight / total))
        token_end = min(max(token_end, cursor + 1), end_ms)
        result.append(WordTimestamp(token, cursor, token_end, language))
        cursor = token_end
    return result


def _language_for_words(words: Sequence[WordTimestamp]) -> str:
    languages = list(dict.fromkeys(word.language for word in words if word.language))
    if len(languages) == 1:
        return languages[0]
    return "Multilingual" if languages else ""


def _segment_words(
    words: Sequence[WordTimestamp], config: SubtitleSegmentationConfig
) -> List[List[WordTimestamp]]:
    groups: List[List[WordTimestamp]] = []
    start = 0
    while start < len(words):
        feasible: List[int] = []
        boundaries = {"sentence": [], "pause": [], "weak": []}
        for end in range(start, len(words)):
            group = words[start : end + 1]
            text = _join_words(group)
            duration = _duration_seconds(group)
            _, max_chars, _ = config.limits_for(text)
            exceeds_chars = _visible_length(text) > max_chars
            exceeds_duration = duration > config.max_duration
            if (exceeds_chars or exceeds_duration) and end > start:
                break
            feasible.append(end)
            if duration >= config.min_duration:
                boundary = _boundary_kind(words, end, config)
                if boundary:
                    boundaries[boundary].append(end)
            if duration >= config.max_duration:
                break

        if not feasible:
            feasible = [start]

        cut = None
        for kind in ("sentence", "pause", "weak"):
            if boundaries[kind]:
                cut = _closest_to_ideal(words, start, boundaries[kind], config)
                break
        if cut is None:
            valid = [
                end
                for end in feasible
                if _duration_seconds(words[start : end + 1]) >= config.min_duration
            ]
            cut = _closest_to_ideal(words, start, valid or feasible, config)

        groups.append(list(words[start : cut + 1]))
        start = cut + 1

    return _merge_short_groups(groups, config)


def _boundary_kind(
    words: Sequence[WordTimestamp], end: int, config: SubtitleSegmentationConfig
) -> str | None:
    token = words[end].text.rstrip()
    if token and token[-1] in SENTENCE_END:
        return "sentence"
    if end + 1 < len(words):
        pause = (words[end + 1].start_ms - words[end].end_ms) / 1000.0
        if pause > config.pause_threshold:
            return "pause"
    if token and token[-1] in WEAK_END:
        return "weak"
    return None


def _closest_to_ideal(
    words: Sequence[WordTimestamp], start: int, candidates: Sequence[int], config: SubtitleSegmentationConfig
) -> int:
    return min(
        candidates,
        key=lambda end: (
            abs(_duration_seconds(words[start : end + 1]) - config.ideal_duration),
            -end,
        ),
    )


def _merge_short_groups(
    groups: List[List[WordTimestamp]], config: SubtitleSegmentationConfig
) -> List[List[WordTimestamp]]:
    index = 0
    while index < len(groups):
        group = groups[index]
        if _duration_seconds(group) >= config.min_duration or len(groups) == 1:
            index += 1
            continue
        choices = []
        if index > 0 and _groups_can_merge(groups[index - 1], group, config):
            choices.append((index - 1, groups[index - 1] + group))
        if index + 1 < len(groups) and _groups_can_merge(group, groups[index + 1], config):
            choices.append((index + 1, group + groups[index + 1]))
        if not choices:
            index += 1
            continue
        neighbor, merged = min(choices, key=lambda item: _duration_seconds(item[1]))
        if neighbor < index:
            groups[neighbor] = merged
            groups.pop(index)
            index = max(neighbor, 0)
        else:
            groups[index] = merged
            groups.pop(neighbor)
    return groups


def _groups_can_merge(
    left: Sequence[WordTimestamp], right: Sequence[WordTimestamp], config: SubtitleSegmentationConfig
) -> bool:
    merged = list(left) + list(right)
    text = _join_words(merged)
    duration = _duration_seconds(merged)
    _, max_chars, max_cps = config.limits_for(text)
    return (
        duration <= config.max_duration
        and _visible_length(text) <= max_chars
        and _visible_length(text) / max(duration, 0.001) <= max_cps
    )


def _split_words_by_capacity(
    words: Sequence[WordTimestamp], config: SubtitleSegmentationConfig
) -> List[List[WordTimestamp]]:
    groups: List[List[WordTimestamp]] = []
    current: List[WordTimestamp] = []
    for word in words:
        candidate = current + [word]
        text = _join_words(candidate)
        _, max_chars, _ = config.limits_for(text)
        if current and _visible_length(text) > max_chars:
            groups.append(current)
            current = [word]
        else:
            current = candidate
    if current:
        groups.append(current)
    return groups


def _text_exceeds_limits(
    text: str, duration: float, config: SubtitleSegmentationConfig
) -> bool:
    per_line, per_subtitle, cps = config.limits_for(text)
    visible = _visible_length(text)
    wrapped = wrap_text(text, config).splitlines()
    return (
        visible > per_subtitle
        or len(wrapped) > config.max_lines
        or any(_visible_length(line) > per_line for line in wrapped)
        or visible / max(duration, 0.001) > cps
    )


def _split_text_into_parts(text: str, count: int) -> List[str]:
    if count <= 1:
        return [text]
    tokens = TOKEN_PATTERN.findall(text or "")
    if not tokens:
        return [""] * count
    parts: List[str] = []
    for index in range(count):
        start = round(index * len(tokens) / count)
        end = round((index + 1) * len(tokens) / count)
        parts.append(_join_token_strings(tokens[start:end]))
    return parts


def _join_words(words: Iterable[WordTimestamp]) -> str:
    return _join_token_strings([word.text for word in words])


def _join_token_strings(tokens: Sequence[str]) -> str:
    result = ""
    for raw in tokens:
        token = raw.strip()
        if not token:
            continue
        if not result:
            result = token
        elif token[0] in NO_SPACE_BEFORE or result[-1] in NO_SPACE_AFTER:
            result += token
        elif is_mainly_cjk(token) or is_mainly_cjk(result[-1]):
            result += token
        else:
            result += " " + token
    return re.sub(r"\s+", " ", result).strip()


def _duration_seconds(words: Sequence[WordTimestamp]) -> float:
    if not words:
        return 0.0
    return max((words[-1].end_ms - words[0].start_ms) / 1000.0, 0.001)


def _visible_length(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


def _wrap_cjk(text: str, width: int) -> List[str]:
    lines = []
    remaining = text
    while _visible_length(remaining) > width:
        cut = min(width, len(remaining))
        lower = max(1, int(width * 0.6))
        punctuation = [i + 1 for i, char in enumerate(remaining[:cut]) if char in SENTENCE_END | WEAK_END]
        valid = [value for value in punctuation if value >= lower]
        if valid:
            cut = valid[-1]
        lines.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        lines.append(remaining)
    return lines


def _wrap_latin(text: str, width: int) -> List[str]:
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        if len(word) > width:
            if current:
                lines.append(current)
                current = ""
            while len(word) > width:
                lines.append(word[:width])
                word = word[width:]
        candidate = word if not current else f"{current} {word}"
        if current and len(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines
