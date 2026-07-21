import json
import re
import unittest

from videocaptioner.core.asr.asr_data import ASRData, ASRDataSeg
from videocaptioner.core.subtitle.resegment import (
    SubtitleSegmentationConfig,
    parse_faster_whisper_json,
    resegment_asr_data,
    resegment_translations,
    wrap_text,
)
from videocaptioner.core.translate.base import BaseTranslator
from videocaptioner.core.translate.types import TargetLanguage


def visible(text):
    return len(re.sub(r"\s+", "", text))


def word_data(tokens, duration_ms=400, gaps=None):
    gaps = gaps or {}
    cursor = 0
    segments = []
    for index, token in enumerate(tokens):
        cursor += gaps.get(index, 0)
        segments.append(ASRDataSeg(token, cursor, cursor + duration_ms))
        cursor += duration_ms
    return ASRData(segments)


class ContextTranslator(BaseTranslator):
    def __init__(self):
        super().__init__(1, 10, TargetLanguage.SIMPLIFIED_CHINESE, None)
        self.seen = []

    def _parallel_translate(self, chunks):
        result = []
        for chunk in chunks:
            for item in chunk:
                self.seen.append(
                    (item.previous_context, item.original_text, item.next_context)
                )
                item.translated_text = f"T:{item.original_text}"
                result.append(item)
        return result

    def _translate_chunk(self, subtitle_chunk):
        return subtitle_chunk


class SubtitleResegmentationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = SubtitleSegmentationConfig()

    def test_25_second_unpunctuated_sentence(self):
        data = word_data([f"word{i}" for i in range(50)], duration_ms=500)
        result = resegment_asr_data(data, self.cfg)
        self.assertGreaterEqual(len(result.segments), 5)
        self.assertTrue(
            all((seg.end_time - seg.start_time) <= 6000 for seg in result.segments)
        )
        joined = " ".join(seg.text.replace("\n", " ") for seg in result.segments)
        self.assertEqual(joined.split(), [f"word{i}" for i in range(50)])

    def test_multiple_pauses_are_preferred(self):
        data = word_data(
            [
                "one", "two", "three", "four", "five", "six",
                "seven", "eight", "nine", "ten", "eleven", "twelve",
            ],
            duration_ms=500,
            gaps={4: 700, 8: 800},
        )
        result = resegment_asr_data(data, self.cfg)
        self.assertGreaterEqual(len(result.segments), 2)
        self.assertGreaterEqual(
            result.segments[1].start_time - result.segments[0].end_time, 450
        )

    def test_fast_chinese_respects_character_and_line_limits(self):
        text = "這是一段用來測試快速中文語音字幕切分邏輯是否能夠正確運作的長句子而且內容還會繼續增加"
        data = word_data(list(text), duration_ms=120)
        result = resegment_asr_data(data, self.cfg)
        self.assertGreater(len(result.segments), 1)
        for seg in result.segments:
            self.assertLessEqual(visible(seg.text), 36)
            self.assertLessEqual(len(seg.text.splitlines()), 2)
            self.assertTrue(all(visible(line) <= 20 for line in seg.text.splitlines()))

    def test_english_to_chinese_translation_is_split_without_loss(self):
        chinese = "這是一段翻譯後顯著變長的中文字幕內容需要根據標點符號，閱讀速度和畫面寬度重新切分。"
        source = ASRData(
            [ASRDataSeg("This is an English source subtitle.", 0, 6000, chinese)]
        )
        result = resegment_translations(source, self.cfg)
        self.assertGreater(len(result.segments), 1)
        self.assertEqual(
            "".join(seg.translated_text.replace("\n", "") for seg in result.segments),
            chinese,
        )
        self.assertTrue(
            all(visible(seg.translated_text) <= 36 for seg in result.segments)
        )

    def test_chinese_to_english_translation_is_split_without_loss(self):
        english = (
            "This translated subtitle becomes substantially longer, and it should be "
            "split at punctuation whenever possible; no words may be removed from the result."
        )
        source = ASRData([ASRDataSeg("中文原文", 0, 6000, english)])
        result = resegment_translations(source, self.cfg)
        self.assertGreater(len(result.segments), 1)
        rebuilt = " ".join(
            seg.translated_text.replace("\n", " ") for seg in result.segments
        )
        self.assertEqual(rebuilt.split(), english.split())
        self.assertTrue(
            all(visible(seg.translated_text) <= 76 for seg in result.segments)
        )

    def test_sub_08_second_cue_merges_when_safe(self):
        data = word_data(["hello.", "short"], duration_ms=400)
        result = resegment_asr_data(data, self.cfg)
        self.assertEqual(len(result.segments), 1)
        self.assertGreaterEqual(
            result.segments[0].end_time - result.segments[0].start_time, 800
        )

    def test_long_translation_second_pass_prefers_punctuation(self):
        translated = (
            "First complete thought ends here. Second complete thought is also rather "
            "long, so the subtitle should split cleanly! Third thought remains intact."
        )
        data = ASRData([ASRDataSeg("原文字幕", 1000, 11000, translated)])
        result = resegment_translations(data, self.cfg)
        self.assertGreaterEqual(len(result.segments), 2)
        self.assertTrue(
            result.segments[0].translated_text.replace("\n", "").endswith(".")
        )

    def test_two_line_wrapping(self):
        english = "one two three four five six seven eight nine"
        wrapped = wrap_text(
            english,
            SubtitleSegmentationConfig(
                max_chars_per_line=24,
                max_lines=2,
                max_chars_per_subtitle=48,
            ),
        )
        self.assertEqual(len(wrapped.splitlines()), 2)
        self.assertTrue(all(len(line) <= 24 for line in wrapped.splitlines()))

    def test_empty_words_and_missing_word_timestamps_fallback(self):
        payload = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 8.0,
                    "text": "fallback text without word timestamps",
                    "words": [],
                }
            ]
        }
        words = parse_faster_whisper_json(json.dumps(payload))
        self.assertGreater(len(words), 1)
        result = resegment_asr_data(ASRData(words), self.cfg)
        self.assertTrue(result.segments)
        self.assertEqual(
            " ".join(seg.text.replace("\n", " ") for seg in result.segments).split(),
            payload["segments"][0]["text"].split(),
        )
        self.assertEqual(resegment_asr_data(ASRData([]), self.cfg).segments, [])

    def test_translation_receives_previous_and_next_context(self):
        source = ASRData(
            [
                ASRDataSeg("first", 0, 1000),
                ASRDataSeg("current", 1000, 2000),
                ASRDataSeg("last", 2000, 3000),
            ]
        )
        translator = ContextTranslator()
        result = translator.translate_subtitle(source)
        self.assertEqual(translator.seen[1], ("first", "current", "last"))
        self.assertEqual(result.segments[1].translated_text, "T:current")
        translator.stop()


if __name__ == "__main__":
    unittest.main()
