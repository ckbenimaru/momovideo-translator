import unittest
from unittest.mock import patch

from videocaptioner.core.asr.asr_data import ASRData, ASRDataSeg
from videocaptioner.core.entities import SubtitleLayoutEnum
from videocaptioner.core.utils.chinese_script import (
    font_name_for_subtitle_text,
    normalize_translated_script,
)
from videocaptioner.core.translate.types import TargetLanguage
from videocaptioner.core.translate.base import BaseTranslator
from videocaptioner.ui.task_factory import TaskFactory


class _EmptyTranslator(BaseTranslator):
    def _translate_chunk(self, subtitle_chunk):
        return subtitle_chunk


class ChineseScriptTests(unittest.TestCase):
    def test_missing_translation_stops_output_instead_of_using_original(self):
        data = ASRData(
            [ASRDataSeg(text="这个男人", start_time=0, end_time=1000)]
        )
        translator = _EmptyTranslator(
            thread_num=1,
            batch_num=1,
            target_language=TargetLanguage.TRADITIONAL_CHINESE,
            update_callback=None,
        )
        try:
            with patch("videocaptioner.core.translate.base.time.sleep"):
                with self.assertRaisesRegex(RuntimeError, "翻译不完整"):
                    translator.translate_subtitle(data)
        finally:
            translator.stop()

    def test_traditional_target_normalizes_simplified_translation(self):
        data = ASRData(
            [
                ASRDataSeg(
                    text="这个男人一定不正常",
                    translated_text="这个男人一定不正常",
                    start_time=0,
                    end_time=1000,
                )
            ]
        )
        normalize_translated_script(data, TargetLanguage.TRADITIONAL_CHINESE)
        self.assertEqual(data.segments[0].translated_text, "這個男人一定不正常")

    def test_simplified_target_normalizes_traditional_translation(self):
        data = ASRData(
            [
                ASRDataSeg(
                    text="這個男人一定不正常",
                    translated_text="這個男人一定不正常",
                    start_time=0,
                    end_time=1000,
                )
            ]
        )
        normalize_translated_script(data, TargetLanguage.SIMPLIFIED_CHINESE)
        self.assertEqual(data.segments[0].translated_text, "这个男人一定不正常")

    def test_font_detection_handles_simplified_traditional_and_english(self):
        self.assertEqual(font_name_for_subtitle_text("这个男人"), "Noto Sans SC")
        self.assertEqual(font_name_for_subtitle_text("這個男人"), "Noto Sans TC")
        self.assertEqual(font_name_for_subtitle_text("This is English"), "Noto Sans")

    def test_builtin_ass_adds_per_dialogue_font_override(self):
        data = ASRData(
            [
                ASRDataSeg(
                    text="這是原文",
                    translated_text="这个男人一定不正常",
                    start_time=0,
                    end_time=1000,
                )
            ]
        )
        style = TaskFactory.get_ass_style(
            "youtube-modern", TargetLanguage.SIMPLIFIED_CHINESE
        )
        ass = data.to_ass(style_str=style, layout=SubtitleLayoutEnum.ONLY_TRANSLATE)
        self.assertIn(r"{\fnNoto Sans SC}这个男人一定不正常", ass)


if __name__ == "__main__":
    unittest.main()
