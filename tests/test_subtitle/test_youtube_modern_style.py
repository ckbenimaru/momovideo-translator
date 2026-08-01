import unittest

from videocaptioner.core.subtitle.rounded_renderer import (
    _fit_text_lines,
    _font_name_for_text,
    render_subtitle_image,
)
from videocaptioner.core.subtitle.style_manager import SubtitleStyle
from videocaptioner.core.subtitle.styles import RoundedBgStyle
from videocaptioner.core.subtitle.text_utils import hex_to_rgba
from videocaptioner.core.translate.types import TargetLanguage
from videocaptioner.ui.task_factory import TaskFactory


class YouTubeModernStyleTests(unittest.TestCase):
    def setUp(self):
        self.style = RoundedBgStyle(**TaskFactory.get_rounded_style())

    def test_default_style_values(self):
        self.assertEqual(self.style.font_weight, 500)
        self.assertEqual(self.style.bg_color, "#00000059")
        self.assertEqual(hex_to_rgba(self.style.bg_color), (0, 0, 0, 89))
        self.assertEqual(self.style.outline_width, 1)
        self.assertEqual(self.style.line_height_ratio, 1.3)
        self.assertEqual(self.style.max_lines, 2)
        self.assertEqual(self.style.max_chars_cjk, 20)
        self.assertEqual(self.style.max_chars_english, 42)
        self.assertEqual(self.style.margin_bottom, 58)

    def test_language_specific_noto_fonts(self):
        self.assertEqual(_font_name_for_text("這是繁體中文", self.style), "Noto Sans TC")
        self.assertEqual(_font_name_for_text("这是简体中文", self.style), "Noto Sans SC")
        self.assertEqual(
            _font_name_for_text("This is an English subtitle", self.style),
            "Noto Sans",
        )

    def test_ass_style_defaults_to_noto_sans_tc(self):
        ass_style = SubtitleStyle(name="youtube-modern").to_ass_string()
        self.assertIn("Style: Default,Noto Sans TC,42,", ass_style)
        self.assertIn("Style: Secondary,Noto Sans TC,29,", ass_style)
        style_lines = [
            line for line in ass_style.splitlines() if line.startswith("Style:")
        ]
        self.assertTrue(all(line.split(",")[7] == "0" for line in style_lines))

    def test_task_factory_ass_style_selects_font_for_target_language(self):
        cases = {
            TargetLanguage.TRADITIONAL_CHINESE: "Noto Sans TC",
            TargetLanguage.CANTONESE: "Noto Sans TC",
            TargetLanguage.SIMPLIFIED_CHINESE: "Noto Sans SC",
            TargetLanguage.ENGLISH: "Noto Sans",
            TargetLanguage.ENGLISH_US: "Noto Sans",
            TargetLanguage.ENGLISH_UK: "Noto Sans",
        }
        for language, font_name in cases.items():
            with self.subTest(language=language):
                ass_style = TaskFactory.get_ass_style("youtube-modern", language)
                self.assertIn("; MomoAutoFontByText: yes", ass_style)
                self.assertIn(f"Style: Default,{font_name},42,", ass_style)
                self.assertIn(f"Style: Secondary,{font_name},29,", ass_style)
                style_lines = [
                    line
                    for line in ass_style.splitlines()
                    if line.startswith("Style:")
                ]
                self.assertTrue(
                    all(line.split(",")[7] == "0" for line in style_lines)
                )

    def test_custom_boolean_bold_remains_ass_compatible(self):
        regular = SubtitleStyle(font_name="Arial", bold=False).to_ass_string()
        bold = SubtitleStyle(font_name="Arial", bold=True).to_ass_string()
        self.assertIn("Style: Default,Arial,42,", regular)
        self.assertTrue(
            all(
                line.split(",")[7] == "0"
                for line in regular.splitlines()
                if line.startswith("Style:")
            )
        )
        self.assertTrue(
            all(
                line.split(",")[7] == "-1"
                for line in bold.splitlines()
                if line.startswith("Style:")
            )
        )

    def test_cjk_wraps_to_two_lines_at_twenty_characters(self):
        text = "這是一段用來測試現代字幕自動換行效果的繁體中文內容一共需要顯示兩行"
        _, lines = _fit_text_lines(text, 1920, self.style, 2)
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(len(line) <= 20 for line in lines))
        self.assertEqual("".join(lines), text)

    def test_english_wraps_to_two_lines_at_forty_two_characters(self):
        text = "Modern subtitles remain clear and readable across bright video scenes."
        _, lines = _fit_text_lines(text, 1920, self.style, 2)
        self.assertLessEqual(len(lines), 2)
        self.assertTrue(all(len(line) <= 42 for line in lines))
        self.assertEqual(" ".join(lines), text)

    def test_render_uses_eight_percent_bottom_safe_margin(self):
        image = render_subtitle_image("這是繁體中文字幕", "", 1280, 720, self.style)
        alpha_bbox = image.getchannel("A").getbbox()
        self.assertIsNotNone(alpha_bbox)
        visible_margin_ratio = (720 - alpha_bbox[3]) / 720
        self.assertGreaterEqual(visible_margin_ratio, 0.075)
        self.assertLessEqual(visible_margin_ratio, 0.09)


if __name__ == "__main__":
    unittest.main()
