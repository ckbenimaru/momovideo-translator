import json
import subprocess
import unittest
from unittest.mock import patch

from videocaptioner.core.subtitle.ass_renderer import _scale_ass_style
from videocaptioner.core.subtitle.geometry import (
    calculate_subtitle_geometry,
    display_dimensions,
    probe_display_geometry,
)
from videocaptioner.core.subtitle.rounded_renderer import (
    _get_video_info,
    _scale_style_for_geometry,
    render_subtitle_image,
)
from videocaptioner.core.subtitle.styles import RoundedBgStyle


ASS_STYLE = (
    "[V4+ Styles]\n"
    "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,"
    "OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,"
    "ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,"
    "MarginR,MarginV,Encoding\n"
    "Style: Default,Noto Sans TC,42,&H00FFFFFF,&H000000FF,&H00202020,"
    "&HC8000000,0,0,0,0,100,100,0,0,3,1,1,2,10,10,58,1"
)


def style_fields(style_text: str) -> list[str]:
    line = next(line for line in style_text.splitlines() if line.startswith("Style:"))
    return line.split(",")


class PortraitGeometryTests(unittest.TestCase):
    @patch("videocaptioner.core.subtitle.rounded_renderer.probe_display_geometry")
    @patch("videocaptioner.core.subtitle.rounded_renderer.subprocess.run")
    def test_rounded_video_info_prefers_probe_and_checks_fallback(self, run, probe):
        run.return_value = subprocess.CompletedProcess(
            ["ffmpeg"], 1, "", "Duration: 00:00:12.50\nStream #0:0: Video: h264, 1920x1080"
        )
        probe.return_value = (1080, 1920)
        self.assertEqual(_get_video_info("phone.mov"), (1080, 1920, 12.5))
        probe.side_effect = RuntimeError("ffprobe failed")
        self.assertEqual(_get_video_info("phone.mov"), (1920, 1080, 12.5))
        run.return_value.stderr = "Invalid data found when processing input"
        with self.assertRaisesRegex(ValueError, "Cannot get video resolution"):
            _get_video_info("phone.mov")

    def test_landscape_keeps_existing_height_scale(self):
        geometry = calculate_subtitle_geometry(1920, 1080)
        self.assertFalse(geometry.is_portrait)
        self.assertEqual(geometry.typography_scale, 1.5)
        self.assertEqual(round(42 * geometry.typography_scale), 63)
        self.assertEqual(geometry.bottom_margin, 86)

    def test_portrait_uses_width_scale_and_safe_areas(self):
        geometry = calculate_subtitle_geometry(1080, 1920)
        self.assertTrue(geometry.is_portrait)
        self.assertAlmostEqual(geometry.typography_scale, 1080 / 900)
        self.assertEqual(round(42 * geometry.typography_scale), 50)
        self.assertEqual(geometry.horizontal_margin, 54)
        self.assertEqual(geometry.bottom_margin, 192)
        self.assertEqual(geometry.max_text_width, 972)
        self.assertEqual(geometry.max_chars_cjk, 18)
        self.assertEqual(geometry.max_chars_english, 34)

    def test_three_by_four_is_treated_as_portrait(self):
        geometry = calculate_subtitle_geometry(1080, 1440)
        self.assertTrue(geometry.is_portrait)
        self.assertEqual(round(42 * geometry.typography_scale), 50)

    def test_rotation_metadata_swaps_display_dimensions(self):
        self.assertEqual(display_dimensions(1920, 1080, 90), (1080, 1920))
        self.assertEqual(display_dimensions(1920, 1080, -90), (1080, 1920))
        self.assertEqual(display_dimensions(1920, 1080, 0), (1920, 1080))

    @patch("videocaptioner.core.subtitle.geometry.subprocess.run")
    def test_probe_reads_side_data_rotation(self, run):
        payload = {
            "streams": [
                {
                    "width": 1920,
                    "height": 1080,
                    "side_data_list": [{"rotation": -90}],
                }
            ]
        }
        run.return_value = subprocess.CompletedProcess(
            ["ffprobe"], 0, json.dumps(payload), ""
        )
        self.assertEqual(probe_display_geometry("phone.mp4"), (1080, 1920))

    def test_ass_portrait_font_and_margins_stay_inside_frame(self):
        fields = style_fields(
            _scale_ass_style(
                ASS_STYLE,
                1920 / 720,
                video_width=1080,
                video_height=1920,
            )
        )
        self.assertEqual(fields[2], "50")
        self.assertEqual(fields[19:22], ["54", "54", "192"])

    def test_ass_landscape_font_remains_legacy_size(self):
        fields = style_fields(
            _scale_ass_style(
                ASS_STYLE,
                1080 / 720,
                video_width=1920,
                video_height=1080,
            )
        )
        self.assertEqual(fields[2], "63")

    def test_rounded_portrait_render_obeys_safe_width_and_bottom_margin(self):
        style = _scale_style_for_geometry(RoundedBgStyle(), 1080, 1920)
        self.assertEqual(style.font_size, 50)
        self.assertEqual(style.margin_bottom, 192)
        self.assertEqual(style.max_chars_cjk, 18)

        image = render_subtitle_image("這是直屏影片字幕邊界測試內容", "", 1080, 1920, style)
        bbox = image.getchannel("A").getbbox()
        self.assertIsNotNone(bbox)
        left, _top, right, bottom = bbox
        self.assertGreaterEqual(left, 54)
        self.assertLessEqual(right, 1080 - 54)
        self.assertGreaterEqual(1920 - bottom, 192)

    def test_low_resolution_portrait_has_valid_font_search_range(self):
        style = _scale_style_for_geometry(RoundedBgStyle(), 480, 854)
        self.assertEqual(style.font_size, 22)
        image = render_subtitle_image("直屏字幕測試", "", 480, 854, style)
        self.assertIsNotNone(image.getchannel("A").getbbox())

    def test_720_portrait_uses_phone_readable_size_without_overflow(self):
        style = _scale_style_for_geometry(RoundedBgStyle(), 720, 1280)
        self.assertEqual(style.font_size, 34)
        image = render_subtitle_image("這是十八個中文字的直屏字幕邊界測試", "", 720, 1280, style)
        bbox = image.getchannel("A").getbbox()
        self.assertIsNotNone(bbox)
        self.assertGreaterEqual(bbox[0], 36)
        self.assertLessEqual(bbox[2], 720 - 36)


if __name__ == "__main__":
    unittest.main()
