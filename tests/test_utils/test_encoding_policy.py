import unittest
from unittest.mock import patch

from videocaptioner.core.entities import VideoQualityEnum
from videocaptioner.core.utils.encoding_policy import (
    SourceVideoInfo,
    build_mac_encoding_args,
    build_soft_mp4_args,
    get_target_bitrate_kbps,
)


SOURCE = SourceVideoInfo(
    codec="h264",
    bit_rate=2_000_000,
    width=1920,
    height=1080,
    fps=30.0,
)


def option_value(args, option):
    return args[args.index(option) + 1]


class NvencEncodingPolicyTests(unittest.TestCase):
    @patch(
        "videocaptioner.core.utils.encoding_policy.probe_source_video",
        return_value=SOURCE,
    )
    @patch(
        "videocaptioner.core.utils.encoding_policy.is_nvenc_available",
        return_value=True,
    )
    def test_nvenc_is_preferred_with_mac_compatible_output(self, _available, _probe):
        args, _, _ = build_mac_encoding_args("input.mp4", "output.mp4", 28, "medium")
        self.assertEqual(option_value(args, "-c:v"), "h264_nvenc")
        self.assertEqual(option_value(args, "-preset"), "p5")
        self.assertEqual(option_value(args, "-rc"), "vbr")
        self.assertEqual(option_value(args, "-multipass"), "fullres")
        self.assertEqual(option_value(args, "-rc-lookahead"), "32")
        self.assertEqual(option_value(args, "-b:v"), "3000k")
        self.assertNotIn("-cq", args)
        self.assertEqual(option_value(args, "-profile:v"), "high")
        self.assertEqual(option_value(args, "-tag:v"), "avc1")
        self.assertEqual(option_value(args, "-pix_fmt"), "yuv420p")
        self.assertEqual(option_value(args, "-c:a"), "aac")
        self.assertEqual(option_value(args, "-profile:a"), "aac_low")
        self.assertEqual(option_value(args, "-b:a"), "192k")
        self.assertEqual(option_value(args, "-movflags"), "+faststart")

    @patch(
        "videocaptioner.core.utils.encoding_policy.probe_source_video",
        return_value=SOURCE,
    )
    def test_explicit_fallback_uses_libx264(self, _probe):
        args, _, _ = build_mac_encoding_args(
            "input.mp4", "output.mp4", 23, "slow", prefer_hardware=False
        )
        self.assertEqual(option_value(args, "-c:v"), "libx264")
        self.assertEqual(option_value(args, "-preset"), "slow")
        self.assertEqual(option_value(args, "-b:v"), "5000k")
        self.assertNotIn("-crf", args)
        self.assertNotIn("-cq", args)

    @patch(
        "videocaptioner.core.utils.encoding_policy.probe_source_video",
        return_value=SOURCE,
    )
    @patch(
        "videocaptioner.core.utils.encoding_policy.is_nvenc_available",
        return_value=False,
    )
    def test_unavailable_nvenc_automatically_uses_libx264(self, _available, _probe):
        args, _, _ = build_mac_encoding_args("input.mp4", "output.mp4", 28, "medium")
        self.assertEqual(option_value(args, "-c:v"), "libx264")

    @patch(
        "videocaptioner.core.utils.encoding_policy.probe_source_video",
        return_value=SOURCE,
    )
    def test_medium_tier_uses_expected_rate_budget(self, _probe):
        args, _, max_rate = build_mac_encoding_args(
            "input.mp4", "output.mp4", 28, "medium", prefer_hardware=False
        )
        self.assertEqual(max_rate, 3_750_000)
        self.assertEqual(option_value(args, "-b:v"), "3000k")
        self.assertEqual(option_value(args, "-maxrate"), "3750k")
        self.assertEqual(option_value(args, "-bufsize"), "6000k")

    def test_all_quality_tiers_have_fixed_targets(self):
        self.assertEqual(get_target_bitrate_kbps(18), 8000)
        self.assertEqual(get_target_bitrate_kbps(23), 5000)
        self.assertEqual(get_target_bitrate_kbps(28), 3000)
        self.assertEqual(get_target_bitrate_kbps(32), 1500)
        for quality in VideoQualityEnum:
            self.assertEqual(
                get_target_bitrate_kbps(quality.get_crf()),
                quality.get_bitrate_kbps(),
            )

    @patch(
        "videocaptioner.core.utils.encoding_policy.probe_source_video",
        return_value=SourceVideoInfo(
            codec="h264", bit_rate=50_000_000, width=3840, height=2160, fps=60.0
        ),
    )
    def test_fixed_tier_is_not_overridden_by_source_complexity(self, _probe):
        args, _, max_rate = build_mac_encoding_args(
            "input.mp4", "output.mp4", 32, "fast", prefer_hardware=False
        )
        self.assertEqual(option_value(args, "-b:v"), "1500k")
        self.assertEqual(max_rate, 1_875_000)


class SoftMp4EncodingPolicyTests(unittest.TestCase):
    @patch("videocaptioner.core.utils.encoding_policy._probe_streams")
    def test_only_incompatible_audio_track_is_converted(self, probe):
        probe.return_value = {
            "streams": [
                {"index": 0, "codec_type": "video", "disposition": {"attached_pic": 1}},
                {
                    "index": 1, "codec_type": "video", "codec_name": "h264",
                    "profile": "High", "pix_fmt": "yuv420p", "level": 31,
                },
                {"index": 2, "codec_type": "audio", "codec_name": "aac", "profile": "LC"},
                {"index": 3, "codec_type": "audio", "codec_name": "pcm_s16le"},
            ]
        }
        args = build_soft_mp4_args("input.mov", 28, "medium")
        self.assertEqual(args[:6], ["-map", "0:1", "-map", "0:a?", "-map", "1:s:0"])
        self.assertEqual(option_value(args, "-c:v"), "copy")
        self.assertEqual(option_value(args, "-c:a"), "copy")
        self.assertNotIn("-c:a:0", args)
        self.assertEqual(option_value(args, "-c:a:1"), "aac")
        self.assertEqual(option_value(args, "-profile:a:1"), "aac_low")
        self.assertEqual(option_value(args, "-tag:v"), "avc1")
        self.assertEqual(option_value(args, "-c:s"), "mov_text")
        self.assertEqual(option_value(args, "-movflags"), "+faststart")
        self.assertEqual(option_value(args, "-f"), "mp4")

    @patch("videocaptioner.core.utils.encoding_policy._probe_streams")
    def test_only_incompatible_video_is_converted(self, probe):
        probe.return_value = {
            "streams": [
                {"index": 0, "codec_type": "video", "codec_name": "prores"},
                {"index": 1, "codec_type": "audio", "codec_name": "aac", "profile": "LC"},
            ]
        }
        args = build_soft_mp4_args("input.mov", 28, "medium")
        self.assertEqual(option_value(args, "-c:v"), "libx264")
        self.assertEqual(option_value(args, "-pix_fmt"), "yuv420p")
        self.assertEqual(option_value(args, "-c:a"), "copy")
        self.assertNotIn("-c:a:0", args)


if __name__ == "__main__":
    unittest.main()
