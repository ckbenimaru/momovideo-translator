import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from videocaptioner.core.asr.qwen3_asr import (
    QWEN_ALIGNER_DIR_NAME,
    Qwen3ASR,
    qwen_runtime_python,
    resolve_qwen_paths,
)
from videocaptioner.core.asr.qwen3_worker import (
    _hard_split,
    deduplicate_overlapping_segments,
    detected_language_summary,
    fallback_segments,
    group_speech_intervals,
    mlx_aligned_segments,
    mlx_fallback_segments,
    try_mlx_alignment,
)
from videocaptioner.core.entities import TranscribeConfig


class Qwen3ASRTests(unittest.TestCase):
    def test_mlx_alignment_missing_tokenizer_uses_estimated_times(self):
        aligner = Mock()
        aligner.generate.side_effect = ImportError("Japanese tokenization requires nagisa")
        result = try_mlx_alignment(aligner, None, "こんにちは", "japanese", 10, 12, 1, 60)
        self.assertEqual(result, [])
        estimates = mlx_fallback_segments("こんにちは", 10, 12)
        self.assertEqual("".join(item["text"] for item in estimates), "こんにちは")
        self.assertEqual(estimates[0]["start_time"], 10)
        self.assertEqual(estimates[-1]["end_time"], 12)

    def test_mlx_alignment_keeps_punctuation_and_rejects_missing_words(self):
        alignment = [
            SimpleNamespace(text="Hello", start_time=0.2, end_time=0.5),
            SimpleNamespace(text="世界", start_time=0.6, end_time=1.0),
        ]
        segments = mlx_aligned_segments("Hello, 世界！", alignment, 10, 12)
        self.assertEqual([segment["text"] for segment in segments], ["Hello,", "世界！"])
        self.assertEqual(segments[0]["start_time"], 10.2)
        self.assertEqual(mlx_aligned_segments("Hello there, 世界！", alignment, 10, 12), [])

    def test_mlx_estimated_segments_keep_punctuation_with_words(self):
        segments = mlx_fallback_segments("Don't stop. 季候風！", 10, 30)
        self.assertEqual([segment["text"] for segment in segments], ["Don't", "stop.", "季", "候", "風！"])
        self.assertTrue(10 < segments[-1]["end_time"] <= 30)

    def test_result_seconds_are_converted_to_milliseconds(self):
        engine = Qwen3ASR.__new__(Qwen3ASR)
        segments = engine._make_segments(
            {
                "segments": [
                    {"text": "你好", "start_time": 1.25, "end_time": 2.5}
                ]
            }
        )
        self.assertEqual(segments[0].text, "你好")
        self.assertEqual(segments[0].start_time, 1250)
        self.assertEqual(segments[0].end_time, 2500)

    def test_result_preserves_per_segment_language(self):
        engine = Qwen3ASR.__new__(Qwen3ASR)
        segments = engine._make_segments(
            {
                "segments": [
                    {
                        "text": "Привет",
                        "start_time": 1.0,
                        "end_time": 2.0,
                        "language": "Russian",
                    }
                ]
            }
        )
        self.assertEqual(segments[0].language, "Russian")

    def test_installation_requires_runtime_asr_and_aligner(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime = root / "runtime"
            python = qwen_runtime_python(runtime)
            python.parent.mkdir(parents=True)
            python.write_bytes(b"python")
            model_root = root / "models"
            asr_path = model_root / "Qwen3-ASR-1.7B-hf"
            aligner_path = model_root / QWEN_ALIGNER_DIR_NAME
            asr_path.mkdir(parents=True)
            aligner_path.mkdir(parents=True)
            (asr_path / "config.json").write_text("{}", encoding="utf-8")
            (aligner_path / "config.json").write_text("{}", encoding="utf-8")
            (asr_path / "model.safetensors").write_bytes(b"weights")
            (aligner_path / "model.safetensors").write_bytes(b"weights")

            engine = Qwen3ASR.__new__(Qwen3ASR)
            engine.runtime_dir = runtime
            engine.model_dir = model_root
            engine.asr_model = "Qwen3-ASR-1.7B-hf"
            engine.device = "cpu"
            engine.use_forced_aligner = True
            self.assertEqual(engine._validate_installation(), python)

            (aligner_path / "config.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "Forced Aligner"):
                engine._validate_installation()

    def test_hard_split_never_exceeds_budget(self):
        chunks = list(_hard_split(0, 1000, 300))
        self.assertEqual(chunks, [(0, 300), (300, 600), (600, 900), (900, 1000)])

    def test_hard_split_keeps_quarter_second_overlap(self):
        chunks = list(_hard_split(0, 1000, 300, overlap_samples=50))
        self.assertEqual(chunks, [(0, 300), (250, 550), (500, 800), (750, 1000)])

    def test_mixed_language_chunks_split_on_one_second_pause(self):
        chunks = group_speech_intervals(
            [(0, 6000), (6200, 12000), (13500, 18000)],
            total_samples=20000,
            sample_rate=1000,
        )
        self.assertEqual(chunks, [(0, 12200), (13300, 18200)])

    def test_uninterrupted_speech_hard_splits_at_twenty_seconds(self):
        chunks = group_speech_intervals(
            [(0, 45000)],
            total_samples=45000,
            sample_rate=1000,
        )
        self.assertEqual(chunks, [(0, 20000), (19750, 39750), (39500, 45000)])

    def test_overlap_deduplication_keeps_language(self):
        segments = deduplicate_overlapping_segments(
            [
                {"text": "мир", "start_time": 19.8, "end_time": 20.1, "language": "Russian"},
                {"text": "мир", "start_time": 19.85, "end_time": 20.12, "language": "Russian"},
                {"text": "你好", "start_time": 20.4, "end_time": 20.8, "language": "Chinese"},
            ]
        )
        self.assertEqual([item["text"] for item in segments], ["мир", "你好"])
        self.assertEqual(segments[0]["language"], "Russian")

    def test_multiple_languages_do_not_use_first_chunk_as_video_language(self):
        overall, languages = detected_language_summary(
            [{"language": "Chinese"}, {"language": "Russian"}, {"language": "Chinese"}]
        )
        self.assertEqual(overall, "Multilingual")
        self.assertEqual(languages, ["Chinese", "Russian"])

    def test_qwen_default_chunk_setting_is_thirty_seconds(self):
        self.assertEqual(TranscribeConfig().qwen_chunk_seconds, 30)

    def test_fallback_timestamps_preserve_text_units(self):
        segments = fallback_segments("Hello 世界！", 2.0, 4.0)
        self.assertEqual("".join(item["text"] for item in segments), "Hello世界！")
        self.assertEqual(segments[0]["start_time"], 2.0)
        self.assertEqual(segments[-1]["end_time"], 4.0)

    def test_qwen_defaults_follow_existing_external_ai_drive(self):
        runtime, models = resolve_qwen_paths(
            faster_whisper_model_dir="D:/AI/WhisperModels"
        )
        self.assertEqual(runtime, Path("D:/AI/Qwen3-ASR-Runtime"))
        self.assertEqual(models, Path("D:/AI/QwenModels"))


if __name__ == "__main__":
    unittest.main()
