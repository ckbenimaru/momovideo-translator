import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from videocaptioner.ui.thread.modelscope_download_thread import (
    ModelscopeDownloadThread,
    download_model_snapshot,
    validate_model_directory,
)


class ModelscopeDownloadThreadTests(unittest.TestCase):
    @staticmethod
    def _write_complete_model(path: Path) -> None:
        for name in ("model.bin", "config.json", "tokenizer.json"):
            (path / name).write_bytes(b"model-data")

    def test_download_uses_current_modelscope_api_and_requested_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "faster-whisper-large-v3"

            def fake_snapshot_download(**kwargs):
                self.assertEqual(kwargs["model_id"], "owner/model")
                self.assertEqual(Path(kwargs["local_dir"]), target)
                self.assertEqual(kwargs["max_workers"], 4)
                self.assertNotIn("progress_callbacks", kwargs)
                self._write_complete_model(target)

            with patch(
                "videocaptioner.ui.thread.modelscope_download_thread.snapshot_download",
                side_effect=fake_snapshot_download,
            ):
                result = download_model_snapshot("owner/model", target)

            self.assertEqual(result, target)

    def test_huggingface_is_preferred_when_available(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "faster-whisper-large-v3"

            def fake_hf_download(**kwargs):
                self.assertEqual(kwargs["repo_id"], "Systran/faster-whisper-large-v3")
                self.assertEqual(Path(kwargs["local_dir"]), target)
                self._write_complete_model(target)

            with (
                patch(
                    "videocaptioner.ui.thread.modelscope_download_thread.hf_snapshot_download",
                    side_effect=fake_hf_download,
                ),
                patch(
                    "videocaptioner.ui.thread.modelscope_download_thread.snapshot_download"
                ) as modelscope_download,
            ):
                result = download_model_snapshot(
                    "owner/model", target, "Systran/faster-whisper-large-v3"
                )

            self.assertEqual(result, target)
            modelscope_download.assert_not_called()

    def test_modelscope_is_used_when_huggingface_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "faster-whisper-large-v3"

            def fake_modelscope_download(**_kwargs):
                self._write_complete_model(target)

            with (
                patch(
                    "videocaptioner.ui.thread.modelscope_download_thread.hf_snapshot_download",
                    side_effect=OSError("Hugging Face unavailable"),
                ),
                patch(
                    "videocaptioner.ui.thread.modelscope_download_thread.snapshot_download",
                    side_effect=fake_modelscope_download,
                ) as modelscope_download,
            ):
                result = download_model_snapshot(
                    "owner/model", target, "Systran/faster-whisper-large-v3"
                )

            self.assertEqual(result, target)
            modelscope_download.assert_called_once()

    def test_incomplete_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir)
            (target / "model.bin").write_bytes(b"model-data")
            with self.assertRaisesRegex(RuntimeError, "config.json"):
                validate_model_directory(target)

    def test_failed_download_does_not_emit_success(self):
        errors = []
        successes = []
        thread = ModelscopeDownloadThread("owner/model", "unused")
        thread.error.connect(errors.append)
        thread.succeeded.connect(successes.append)

        with patch(
            "videocaptioner.ui.thread.modelscope_download_thread.download_model_snapshot",
            side_effect=RuntimeError("network failed"),
        ):
            thread.run()

        self.assertEqual(successes, [])
        self.assertEqual(errors, ["RuntimeError: network failed"])


if __name__ == "__main__":
    unittest.main()
