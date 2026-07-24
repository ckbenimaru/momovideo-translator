import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from videocaptioner.ui.thread.qwen_model_download_thread import (
    download_qwen_model,
    validate_qwen_model,
)
from videocaptioner.ui.thread.qwen_runtime_thread import qwen_runtime_commands


class QwenInstallationTests(unittest.TestCase):
    def test_model_download_uses_requested_repo_and_validates_weights(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "Qwen3-ASR-1.7B-hf"

            def fake_download(**kwargs):
                self.assertEqual(kwargs["repo_id"], "Qwen/Qwen3-ASR-1.7B-hf")
                self.assertEqual(Path(kwargs["local_dir"]), target)
                self.assertEqual(
                    Path(kwargs["cache_dir"]),
                    target.parent / ".huggingface-cache" / "hub",
                )
                target.mkdir(parents=True, exist_ok=True)
                (target / "config.json").write_text("{}", encoding="utf-8")
                (target / "model.safetensors").write_bytes(b"weights")

            with patch(
                "videocaptioner.ui.thread.qwen_model_download_thread.snapshot_download",
                side_effect=fake_download,
            ):
                self.assertEqual(
                    download_qwen_model("Qwen/Qwen3-ASR-1.7B-hf", target), target
                )

    def test_incomplete_qwen_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir)
            (target / "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "safetensors"):
                validate_qwen_model(target)

    def test_runtime_install_is_isolated_and_pins_transformers(self):
        commands = qwen_runtime_commands("X:/QwenRuntime")
        self.assertIn("venv", commands[0])
        flattened = [item for command in commands for item in command]
        self.assertIn("transformers==5.14.1", flattened)
        self.assertTrue(any("torch" in item for item in flattened))
        self.assertIn("--no-cache-dir", flattened)


if __name__ == "__main__":
    unittest.main()
