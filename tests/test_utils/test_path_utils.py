import unittest
from pathlib import Path
from unittest.mock import patch

from videocaptioner.core.utils.path_utils import sanitize_filename_component
from videocaptioner.ui.common.config import cfg
from videocaptioner.ui.task_factory import TaskFactory


class PathUtilsTests(unittest.TestCase):
    def test_trailing_space_and_dot_are_removed(self):
        self.assertEqual(sanitize_filename_component("Rona . "), "Rona")

    def test_invalid_and_reserved_windows_names_are_safe(self):
        self.assertEqual(sanitize_filename_component('a:b?c*'), "a_b_c_")
        self.assertEqual(sanitize_filename_component("CON.txt"), "CON_.txt")

    def test_long_names_are_stable_and_bounded(self):
        value = sanitize_filename_component("長" * 300)
        self.assertLessEqual(len(value), 120)
        self.assertEqual(value, sanitize_filename_component("長" * 300))

    def test_transcription_work_path_uses_sanitized_stem(self):
        with patch.object(
            cfg.work_dir,
            "_ConfigItem__value",
            "D:/MomoWork",
        ):
            task = TaskFactory.create_transcribe_task(
                "D:/下載/節目 Rona .mp4",
                need_next_task=True,
            )
        output = Path(task.output_path)
        self.assertEqual(output.parent.parent.name, "節目 Rona")
        self.assertEqual(output.parent.name, "subtitle")

    def test_synthesis_done_name_has_no_trailing_space(self):
        task = TaskFactory.create_synthesis_task(
            "D:/下載/節目 Rona .mp4",
            "D:/下載/subtitle.ass",
        )
        self.assertEqual(Path(task.output_path).name, "節目 Rona.done.mp4")


if __name__ == "__main__":
    unittest.main()
