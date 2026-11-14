"""Tests for VideoSynthesisThread."""

import os
from pathlib import Path

import pytest

from tests.test_thread.conftest import run_thread_with_timeout
from videocaptioner.core.entities import SynthesisConfig, SynthesisTask
from videocaptioner.ui.task_factory import TaskFactory
from videocaptioner.ui.thread.video_synthesis_thread import VideoSynthesisThread


def test_mov_synthesis_defaults_to_mp4(tmp_path):
    video_path = str(tmp_path / "clip.MOV")
    expected = str(tmp_path / "clip.done.mp4")
    assert TaskFactory.create_synthesis_task(video_path, "clip.srt").output_path == expected
    assert TaskFactory.create_full_process_task(video_path).output_path == expected


def test_matching_mov_and_mp4_stems_have_distinct_outputs(tmp_path):
    mov, mp4 = tmp_path / "clip.MOV", tmp_path / "clip.mp4"
    mov.touch()
    mp4.touch()
    assert TaskFactory.create_synthesis_task(str(mov), "clip.srt").output_path == str(
        tmp_path / "clip.mov.done.mp4"
    )
    assert TaskFactory.create_synthesis_task(str(mp4), "clip.srt").output_path == str(
        tmp_path / "clip.done.mp4"
    )


@pytest.mark.integration
class TestVideoSynthesisThread:
    """Test suite for VideoSynthesisThread."""

    @pytest.fixture
    def base_config(self) -> SynthesisConfig:
        """Create base synthesis configuration."""
        return SynthesisConfig(
            soft_subtitle=False,
            need_video=True,
        )

    def test_synthesize_skip_video(
        self,
        sample_video_path: str,
        sample_subtitle_path: str,
        output_dir: str,
        base_config: SynthesisConfig,
        qapp,
    ):
        """Test synthesis with need_video=False."""
        base_config.need_video = False
        output_path = os.path.join(output_dir, "output_skip.mp4")
        task = SynthesisTask(
            video_path=sample_video_path,
            subtitle_path=sample_subtitle_path,
            synthesis_config=base_config,
            output_path=output_path,
        )

        thread = VideoSynthesisThread(task)
        results = run_thread_with_timeout(thread, timeout_ms=5000)

        assert results["error"] is None, "Thread should not error when skipping video"
        assert results["finished"], "Thread should finish successfully"
        assert not Path(output_path).exists(), "Output file should not be created"

    def test_synthesize_missing_video(
        self,
        sample_subtitle_path: str,
        output_dir: str,
        base_config: SynthesisConfig,
        qapp,
    ):
        """Test synthesis with missing video file."""
        output_path = os.path.join(output_dir, "output.mp4")
        task = SynthesisTask(
            video_path="/nonexistent/video.mp4",
            subtitle_path=sample_subtitle_path,
            synthesis_config=base_config,
            output_path=output_path,
        )

        thread = VideoSynthesisThread(task)
        results = run_thread_with_timeout(thread, timeout_ms=5000)

        assert results["error"] is not None, "Expected error for missing video"

    def test_synthesize_empty_paths(
        self, output_dir: str, base_config: SynthesisConfig, qapp
    ):
        """Test synthesis with empty paths."""
        output_path = os.path.join(output_dir, "output.mp4")
        task = SynthesisTask(
            video_path="",
            subtitle_path="",
            synthesis_config=base_config,
            output_path=output_path,
        )

        thread = VideoSynthesisThread(task)
        results = run_thread_with_timeout(thread, timeout_ms=5000)

        assert results["error"] is not None, "Expected error for empty paths"
