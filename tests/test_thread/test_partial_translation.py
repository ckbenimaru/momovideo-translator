from pathlib import Path

import pytest
from PyQt5.QtCore import QEventLoop, Qt, QTimer

from videocaptioner.core.asr.asr_data import ASRData, ASRDataSeg
from videocaptioner.core.entities import (
    SubtitleConfig,
    SubtitleLayoutEnum,
    SubtitleTask,
    TranslatorServiceEnum,
)
from videocaptioner.core.translate.base import BaseTranslator
from videocaptioner.core.translate.types import TRANSLATION_REVIEW_MARKER, TargetLanguage
from videocaptioner.ui.thread import subtitle_thread, video_synthesis_thread
from videocaptioner.ui.view import subtitle_interface as ui


class SampleTranslator(BaseTranslator):
    def __init__(self, callback, fail_on, requested):
        super().__init__(1, 1, TargetLanguage.TRADITIONAL_CHINESE, callback)
        self.fail_on = fail_on
        self.requested = requested

    def _translate_chunk(self, chunk):
        cue = chunk[0]
        self.requested.append((cue.original_text, cue.previous_context, cue.next_context))
        if cue.original_text in self.fail_on:
            raise RuntimeError("Test provider rejected a cue")
        cue.translated_text = {"行です": "可以", "待ちます": "請等一下", "帰ります": "回家了"}[cue.original_text]
        return chunk


def test_partial_mode_reports_provider_error_when_no_cues_translate(monkeypatch):
    translator = SampleTranslator(None, set(), [])

    def disconnected(chunk):
        raise RuntimeError("Cannot connect to local translation service")

    monkeypatch.setattr(translator, "_safe_translate_chunk", disconnected)
    try:
        with pytest.raises(RuntimeError, match="Cannot connect to local translation service"):
            translator.translate_subtitle(
                ASRData([ASRDataSeg("待ちます", 0, 1000)]), allow_partial=True
            )
    finally:
        translator.stop(wait=True)


def wait_for_review(view):
    loop = QEventLoop()
    poll = QTimer()
    poll.timeout.connect(lambda: loop.quit() if not view._is_processing() else None)
    poll.start(10)
    QTimer.singleShot(5000, loop.quit)
    loop.exec()
    poll.stop()
    assert not view._is_processing()


@pytest.mark.parametrize("marked", [False, True])
def test_review_retry_resume_and_skip(qapp, tmp_path, monkeypatch, marked):
    monkeypatch.setattr(ui.cfg, "file", tmp_path / "settings.json")
    monkeypatch.setattr(ui.cfg.subtitle_layout, "value", SubtitleLayoutEnum.ONLY_TRANSLATE)
    monkeypatch.setattr(ui.cfg.need_translate, "value", True)
    monkeypatch.setattr(ui.cfg.translator_service, "value", TranslatorServiceEnum.GOOGLE)
    monkeypatch.setattr(ui.cfg.target_language, "value", TargetLanguage.TRADITIONAL_CHINESE)
    for name in ("info", "success", "error"):
        monkeypatch.setattr(ui.InfoBar, name, lambda *args, **kwargs: None)
    monkeypatch.setattr("videocaptioner.core.translate.base.time.sleep", lambda seconds: None)

    source = tmp_path / "original.srt"
    source.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n行です\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\n待ちます\n\n"
        "3\n00:00:02,000 --> 00:00:03,000\n帰ります\n",
        encoding="utf-8",
    )
    video = tmp_path / "test.mov"
    video.touch()
    output = tmp_path / "complete.srt"
    config = SubtitleConfig(
        need_translate=True,
        target_language=TargetLanguage.TRADITIONAL_CHINESE,
        subtitle_layout=SubtitleLayoutEnum.ONLY_TRANSLATE,
        translator_service=TranslatorServiceEnum.GOOGLE,
        need_split=False,
    )
    fail_on = {"帰ります"} if marked else {"待ちます", "帰ります"}
    review_text = f"稍等{TRANSLATION_REVIEW_MARKER}"
    requested = []

    class TestCache:
        def get(self, key, default=None):
            return default

        def set(self, *args, **kwargs):
            pass

    class ReviewTranslator(SampleTranslator):
        def _translate_chunk(self, chunk):
            result = super()._translate_chunk(chunk)
            cue = result[0]
            if marked and cue.original_text == "待ちます":
                cue.translated_text = (
                    f"稍候{TRANSLATION_REVIEW_MARKER}" if self.force_retry else review_text
                )
            return result

    def translator(config, prompt, callback):
        result = ReviewTranslator(callback, fail_on, requested)
        result._cache = TestCache()
        return result

    monkeypatch.setattr(subtitle_thread, "create_translator_from_config", translator)
    view = ui.SubtitleInterface()
    view.set_task(SubtitleTask(
        subtitle_path=str(source), video_path=str(video), output_path=str(output),
        subtitle_config=config, need_next_task=True,
    ))
    view.process()
    wait_for_review(view)
    assert not output.exists()
    assert view._missing_translation_rows() == [1, 2]
    assert view._review_path().is_file()
    view.close()
    qapp.processEvents()

    resumed = ui.SubtitleInterface()
    resumed.load_subtitle_file(str(source))
    assert resumed._missing_translation_rows() == [1, 2]
    resumed._missing_rows_button.menu().actions()[0].trigger()
    assert resumed.subtitle_table.currentIndex().row() == 1
    fail_on.remove("帰ります")
    requested.clear()
    resumed._retry_missing_button.click()
    wait_for_review(resumed)
    assert {x[0] for x in requested} == {"待ちます", "帰ります"}
    assert ("待ちます", "行です", "帰ります") in requested
    assert resumed._missing_translation_rows() == [1]
    assert resumed.model._data["2"]["translated_subtitle"] == (review_text if marked else "")
    assert resumed.model._data["1"]["translated_subtitle"] == "可以"
    assert resumed.model._data["3"]["translated_subtitle"] == "回家了"

    handoff = []
    resumed.finished.connect(lambda *args: handoff.append(args))
    resumed._continue_button.click()
    assert handoff == [(str(video), str(resumed._review_path()))]

    from videocaptioner.ui.task_factory import TaskFactory

    rendered = []

    def soft_subtitles(video_file, subtitle_path, output, **kwargs):
        rendered.append(Path(subtitle_path).read_text(encoding="utf-8"))

    monkeypatch.setattr(video_synthesis_thread, "add_subtitles", soft_subtitles)
    for layout in (SubtitleLayoutEnum.ONLY_TRANSLATE, SubtitleLayoutEnum.ORIGINAL_ON_TOP):
        synthesis = TaskFactory.create_synthesis_task(*handoff[0])
        synthesis.synthesis_config.soft_subtitle = True
        synthesis.synthesis_config.need_video = True
        synthesis.synthesis_config.subtitle_layout = layout
        worker = video_synthesis_thread.VideoSynthesisThread(synthesis)
        errors = []
        worker.error.connect(errors.append)
        worker.run()
        assert not errors
    assert "待ちます" not in rendered[0]
    assert ("00:00:01,000 --> 00:00:02,000" in rendered[0]) is marked
    assert (review_text in rendered[0]) is marked
    assert "00:00:02,000 --> 00:00:03,000" in rendered[0]
    assert "待ちます" in rendered[1]
    resumed.model.setData(resumed.model.index(1, 3), "稍等一下", Qt.EditRole)
    assert resumed._missing_translation_rows() == []
    resumed.close()
    qapp.processEvents()
