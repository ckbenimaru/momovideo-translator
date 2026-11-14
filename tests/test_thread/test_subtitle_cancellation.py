from threading import Event
from time import monotonic

from PyQt5.QtCore import QEventLoop, QTimer
from diskcache import Cache

from videocaptioner.core.entities import SubtitleConfig, SubtitleTask, TranslatorServiceEnum
from videocaptioner.core.translate.llm_translator import LLMTranslator
from videocaptioner.core.translate.types import TargetLanguage
from videocaptioner.ui.thread import subtitle_thread
from videocaptioner.ui.view.subtitle_interface import SubtitleInterface


def test_cancel_drains_translation_before_starting_replacement(qapp, tmp_path, monkeypatch):
    for name in ("info", "success", "error"):
        monkeypatch.setattr(
            "videocaptioner.ui.view.subtitle_interface.InfoBar." + name,
            lambda *args, **kwargs: None,
        )
    original = tmp_path / "original.srt"
    original.write_text(
        "".join(
            f"{index}\n00:00:{index:02},000 --> 00:00:{index:02},500\nCue {index}\n\n"
            for index in range(1, 7)
        ),
        encoding="utf-8",
    )
    started = Event()
    release = Event()
    first_calls = []
    output = tmp_path / "cancelled.srt"
    replacement_output = tmp_path / "replacement.srt"

    cache = Cache(str(tmp_path / "cache"))

    def make_slow_translator(config, prompt, callback):
        translator = LLMTranslator(
            1, 1, TargetLanguage.TRADITIONAL_CHINESE, "test", prompt, False, callback
        )
        translator._cache = cache

        def translate(chunk):
            first_calls.append(chunk[0].index)
            started.set()
            assert release.wait(5)
            chunk[0].translated_text = "測試字幕"
            return chunk

        translator._translate_chunk = translate
        return translator

    monkeypatch.setattr(subtitle_thread, "create_translator_from_config", make_slow_translator)
    view = SubtitleInterface()
    view.set_task(
        SubtitleTask(
            subtitle_path=str(original),
            output_path=str(output),
            subtitle_config=SubtitleConfig(
                translator_service=TranslatorServiceEnum.GOOGLE,
                need_split=False,
                need_translate=True,
                target_language=TargetLanguage.TRADITIONAL_CHINESE,
            ),
        )
    )
    view.process()
    first_worker = view.subtitle_optimization_thread
    try:
        assert started.wait(2)
        before = monotonic()
        view.cancel_optimization()
        view.set_task(
            SubtitleTask(
                subtitle_path=str(original),
                output_path=str(replacement_output),
                subtitle_config=SubtitleConfig(need_split=False),
            )
        )
        view.process()
        assert monotonic() - before < 0.5
        assert view.subtitle_optimization_thread is first_worker
        assert not view.start_button.isEnabled()

        loop = QEventLoop()
        timer = QTimer()
        timer.timeout.connect(lambda: loop.quit() if not view._is_processing() else None)
        timer.start(10)
        QTimer.singleShot(4000, loop.quit)
        release.set()
        loop.exec()
        timer.stop()

        assert not view._is_processing()
        assert first_calls == [1]
        assert not output.exists()
        assert replacement_output.is_file()
        assert view.start_button.isEnabled()
    finally:
        release.set()
        worker = view.subtitle_optimization_thread
        if worker is not None:
            worker.stop()
            worker.wait(5000)
        qapp.processEvents()
        view.close()
        cache.close()
