from types import SimpleNamespace

from videocaptioner.core.entities import SubtitleLayoutEnum
from videocaptioner.core.optimize import optimize
from videocaptioner.ui.task_factory import TaskFactory
from videocaptioner.ui.view import subtitle_interface as ui


def test_translation_only_layout_runs_translation_even_with_old_settings(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(ui.cfg, "file", tmp_path / "settings.json")
    monkeypatch.setattr(ui.cfg.need_translate, "value", False)
    monkeypatch.setattr(ui.cfg.subtitle_layout, "value", SubtitleLayoutEnum.ONLY_TRANSLATE)

    task = TaskFactory.create_subtitle_task(str(tmp_path / "source.srt"))
    assert task.subtitle_config.need_translate

    view = ui.SubtitleInterface()
    try:
        assert view.translate_button.isChecked()
        assert view.target_language_button.isEnabled()
        view.on_subtitle_translation_changed(False)
        assert ui.cfg.subtitle_layout.value is SubtitleLayoutEnum.ONLY_ORIGINAL
        assert not view.target_language_button.isEnabled()
        view.on_subtitle_layout_changed(SubtitleLayoutEnum.ONLY_TRANSLATE.value)
        assert ui.cfg.need_translate.value
        assert view.translate_button.isChecked()
    finally:
        view.close()
        qapp.processEvents()


def test_correction_preserves_source_after_invalid_model_replies(monkeypatch):
    original = {"1": "スポーツとかはやられてますか？"}
    reply = SimpleNamespace(choices=[
        SimpleNamespace(message=SimpleNamespace(content='{"1": "Do you play sports?"}'))
    ])
    monkeypatch.setattr(optimize, "call_llm", lambda **kwargs: reply)
    optimizer = optimize.SubtitleOptimizer(1, 1, "test", "")
    try:
        assert optimizer.agent_loop(original) == original
    finally:
        optimizer.stop()
