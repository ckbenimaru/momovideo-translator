import json
from threading import Event

from PIL import Image
from PyQt5.QtCore import QEventLoop, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication

from videocaptioner.core.entities import SubtitleLayoutEnum, SubtitleRenderModeEnum
from videocaptioner.ui.view import subtitle_style_interface as ui


def test_preview_finishes_before_rendering_latest_settings(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(ui, "SUBTITLE_STYLE_PATH", tmp_path / "styles")
    monkeypatch.setattr(ui.cfg, "file", tmp_path / "settings.json")
    monkeypatch.setattr(ui.cfg.subtitle_render_mode, "value", SubtitleRenderModeEnum.ROUNDED_BG)
    monkeypatch.setattr(ui.cfg.subtitle_style_name, "value", "youtube-modern")

    started = Event()
    release = Event()
    sizes = []
    paths = []

    def render(**kwargs):
        size = kwargs["style"].font_size
        sizes.append(size)
        if len(sizes) == 1:
            started.set()
            release.wait(5)
        path = tmp_path / f"preview-{size}.png"
        paths.append(path)
        Image.new("RGB", (size, 20), "black").save(path)
        return str(path)

    monkeypatch.setattr(ui, "render_preview", render)
    view = ui.SubtitleStyleInterface()
    try:
        assert started.wait(2)
        worker = view.preview_thread
        for size in (44, 37, 30):
            view.roundedFontSizeCard.spinBox.setValue(size)
        assert view.preview_thread is worker
        assert sizes == [42]

        release.set()
        loop = QEventLoop()
        timer = QTimer()
        timer.timeout.connect(
            lambda: loop.quit() if view.preview_thread is None else None
        )
        timer.start(10)
        QTimer.singleShot(5000, loop.quit)
        loop.exec()
        timer.stop()

        assert view.preview_thread is None
        assert sizes == [42, 30]
        assert view.previewImage.image.width() == 30
        assert not any(path.exists() for path in paths)
    finally:
        release.set()
        view._stopPreview()
        view.close()
        app.processEvents()


def test_rounded_controls_update_preview_and_synthesis_settings(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(ui, "SUBTITLE_STYLE_PATH", tmp_path / "styles")
    monkeypatch.setattr(ui.cfg, "file", tmp_path / "settings.json")
    monkeypatch.setattr(ui.cfg.subtitle_render_mode, "value", SubtitleRenderModeEnum.ROUNDED_BG)
    monkeypatch.setattr(ui.cfg.subtitle_style_name, "value", "youtube-modern")
    monkeypatch.setattr(ui.cfg.subtitle_layout, "value", SubtitleLayoutEnum.ONLY_TRANSLATE)
    monkeypatch.setattr(ui.cfg.use_subtitle_style, "value", True)
    for key in (
        "font_name", "font_size", "text_color", "color", "corner_radius",
        "padding_h", "padding_v", "margin_bottom", "line_spacing", "letter_spacing",
    ):
        item = getattr(ui.cfg, f"rounded_bg_{key}")
        monkeypatch.setattr(item, "value", item.value)

    def color_change(card, color):
        # Accept a color through the same signal as the picker dialog.
        picker = card.colorPicker
        picker.setColor(color)
        picker.colorChanged.emit(color)

    def wait_for_image():
        loop = QEventLoop()
        timer = QTimer()
        timer.timeout.connect(lambda: loop.quit() if view.preview_thread is None else None)
        timer.start(10)
        QTimer.singleShot(5000, loop.quit)
        loop.exec()
        timer.stop()
        assert view.preview_thread is None
        assert not view.previewImage.image.isNull()
        return view.previewImage.image.copy()

    view = ui.SubtitleStyleInterface()
    try:
        previous = wait_for_image()
        edits = (
            ("font_name", "LXGW WenKai", lambda: view.roundedFontCard.setCurrentText("LXGW WenKai")),
            ("font_name", "", lambda: view.roundedFontCard.setCurrentIndex(0)),
            ("font_size", 36, lambda: view.roundedFontSizeCard.spinBox.setValue(36)),
            ("text_color", "#ff0000", lambda: color_change(view.roundedTextColorCard, QColor("red"))),
            (
                "bg_color",
                "#0078e6dc",
                lambda: color_change(view.roundedBgColorCard, QColor(0, 120, 230, 220)),
            ),
            ("corner_radius", 0, lambda: view.roundedCornerRadiusCard.spinBox.setValue(0)),
            ("padding_h", 50, lambda: view.roundedPaddingHCard.spinBox.setValue(50)),
            ("padding_v", 24, lambda: view.roundedPaddingVCard.spinBox.setValue(24)),
            ("margin_bottom", 150, lambda: view.roundedMarginBottomCard.spinBox.setValue(150)),
            ("line_spacing", 4, lambda: view.roundedLineSpacingCard.spinBox.setValue(4)),
            ("line_spacing", 0, lambda: view.roundedLineSpacingCard.spinBox.setValue(0)),
            ("letter_spacing", 5, lambda: view.roundedLetterSpacingCard.spinBox.setValue(5)),
        )
        for field, value, edit in edits:
            edit()
            current = wait_for_image()
            assert current != previous, field
            previous = current
            task = ui.TaskFactory.create_synthesis_task("video.mov", "video.srt")
            assert task.synthesis_config.rounded_style[field] == value

        preset = tmp_path / "styles" / "rounded-youtube-modern.json"
        saved = json.loads(preset.read_text(encoding="utf-8"))
        assert saved["font_name"] == ""
        assert saved["margin_bottom"] == 150
        assert saved["bg_color"] == "#0078e6dc"
        view.loadStyle("youtube-modern")
        assert wait_for_image() == previous
    finally:
        view._stopPreview()
        view.close()
        app.processEvents()
