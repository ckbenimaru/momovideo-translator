import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from videocaptioner.core.entities import TranscribeModelEnum
from videocaptioner.ui.components.Qwen3ASRSettingWidget import Qwen3ASRSettingWidget
from videocaptioner.ui.components.transcription_setting_card import (
    TranscriptionSettingCard,
)
from videocaptioner.ui.thread.qwen_runtime_thread import QWEN_RUNTIME_PACKAGES


class QwenSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_qwen_engine_exists_in_model_selector_enum(self):
        self.assertIn(TranscribeModelEnum.QWEN3_ASR, list(TranscribeModelEnum))

    def test_qwen_settings_exposes_asr_and_aligner_management(self):
        widget = Qwen3ASRSettingWidget()
        self.assertEqual(widget.model_card.comboBox.count(), 1)
        self.assertIn("1.7B", widget.asr_download_card.titleLabel.text())
        self.assertIn("ForcedAligner", widget.aligner_download_card.titleLabel.text())
        self.assertIn("background: transparent", widget.container.styleSheet())
        self.assertIn("background: transparent", widget.scroll_area.styleSheet())
        self.assertIn(
            "background: transparent", widget.scroll_area.viewport().styleSheet()
        )

    def test_transcription_card_can_switch_to_qwen(self):
        card = TranscriptionSettingCard()
        card.on_model_changed(TranscribeModelEnum.QWEN3_ASR.value)
        self.assertIs(card.stacked_widget.currentWidget(), card.qwen3_asr_widget)

    def test_progress_is_visible_on_active_install_card(self):
        widget = Qwen3ASRSettingWidget()
        widget._active_card = widget.runtime_install_card
        widget._set_busy(True)
        widget._on_progress(46, "安装 CUDA PyTorch")
        self.assertEqual(widget.progress_bar.value(), 46)
        self.assertIn("CUDA PyTorch", widget.runtime_install_card.contentLabel.text())

    def test_runtime_uses_native_transformers_qwen_api(self):
        self.assertIn("transformers==5.14.1", QWEN_RUNTIME_PACKAGES)


if __name__ == "__main__":
    unittest.main()
