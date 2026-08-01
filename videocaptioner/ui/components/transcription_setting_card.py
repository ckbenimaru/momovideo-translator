from typing import TYPE_CHECKING, Optional

from PyQt5.QtWidgets import (
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from videocaptioner.core.entities import (
    TranscribeModelEnum,
)
from videocaptioner.core.utils.platform_utils import is_macos

from .WhisperAPISettingWidget import WhisperAPISettingWidget
from .WhisperCppSettingWidget import WhisperCppSettingWidget

if TYPE_CHECKING:
    from .FasterWhisperSettingWidget import FasterWhisperSettingWidget
    from .Qwen3ASRSettingWidget import Qwen3ASRSettingWidget


class TranscriptionSettingCard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        # 设置界面堆叠
        self.stacked_widget = QStackedWidget(self)

        # 添加各个设置界面
        self.empty_widget = QWidget(self)  # 添加空白页面作为默认显示
        self.whisper_cpp_widget = WhisperCppSettingWidget(self)
        self.whisper_api_widget = WhisperAPISettingWidget(self)

        # FasterWhisper 在 macOS 上不可用
        self.faster_whisper_widget: Optional["FasterWhisperSettingWidget"] = None
        self.qwen3_asr_widget: Optional["Qwen3ASRSettingWidget"] = None

        self.stacked_widget.addWidget(self.empty_widget)  # 添加空白页面
        self.stacked_widget.addWidget(self.whisper_cpp_widget)
        self.stacked_widget.addWidget(self.whisper_api_widget)
        self.main_layout.addWidget(self.stacked_widget)

    def _ensure_faster_whisper_widget(self):
        if self.faster_whisper_widget is None and not is_macos():
            from .FasterWhisperSettingWidget import FasterWhisperSettingWidget

            self.faster_whisper_widget = FasterWhisperSettingWidget(self)
            self.stacked_widget.addWidget(self.faster_whisper_widget)
        return self.faster_whisper_widget

    def _ensure_qwen3_asr_widget(self):
        if self.qwen3_asr_widget is None and not is_macos():
            from .Qwen3ASRSettingWidget import Qwen3ASRSettingWidget

            self.qwen3_asr_widget = Qwen3ASRSettingWidget(self)
            self.stacked_widget.addWidget(self.qwen3_asr_widget)
        return self.qwen3_asr_widget

    def on_model_changed(self, value):
        # 切换对应的设置界面
        if value == TranscribeModelEnum.WHISPER_CPP.value:
            self.stacked_widget.setCurrentWidget(self.whisper_cpp_widget)
        elif value == TranscribeModelEnum.WHISPER_API.value:
            self.stacked_widget.setCurrentWidget(self.whisper_api_widget)
        elif value == TranscribeModelEnum.FASTER_WHISPER.value:
            widget = self._ensure_faster_whisper_widget()
            self.stacked_widget.setCurrentWidget(widget or self.empty_widget)
        elif value == TranscribeModelEnum.QWEN3_ASR.value:
            widget = self._ensure_qwen3_asr_widget()
            self.stacked_widget.setCurrentWidget(widget or self.empty_widget)
        else:
            self.stacked_widget.setCurrentWidget(self.empty_widget)
