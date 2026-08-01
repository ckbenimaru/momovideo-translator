from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from qfluentwidgets import (
    ComboBoxSettingCard,
    InfoBar,
    InfoBarPosition,
    ProgressBar,
    PushSettingCard,
    SettingCardGroup,
    SingleDirectionScrollArea,
    SwitchSettingCard,
)
from qfluentwidgets import FluentIcon as FIF

from videocaptioner.core.asr.qwen3_asr import (
    QWEN_ALIGNER_DIR_NAME,
    qwen_runtime_python,
    resolve_qwen_paths,
)
from videocaptioner.core.entities import Qwen3ASRModelEnum
from videocaptioner.ui.common.config import cfg
from videocaptioner.ui.components.LineEditSettingCard import LineEditSettingCard
from videocaptioner.ui.components.SpinBoxSettingCard import SpinBoxSettingCard
from videocaptioner.ui.thread.qwen_model_download_thread import (
    QwenModelDownloadThread,
    validate_qwen_model,
)
from videocaptioner.ui.thread.qwen_runtime_thread import (
    QwenRuntimeCheckThread,
    QwenRuntimeInstallThread,
    validate_qwen_runtime,
)


QWEN_ASR_REPO = "Qwen/Qwen3-ASR-1.7B-hf"
QWEN_ALIGNER_REPO = "Qwen/Qwen3-ForcedAligner-0.6B-hf"


def get_qwen_runtime_dir() -> Path:
    runtime, _models = resolve_qwen_paths(
        str(cfg.qwen_runtime_dir.value or "").strip(),
        str(cfg.qwen_model_dir.value or "").strip(),
        str(cfg.faster_whisper_model_dir.value or "").strip(),
    )
    return runtime


def get_qwen_model_root() -> Path:
    _runtime, models = resolve_qwen_paths(
        str(cfg.qwen_runtime_dir.value or "").strip(),
        str(cfg.qwen_model_dir.value or "").strip(),
        str(cfg.faster_whisper_model_dir.value or "").strip(),
    )
    return models


class Qwen3ASRSettingWidget(QWidget):
    """Settings and one-click installation for Qwen3-ASR."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Migrate the previous 120s default. Explicit non-default user values
        # remain untouched, while existing installations adopt mixed-language
        # friendly chunking immediately.
        if int(cfg.qwen_chunk_seconds.value) == 120:
            cfg.set(cfg.qwen_chunk_seconds, 30)
        self.runtime_thread: Optional[QwenRuntimeInstallThread] = None
        self.runtime_check_thread: Optional[QwenRuntimeCheckThread] = None
        self.download_thread: Optional[QwenModelDownloadThread] = None
        self._active_card = None
        self._setup_ui()
        self._connect_signals()
        self._refresh_model_status()
        self._set_initial_runtime_status()
        QTimer.singleShot(0, self.refresh_status)

    def _setup_ui(self):
        # Match the main window/Faster Whisper page: the page and its scroll
        # viewport must be transparent so qfluentwidgets can paint the current
        # light/dark theme. A plain QWidget otherwise defaults to an opaque
        # white panel, producing white text on white in dark mode.
        self.setStyleSheet("Qwen3ASRSettingWidget{background: transparent}")
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_area = SingleDirectionScrollArea(orient=Qt.Vertical, parent=self)  # type: ignore
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet(
            "QScrollArea{background: transparent; border: none}"
        )
        self.scroll_area.viewport().setStyleSheet("background: transparent")
        self.container = QWidget(self)
        self.container.setStyleSheet("QWidget{background: transparent}")
        self.container_layout = QVBoxLayout(self.container)

        self.model_group = SettingCardGroup(self.tr("Qwen3-ASR 模型"), self.container)
        self.model_card = ComboBoxSettingCard(
            cfg.qwen_asr_model,
            FIF.ROBOT,
            self.tr("语音识别模型"),
            self.tr("1.7B：最高准确率，推荐 RTX 4060 8GB 或以上"),
            [model.value for model in Qwen3ASRModelEnum],
            self.model_group,
        )
        self.aligner_card = SwitchSettingCard(
            FIF.ALIGNMENT,
            self.tr("Forced Aligner 时间轴"),
            self.tr("生成词／字级时间戳；字幕工作流建议保持开启"),
            cfg.qwen_use_forced_aligner,
            self.model_group,
        )
        self.device_card = ComboBoxSettingCard(
            cfg.qwen_device,
            FIF.SPEED_HIGH,
            self.tr("推理设备"),
            self.tr("RTX 4060 使用 CUDA；CPU 仅作兼容回退"),
            ["cuda", "cpu"],
            self.model_group,
        )
        self.chunk_card = SpinBoxSettingCard(
            cfg.qwen_chunk_seconds,
            FIF.CUT,
            self.tr("单段最长秒数"),
            self.tr("默认 30 秒；自动语言按停顿切成 5～15 秒，最长 20 秒"),
            30,
            240,
        )

        self.path_group = SettingCardGroup(self.tr("储存位置"), self.container)
        self.runtime_path_card = LineEditSettingCard(
            cfg.qwen_runtime_dir,
            FIF.CODE,
            self.tr("独立运行环境"),
            self.tr("留空使用 momo 数据目录；可改到 D 槽"),
            str(get_qwen_runtime_dir()),
            self.path_group,
        )
        self.model_path_card = LineEditSettingCard(
            cfg.qwen_model_dir,
            FIF.FOLDER,
            self.tr("模型目录"),
            self.tr("ASR 与 Aligner 模型统一储存位置"),
            str(get_qwen_model_root()),
            self.path_group,
        )

        self.install_group = SettingCardGroup(self.tr("安装与模型管理"), self.container)
        self.runtime_install_card = PushSettingCard(
            self.tr("安装／修复"),
            FIF.DOWNLOAD,
            self.tr("Qwen 推理环境"),
            self.tr("建立隔离环境并安装 CUDA PyTorch、Transformers 等依赖"),
            self.install_group,
        )
        self.asr_download_card = PushSettingCard(
            self.tr("下载"),
            FIF.DOWNLOAD,
            self.tr("Qwen3-ASR-1.7B"),
            self.tr("最高准确率语音识别模型"),
            self.install_group,
        )
        self.aligner_download_card = PushSettingCard(
            self.tr("下载"),
            FIF.DOWNLOAD,
            self.tr("Qwen3-ForcedAligner-0.6B"),
            self.tr("字幕词／字级时间轴对齐模型"),
            self.install_group,
        )
        self.refresh_card = PushSettingCard(
            self.tr("重新检查"),
            FIF.SYNC,
            self.tr("安装状态"),
            self.tr("检查运行环境、CUDA 与模型完整性"),
            self.install_group,
        )
        self.progress_bar = ProgressBar(self.container)
        self.progress_bar.hide()

        for card in (
            self.model_card,
            self.aligner_card,
            self.device_card,
            self.chunk_card,
        ):
            self.model_group.addSettingCard(card)
        for card in (self.runtime_path_card, self.model_path_card):
            self.path_group.addSettingCard(card)
        for card in (
            self.runtime_install_card,
            self.asr_download_card,
            self.aligner_download_card,
            self.refresh_card,
        ):
            self.install_group.addSettingCard(card)

        self.container_layout.addWidget(self.model_group)
        self.container_layout.addWidget(self.path_group)
        self.container_layout.addWidget(self.install_group)
        self.container_layout.addWidget(self.progress_bar)
        self.container_layout.addStretch(1)
        self.scroll_area.setWidget(self.container)
        self.main_layout.addWidget(self.scroll_area)

    def _connect_signals(self):
        self.runtime_install_card.clicked.connect(self.install_runtime)
        self.asr_download_card.clicked.connect(
            lambda: self.download_model(QWEN_ASR_REPO, QWEN_ASR_REPO.split("/")[-1])
        )
        self.aligner_download_card.clicked.connect(
            lambda: self.download_model(QWEN_ALIGNER_REPO, QWEN_ALIGNER_DIR_NAME)
        )
        self.refresh_card.clicked.connect(self.refresh_status)

    def _set_busy(self, busy: bool):
        for card in (
            self.runtime_install_card,
            self.asr_download_card,
            self.aligner_download_card,
            self.refresh_card,
        ):
            card.setEnabled(not busy)
        self.progress_bar.setVisible(busy)
        if busy:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
        else:
            self.progress_bar.setRange(0, 100)
            self._active_card = None

    def install_runtime(self):
        self._active_card = self.runtime_install_card
        self.runtime_install_card.setContent(self.tr("正在准备安装，请勿关闭 momo…"))
        self._set_busy(True)
        runtime_dir = get_qwen_runtime_dir()
        runtime_dir.parent.mkdir(parents=True, exist_ok=True)
        self.runtime_thread = QwenRuntimeInstallThread(str(runtime_dir))
        self.runtime_thread.progress.connect(self._on_progress)
        self.runtime_thread.succeeded.connect(self._on_success)
        self.runtime_thread.error.connect(self._on_error)
        self.runtime_thread.start()

    def download_model(self, repo_id: str, directory_name: str):
        self._active_card = (
            self.aligner_download_card
            if directory_name == QWEN_ALIGNER_DIR_NAME
            else self.asr_download_card
        )
        self._active_card.setContent(self.tr("正在连接模型仓库，请勿关闭 momo…"))
        self._set_busy(True)
        destination = get_qwen_model_root() / directory_name
        self.download_thread = QwenModelDownloadThread(repo_id, str(destination))
        self.download_thread.progress.connect(self._on_progress)
        self.download_thread.succeeded.connect(self._on_success)
        self.download_thread.error.connect(self._on_error)
        self.download_thread.start()

    def _on_progress(self, value: int, message: str):
        self.progress_bar.setValue(value)
        if self._active_card is not None:
            self._active_card.setContent(
                self.tr("处理中 ") + f"{value}%：" + self.tr(message)
            )

    def _on_success(self, details: str):
        self._set_busy(False)
        self.refresh_status()
        InfoBar.success(
            self.tr("完成"),
            details,
            duration=5000,
            position=InfoBarPosition.BOTTOM,
            parent=self.window(),
        )

    def _on_error(self, error: str):
        if self._active_card is not None:
            self._active_card.setContent(self.tr("失败：") + error)
        self._set_busy(False)
        InfoBar.error(
            self.tr("Qwen3-ASR 安装失败"),
            error,
            duration=10000,
            position=InfoBarPosition.BOTTOM,
            parent=self.window(),
        )

    def _set_initial_runtime_status(self):
        if qwen_runtime_python(get_qwen_runtime_dir()).is_file():
            self.runtime_install_card.setContent(self.tr("等待背景檢查…"))
        else:
            self.runtime_install_card.setContent(
                self.tr("未安裝：Python 環境不存在")
            )

    def _refresh_model_status(self):
        asr_path = get_qwen_model_root() / QWEN_ASR_REPO.split("/")[-1]
        aligner_path = get_qwen_model_root() / QWEN_ALIGNER_DIR_NAME
        try:
            validate_qwen_model(asr_path)
            self.asr_download_card.setContent(
                self.tr("已下載：") + str(asr_path)
            )
        except RuntimeError:
            self.asr_download_card.setContent(
                self.tr("未下載，約需數 GB 空間")
            )
        try:
            validate_qwen_model(aligner_path)
            self.aligner_download_card.setContent(
                self.tr("已下載：") + str(aligner_path)
            )
        except RuntimeError:
            self.aligner_download_card.setContent(
                self.tr("未下載，時間軸功能需要此模型")
            )

    def refresh_status(self):
        """Refresh model files immediately and validate CUDA in the background."""
        self._refresh_model_status()
        if (
            self.runtime_check_thread is not None
            and self.runtime_check_thread.isRunning()
        ):
            return

        runtime_dir = get_qwen_runtime_dir()
        if not qwen_runtime_python(runtime_dir).is_file():
            self._set_initial_runtime_status()
            return

        self.runtime_install_card.setContent(
            self.tr("正在背景檢查 PyTorch／CUDA…")
        )
        self.runtime_check_thread = QwenRuntimeCheckThread(str(runtime_dir))
        self.runtime_check_thread.completed.connect(
            self._on_runtime_status_checked
        )
        self.runtime_check_thread.start()

    def _on_runtime_status_checked(
        self, runtime_ok: bool, runtime_details: str
    ):
        self.runtime_install_card.setContent(
            self.tr("已安裝：") + runtime_details
            if runtime_ok
            else self.tr("未安裝：") + runtime_details
        )

    def check_qwen_installation(self) -> bool:
        runtime_ok, _details = validate_qwen_runtime(get_qwen_runtime_dir())
        if not runtime_ok:
            return False
        try:
            validate_qwen_model(
                get_qwen_model_root() / QWEN_ASR_REPO.split("/")[-1]
            )
            if cfg.qwen_use_forced_aligner.value:
                validate_qwen_model(get_qwen_model_root() / QWEN_ALIGNER_DIR_NAME)
        except RuntimeError:
            return False
        return True
