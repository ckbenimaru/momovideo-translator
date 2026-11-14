import os
from concurrent.futures import CancelledError
from pathlib import Path
from threading import Event
from typing import List

from PyQt5.QtCore import QThread, pyqtSignal

from videocaptioner.core.asr.asr_data import ASRData
from videocaptioner.core.entities import (
    SubtitleConfig,
    SubtitleLayoutEnum,
    SubtitleProcessData,
    SubtitleTask,
    TranslatorServiceEnum,
)
from videocaptioner.core.llm.check_llm import check_llm_connection
from videocaptioner.core.llm.client import reset_llm_client
from videocaptioner.core.llm.context import (
    clear_task_context,
    generate_task_id,
    set_task_context,
    update_stage,
)
from videocaptioner.core.optimize.optimize import SubtitleOptimizer
from videocaptioner.core.split.split import SubtitleSplitter
from videocaptioner.core.subtitle.resegment import (
    config_from_object,
    resegment_translations,
)
from videocaptioner.core.translate.factory import TranslatorFactory
from videocaptioner.core.translate.ollama_runtime import (
    is_local_ollama_url,
    log_ollama_model_placement,
    prepare_local_ollama,
    release_ollama_model,
)
from videocaptioner.core.translate.types import TranslatorType, translation_needs_review
from videocaptioner.core.utils.logger import setup_logger

SERVICE_TO_TYPE = {
    TranslatorServiceEnum.OPENAI: TranslatorType.OPENAI,
    TranslatorServiceEnum.GOOGLE: TranslatorType.GOOGLE,
    TranslatorServiceEnum.BING: TranslatorType.BING,
    TranslatorServiceEnum.DEEPLX: TranslatorType.DEEPLX,
}

logger = setup_logger("subtitle_optimization_thread")


def create_translator_from_config(
    config: SubtitleConfig,
    custom_prompt: str = "",
    callback=None,
):
    """根据 SubtitleConfig 创建翻译器"""
    translator_service = config.translator_service
    if translator_service not in SERVICE_TO_TYPE:
        raise ValueError(f"不支持的翻译服务: {translator_service}")

    if translator_service == TranslatorServiceEnum.DEEPLX:
        os.environ["DEEPLX_ENDPOINT"] = config.deeplx_endpoint or ""

    return TranslatorFactory.create_translator(
        translator_type=SERVICE_TO_TYPE[translator_service],
        thread_num=config.thread_num,
        batch_num=config.batch_size,
        target_language=config.target_language,
        model=config.llm_model or "",
        custom_prompt=custom_prompt,
        is_reflect=config.need_reflect,
        update_callback=callback,
    )


class SubtitleThread(QThread):
    finished = pyqtSignal(str, str)
    progress = pyqtSignal(int, str)
    update = pyqtSignal(dict)
    update_all = pyqtSignal(dict)
    error = pyqtSignal(str)
    cancelled = pyqtSignal()
    partial = pyqtSignal(dict)

    def __init__(
        self, task: SubtitleTask, *, input_data: ASRData | None = None, allow_partial: bool = False
    ):
        super().__init__()
        self.task: SubtitleTask = task
        self.input_data = input_data
        self.allow_partial = allow_partial
        self.subtitle_length = 0
        self.finished_subtitle_length = 0
        self.custom_prompt_text = ""
        self._cancel_requested = Event()

    def set_custom_prompt_text(self, text: str):
        self.custom_prompt_text = text

    def _setup_llm_config(self) -> SubtitleConfig:
        """验证 LLM 配置并设置环境变量，返回 SubtitleConfig"""
        config = self.task.subtitle_config
        if not config:
            raise Exception(self.tr("LLM API 未配置, 请检查LLM配置"))
        if config.base_url and config.api_key and config.llm_model:
            if is_local_ollama_url(config.base_url):
                self.progress.emit(2, self.tr("正在啟動 Qwen3 離線翻譯..."))
                prepare_local_ollama(
                    config.llm_model,
                    parallelism=config.thread_num,
                    on_download=lambda: self.progress.emit(
                        2, self.tr("首次使用，正在下載 Qwen3 翻譯模型…")
                    ),
                    cancel_event=self._cancel_requested,
                )
            else:
                success, message = check_llm_connection(
                    config.base_url,
                    config.api_key,
                    config.llm_model,
                )
                if not success:
                    raise Exception(f"{self.tr('LLM API 测试失败: ')}{message or ''}")
            os.environ["OPENAI_BASE_URL"] = config.base_url
            os.environ["OPENAI_API_KEY"] = config.api_key
            reset_llm_client()
            return config
        else:
            raise Exception(self.tr("LLM API 未配置, 请检查LLM配置"))

    def run(self):
        if self._cancel_requested.is_set():
            self.cancelled.emit()
            return
        # 设置任务上下文
        task_file = (
            Path(self.task.video_path) if self.task.video_path else Path(self.task.subtitle_path)
        )
        set_task_context(
            task_id=self.task.task_id,
            file_name=task_file.name,
            stage="subtitle",
        )

        uses_local_ollama = False
        completed = False
        partial_data = None
        try:
            self._check_cancelled()
            logger.info(f"\n{self.task.subtitle_config.print_config()}")

            # 字幕文件路径检查、对断句字幕路径进行定义
            subtitle_path = self.task.subtitle_path
            assert subtitle_path is not None, self.tr("字幕文件路径为空")

            subtitle_config = self.task.subtitle_config
            assert subtitle_config is not None, self.tr("字幕配置为空")
            if (
                subtitle_config.subtitle_layout is SubtitleLayoutEnum.ONLY_TRANSLATE
                and not subtitle_config.need_translate
            ):
                raise ValueError(self.tr("「僅譯文」需要啟用字幕翻譯。"))

            asr_data = self.input_data
            if asr_data is None:
                asr_data = ASRData.from_subtitle_file(subtitle_path)

            # 1. 分割成字词级时间戳（对于非断句字幕且开启分割选项）
            if subtitle_config.need_split and not asr_data.is_word_timestamp():
                asr_data.split_to_word_segments()
                self.update_all.emit(asr_data.to_json())

            # 验证 LLM 配置
            if self.need_llm(subtitle_config, asr_data):
                self.progress.emit(2, self.tr("开始验证 LLM 配置..."))
                subtitle_config = self._setup_llm_config()
                uses_local_ollama = is_local_ollama_url(subtitle_config.base_url)
            self._check_cancelled()

            # 2. 重新断句（对于字词级字幕）
            if asr_data.is_word_timestamp():
                update_stage("split")
                self.progress.emit(5, self.tr("字幕断句..."))
                logger.info("正在字幕断句...")
                splitter = SubtitleSplitter(
                    thread_num=subtitle_config.thread_num,
                    model=subtitle_config.llm_model,
                    max_word_count_cjk=subtitle_config.max_word_count_cjk,
                    max_word_count_english=subtitle_config.max_word_count_english,
                )
                try:
                    asr_data = splitter.split_subtitle(asr_data)
                finally:
                    splitter.stop()
                self._check_cancelled()
                self.update_all.emit(asr_data.to_json())

            # 3. 优化字幕
            context_info = f'The subtitles below are from a file named "{task_file}". Use this context to improve accuracy if needed.\n'
            custom_prompt = context_info + (subtitle_config.custom_prompt_text or "") + "\n"
            self.subtitle_length = len(asr_data.segments)

            if subtitle_config.need_optimize:
                update_stage("optimize")
                self.progress.emit(0, self.tr("优化字幕..."))
                logger.info("正在优化字幕...")
                self.finished_subtitle_length = 0
                if not subtitle_config.llm_model:
                    raise Exception(self.tr("LLM 模型未配置"))
                optimizer = SubtitleOptimizer(
                    thread_num=subtitle_config.thread_num,
                    batch_num=subtitle_config.batch_size,
                    model=subtitle_config.llm_model,
                    custom_prompt=custom_prompt or "",
                    update_callback=self._update_progress,
                )
                try:
                    asr_data = optimizer.optimize_subtitle(asr_data)
                finally:
                    optimizer.stop()
                self._check_cancelled()
                asr_data.remove_punctuation()
                self.update_all.emit(asr_data.to_json())

            # 4. 翻译字幕
            if subtitle_config.need_translate:
                update_stage("translate")
                self.progress.emit(0, self.tr("翻译字幕..."))
                logger.info("正在翻译字幕...")
                self.finished_subtitle_length = 0

                if not subtitle_config.target_language:
                    raise Exception(self.tr("目标语言未配置"))

                translator = create_translator_from_config(
                    subtitle_config, custom_prompt, self.callback
                )

                try:
                    asr_data = translator.translate_subtitle(
                        asr_data, cancel_event=self._cancel_requested, allow_partial=self.allow_partial
                    )
                finally:
                    translator.stop(wait=True)
                self._check_cancelled()
                if self.allow_partial and any(
                    cue.text.strip() and translation_needs_review(cue.translated_text)
                    for cue in asr_data.segments
                ):
                    partial_data = asr_data.to_json()
                    return
                # Re-check translated cue duration, reading speed and line
                # wrapping. Overlong text is split at punctuation, not cut.
                asr_data = resegment_translations(
                    asr_data, config_from_object(subtitle_config)
                )
                self._check_cancelled()

                # 移除末尾标点符号
                asr_data.remove_punctuation()
                self.update_all.emit(asr_data.to_json())

                # 保存翻译结果(单语、双语)
                self._check_cancelled()
                if self.task.need_next_task and self.task.video_path:
                    for layout in SubtitleLayoutEnum:
                        save_path = str(
                            Path(self.task.subtitle_path).parent
                            / f"{Path(self.task.video_path).stem}-{layout.value}.srt"
                        )
                        asr_data.save(
                            save_path=save_path,
                            ass_style=subtitle_config.subtitle_style or "",
                            layout=layout,
                        )
                        logger.info(f"翻译字幕保存到：{save_path}")

            # 5. 保存字幕
            self._check_cancelled()
            asr_data.save(
                save_path=self.task.output_path or "",
                ass_style=subtitle_config.subtitle_style or "",
                layout=subtitle_config.subtitle_layout or SubtitleLayoutEnum.ONLY_ORIGINAL,
            )
            logger.info(f"字幕保存到 {self.task.output_path}")

            # 6. 文件移动与清理
            if self.task.need_next_task and self.task.video_path:
                # 保存srt/ass文件到视频目录（对于全流程任务）
                save_srt_path = (
                    Path(self.task.video_path).parent / f"{Path(self.task.video_path).stem}.srt"
                )
                asr_data.to_srt(
                    save_path=str(save_srt_path),
                    layout=subtitle_config.subtitle_layout,
                )
                save_ass_path = (
                    Path(self.task.video_path).parent / f"{Path(self.task.video_path).stem}.ass"
                )
                asr_data.to_ass(
                    save_path=str(save_ass_path),
                    layout=subtitle_config.subtitle_layout,
                    style_str=subtitle_config.subtitle_style,
                )

            completed = True

        except CancelledError:
            logger.info("字幕處理已取消")
        except Exception as e:
            if not self._cancel_requested.is_set():
                logger.exception(f"字幕处理失败: {str(e)}")
                self.error.emit(str(e))
                self.progress.emit(100, self.tr("字幕处理失败"))
        finally:
            config = self.task.subtitle_config
            if uses_local_ollama and config and config.llm_model:
                log_ollama_model_placement(config.llm_model)
                release_ollama_model(config.llm_model)
            clear_task_context()
            if self._cancel_requested.is_set():
                self.cancelled.emit()
            elif partial_data is not None:
                self.partial.emit(partial_data)
            elif completed:
                status = self.tr("翻譯完成") if config and config.need_translate else self.tr("字幕處理完成")
                self.progress.emit(100, status)
                logger.info(status)
                self.finished.emit(self.task.video_path, self.task.output_path)

    def need_llm(self, subtitle_config: SubtitleConfig, asr_data: ASRData):
        return (
            subtitle_config.need_optimize
            or asr_data.is_word_timestamp()
            or (
                subtitle_config.need_translate
                and subtitle_config.translator_service
                not in [
                    TranslatorServiceEnum.DEEPLX,
                    TranslatorServiceEnum.BING,
                    TranslatorServiceEnum.GOOGLE,
                ]
            )
        )

    def _update_progress(self, result: List[SubtitleProcessData]):
        if self._cancel_requested.is_set():
            return
        self.finished_subtitle_length += len(result)
        # 简单计算当前进度（0-100%）
        progress = min(int((self.finished_subtitle_length / max(self.subtitle_length, 1)) * 100), 100)
        self.progress.emit(progress, self.tr("{0}% 处理字幕").format(progress))

    def callback(self, result: List[SubtitleProcessData]):
        if self._cancel_requested.is_set():
            return
        self._update_progress(result)
        self.update.emit({str(data.index): data.translated_text for data in result})

    def _check_cancelled(self):
        if self._cancel_requested.is_set():
            raise CancelledError()

    def stop(self):
        """Request cancellation and let the worker release its own resources."""
        self._cancel_requested.set()


class RetranslateThread(QThread):
    """重新翻译选中行的轻量线程"""

    finished = pyqtSignal(dict)  # {key: translated_text}
    progress = pyqtSignal(int, str)  # (百分比, 状态描述)
    error = pyqtSignal(str)

    def __init__(
        self,
        selected_data: dict,
        subtitle_config: SubtitleConfig,
        file_name: str = "",
        *,
        context_by_row: dict[str, tuple[str, str]] | None = None,
    ):
        """
        selected_data: model._data 中选中的条目，键为行号字符串
        subtitle_config: 当前任务配置
        file_name: 用于日志上下文的文件名
        """
        super().__init__()
        self.selected_data = selected_data
        self.subtitle_config = subtitle_config
        self.file_name = file_name
        self.context_by_row = context_by_row or {}
        self.total = len(selected_data)
        self.done = 0
        self._cancel_requested = Event()

    def stop(self):
        self._cancel_requested.set()

    def _callback(self, result: List[SubtitleProcessData]):
        if self._cancel_requested.is_set():
            return
        self.done += len(result)
        pct = min(int(self.done / self.total * 100), 100)
        self.progress.emit(pct, self.tr("{0}% 翻译中").format(pct))

    def run(self):
        set_task_context(
            task_id=generate_task_id(),
            file_name=self.file_name,
            stage="translate",
        )
        result = None
        uses_local_ollama = False
        try:
            if self._cancel_requested.is_set():
                return
            config = self.subtitle_config
            if not config.target_language:
                raise Exception("目标语言未配置")

            # 设置 LLM 环境变量（LLM 翻译需要）
            if config.translator_service == TranslatorServiceEnum.OPENAI:
                if not (config.base_url and config.api_key and config.llm_model):
                    raise Exception("LLM API 未配置，请检查 LLM 配置")
                if is_local_ollama_url(config.base_url):
                    prepare_local_ollama(
                        config.llm_model,
                        parallelism=config.thread_num,
                        on_download=lambda: self.progress.emit(
                            2, self.tr("首次使用，正在下載 Qwen3 翻譯模型…")
                        ),
                        cancel_event=self._cancel_requested,
                    )
                    uses_local_ollama = True
                os.environ["OPENAI_BASE_URL"] = config.base_url
                os.environ["OPENAI_API_KEY"] = config.api_key
                reset_llm_client()

            # 构建仅含选中行的 ASRData
            asr_data = ASRData.from_json(self.selected_data)

            # 创建翻译器并翻译
            keys = sorted(self.selected_data, key=int)
            context_by_index = {
                index: self.context_by_row[key]
                for index, key in enumerate(keys, 1)
                if key in self.context_by_row
            }
            translator = create_translator_from_config(
                config, config.custom_prompt_text or "", callback=self._callback
            )
            try:
                asr_data = translator.translate_subtitle(
                    asr_data,
                    context_by_index=context_by_index,
                    cancel_event=self._cancel_requested,
                    allow_partial=True,
                    force_retry=True,
                )
            finally:
                translator.stop(wait=True)

            # 构建 {原始行号: translated_text} 映射
            result = {
                keys[i]: seg.translated_text
                for i, seg in enumerate(asr_data.segments)
                if seg.translated_text.strip()
                and (
                    not translation_needs_review(seg.translated_text)
                    or not self.selected_data[keys[i]]["translated_subtitle"].strip()
                )
            }
        except CancelledError:
            pass
        except Exception as e:
            if not self._cancel_requested.is_set():
                logger.exception(f"重新翻译失败: {e}")
                self.error.emit(str(e))
        finally:
            config = self.subtitle_config
            if uses_local_ollama and config.llm_model:
                release_ollama_model(config.llm_model)
            clear_task_context()
            if result is not None and not self._cancel_requested.is_set():
                self.finished.emit(result)
