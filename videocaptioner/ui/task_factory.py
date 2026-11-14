import datetime
from pathlib import Path
from typing import Optional

from videocaptioner.config import MODEL_PATH
from videocaptioner.core.entities import (
    LANGUAGES,
    FullProcessTask,
    LLMServiceEnum,
    SubtitleConfig,
    SubtitleLayoutEnum,
    SubtitleTask,
    SupportedVideoFormats,
    SynthesisConfig,
    SynthesisTask,
    TranscribeConfig,
    TranscribeTask,
    TranscriptAndSubtitleTask,
    TranslatorServiceEnum,
)
from videocaptioner.core.translate.ollama_runtime import (
    apple_silicon_ollama_parallelism,
    is_local_ollama_url,
)
from videocaptioner.core.translate.types import TargetLanguage
from videocaptioner.core.utils.path_utils import sanitize_filename_component
from videocaptioner.ui.common.config import cfg


def _default_synthesis_path(video_path: str) -> str:
    source = Path(video_path)
    stem = sanitize_filename_component(source.stem)
    suffix = source.suffix.lower()
    if suffix != ".mp4" and source.parent.is_dir():
        video_suffixes = {f".{fmt.value}" for fmt in SupportedVideoFormats}
        if any(
            sibling.suffix.lower() in video_suffixes
            and sibling.suffix.lower() != suffix
            and sibling.stem.casefold() == source.stem.casefold()
            and sibling.is_file()
            for sibling in source.parent.iterdir()
        ):
            stem = sanitize_filename_component(f"{source.stem}{suffix}")
    return str(source.parent / f"{stem}.done.mp4")


class TaskFactory:
    """任务工厂类，用于创建各种类型的任务"""

    @staticmethod
    def get_ass_style(
        style_name: str,
        target_language: Optional[TargetLanguage] = None,
    ) -> str:
        """获取 ASS 字幕样式内容 (via style_manager, JSON-first with .txt fallback)"""
        from videocaptioner.core.subtitle.style_manager import SubtitleStyle, load_style

        is_builtin = style_name in {"default", "youtube-modern"}
        style = SubtitleStyle(name="youtube-modern") if is_builtin else load_style(style_name)
        if style is not None:
            if is_builtin:
                font_by_language = {
                    TargetLanguage.TRADITIONAL_CHINESE: "Noto Sans TC",
                    TargetLanguage.CANTONESE: "Noto Sans TC",
                    TargetLanguage.SIMPLIFIED_CHINESE: "Noto Sans SC",
                    TargetLanguage.ENGLISH: "Noto Sans",
                    TargetLanguage.ENGLISH_US: "Noto Sans",
                    TargetLanguage.ENGLISH_UK: "Noto Sans",
                }
                style.font_name = font_by_language.get(
                    target_language,
                    "Noto Sans TC" if target_language is None else "Noto Sans",
                )
                # to_ass_string() inherits the main font for Secondary when no
                # explicit secondary style exists, keeping bilingual cues safe.
                style.secondary = None
                style.bold = False
            # Older portable builds could persist an 8 px, zero-outline preset.
            # It is technically rendered by FFmpeg but is effectively invisible
            # on 720p/1080p video. Repair only clearly unusable presets so normal
            # user-defined styles remain untouched.
            if style.font_size < 18:
                style.font_size = 42
                style.outline_width = max(style.outline_width, 2.0)
                style.margin_bottom = max(style.margin_bottom, 30)
                if style.secondary is not None:
                    style.secondary.font_size = 30
                    style.secondary.outline_width = max(
                        style.secondary.outline_width, 2.0
                    )
            ass_style = style.to_ass_string()
            if is_builtin:
                ass_style = f"; MomoAutoFontByText: yes\n{ass_style}"
            return ass_style
        return ""

    @staticmethod
    def get_rounded_style() -> dict:
        """Use the active settings for both preview and video synthesis."""
        return {
            "font_name": cfg.rounded_bg_font_name.value,
            "font_size": cfg.rounded_bg_font_size.value,
            "bg_color": cfg.rounded_bg_color.value,
            "text_color": cfg.rounded_bg_text_color.value,
            "corner_radius": cfg.rounded_bg_corner_radius.value,
            "padding_h": cfg.rounded_bg_padding_h.value,
            "padding_v": cfg.rounded_bg_padding_v.value,
            "margin_bottom": cfg.rounded_bg_margin_bottom.value,
            "line_spacing": cfg.rounded_bg_line_spacing.value,
            "letter_spacing": cfg.rounded_bg_letter_spacing.value,
        }

    @staticmethod
    def create_transcribe_task(
        file_path: str,
        need_next_task: bool = False,
        task_id: Optional[str] = None,
    ) -> TranscribeTask:
        """创建转录任务"""
        # 获取文件名
        file_name = sanitize_filename_component(Path(file_path).stem)

        # 构建输出路径
        if need_next_task:
            need_word_time_stamp = cfg.need_split.value
            output_path = str(
                Path(cfg.work_dir.value)
                / file_name
                / "subtitle"
                / f"【原始字幕】{file_name}-{cfg.transcribe_model.value.value}-{cfg.transcribe_language.value.value}.srt"
            )
        else:
            need_word_time_stamp = False
            output_path = str(Path(file_path).parent / f"{file_name}.srt")

        from videocaptioner.core.asr.qwen3_asr import resolve_qwen_paths

        qwen_runtime_dir, qwen_model_dir = resolve_qwen_paths(
            cfg.qwen_runtime_dir.value,
            cfg.qwen_model_dir.value,
            cfg.faster_whisper_model_dir.value,
        )

        config = TranscribeConfig(
            transcribe_model=cfg.transcribe_model.value,
            transcribe_language=LANGUAGES[cfg.transcribe_language.value.value],
            need_word_time_stamp=need_word_time_stamp,
            output_format=cfg.transcribe_output_format.value,
            # Whisper Cpp 配置
            whisper_model=cfg.whisper_model.value,
            # Whisper API 配置
            whisper_api_key=cfg.whisper_api_key.value,
            whisper_api_base=cfg.whisper_api_base.value,
            whisper_api_model=cfg.whisper_api_model.value,
            whisper_api_prompt=cfg.whisper_api_prompt.value,
            # Faster Whisper 配置
            faster_whisper_program=cfg.faster_whisper_program.value,
            faster_whisper_model=cfg.faster_whisper_model.value,
            faster_whisper_model_dir=cfg.faster_whisper_model_dir.value or str(MODEL_PATH),
            faster_whisper_device=cfg.faster_whisper_device.value,
            faster_whisper_vad_filter=cfg.faster_whisper_vad_filter.value,
            faster_whisper_vad_threshold=cfg.faster_whisper_vad_threshold.value,
            faster_whisper_vad_method=cfg.faster_whisper_vad_method.value,
            faster_whisper_vad_min_silence_duration_ms=cfg.faster_whisper_vad_min_silence_duration_ms.value,
            faster_whisper_vad_speech_pad_ms=cfg.faster_whisper_vad_speech_pad_ms.value,
            faster_whisper_ff_mdx_kim2=cfg.faster_whisper_ff_mdx_kim2.value,
            faster_whisper_one_word=cfg.faster_whisper_one_word.value,
            faster_whisper_prompt=cfg.faster_whisper_prompt.value,
            # Qwen3-ASR 配置
            qwen_runtime_dir=str(qwen_runtime_dir),
            qwen_model_dir=str(qwen_model_dir),
            qwen_asr_model=cfg.qwen_asr_model.value,
            qwen_use_forced_aligner=cfg.qwen_use_forced_aligner.value,
            qwen_device=cfg.qwen_device.value,
            qwen_chunk_seconds=cfg.qwen_chunk_seconds.value,
            ideal_duration=cfg.subtitle_ideal_duration.value,
            max_duration=cfg.subtitle_max_duration.value,
            min_duration=cfg.subtitle_min_duration.value,
            pause_threshold=cfg.subtitle_pause_threshold.value,
            max_chars_per_line=cfg.subtitle_max_chars_per_line.value,
            max_lines=cfg.subtitle_max_lines.value,
            max_chars_per_subtitle=cfg.subtitle_max_chars_per_subtitle.value,
            max_chars_per_second=cfg.subtitle_max_chars_per_second.value,
        )

        task = TranscribeTask(
            queued_at=datetime.datetime.now(),
            file_path=file_path,
            output_path=output_path,
            transcribe_config=config,
            need_next_task=need_next_task,
        )
        if task_id:
            task.task_id = task_id
        return task

    @staticmethod
    def create_subtitle_task(
        file_path: str,
        video_path: Optional[str] = None,
        need_next_task: bool = False,
        task_id: Optional[str] = None,
    ) -> SubtitleTask:
        """创建字幕任务"""
        output_name = (
            Path(file_path).stem.replace("【原始字幕】", "").replace("【下载字幕】", "")
        )
        output_name = sanitize_filename_component(output_name)
        subtitle_layout = cfg.subtitle_layout.value
        need_translate = (
            cfg.need_translate.value or subtitle_layout is SubtitleLayoutEnum.ONLY_TRANSLATE
        )
        # 只在需要翻译时添加翻译服务后缀
        suffix = (
            f"-{cfg.translator_service.value.value}" if need_translate else ""
        )

        if need_next_task:
            output_path = str(
                Path(file_path).parent / f"【样式字幕】{output_name}{suffix}.ass"
            )
        else:
            output_path = str(
                Path(file_path).parent / f"【字幕】{output_name}{suffix}.srt"
            )

        # 根据当前选择的LLM服务获取对应的配置
        current_service = cfg.llm_service.value
        if current_service == LLMServiceEnum.OPENAI:
            base_url = cfg.openai_api_base.value
            api_key = cfg.openai_api_key.value
            llm_model = cfg.openai_model.value
        elif current_service == LLMServiceEnum.SILICON_CLOUD:
            base_url = cfg.silicon_cloud_api_base.value
            api_key = cfg.silicon_cloud_api_key.value
            llm_model = cfg.silicon_cloud_model.value
        elif current_service == LLMServiceEnum.DEEPSEEK:
            base_url = cfg.deepseek_api_base.value
            api_key = cfg.deepseek_api_key.value
            llm_model = cfg.deepseek_model.value
        elif current_service == LLMServiceEnum.OLLAMA:
            base_url = cfg.ollama_api_base.value
            api_key = cfg.ollama_api_key.value
            llm_model = cfg.ollama_model.value
        elif current_service == LLMServiceEnum.LM_STUDIO:
            base_url = cfg.lm_studio_api_base.value
            api_key = cfg.lm_studio_api_key.value
            llm_model = cfg.lm_studio_model.value
        elif current_service == LLMServiceEnum.GEMINI:
            base_url = cfg.gemini_api_base.value
            api_key = cfg.gemini_api_key.value
            llm_model = cfg.gemini_model.value
        elif current_service == LLMServiceEnum.CHATGLM:
            base_url = cfg.chatglm_api_base.value
            api_key = cfg.chatglm_api_key.value
            llm_model = cfg.chatglm_model.value
        else:
            base_url = ""
            api_key = ""
            llm_model = ""

        local_ollama = current_service == LLMServiceEnum.OLLAMA
        local_qwen_4b = (
            local_ollama
            and is_local_ollama_url(base_url)
            and llm_model.startswith(("qwen3:4b-instruct-", "qwen3-4b-instruct-"))
        )
        batch_size = min(cfg.batch_size.value, 8) if local_ollama else cfg.batch_size.value
        thread_num = 1 if local_ollama else cfg.thread_num.value
        if local_qwen_4b and cfg.translator_service.value == TranslatorServiceEnum.OPENAI:
            thread_num = min(cfg.thread_num.value, apple_silicon_ollama_parallelism())
        config = SubtitleConfig(
            # 翻译配置
            base_url=base_url,
            api_key=api_key,
            llm_model=llm_model,
            deeplx_endpoint=cfg.deeplx_endpoint.value,
            # 翻译服务
            translator_service=cfg.translator_service.value,
            # 字幕处理
            need_reflect=cfg.need_reflect_translate.value,
            need_translate=need_translate,
            need_optimize=cfg.need_optimize.value,
            # The local 4B model can shift translations to adjacent keys in
            # a batch even when it returns the right number of keys.
            thread_num=thread_num,
            batch_size=1 if local_qwen_4b else batch_size,
            # 字幕布局、样式
            subtitle_layout=subtitle_layout,
            subtitle_style=TaskFactory.get_ass_style(
                cfg.subtitle_style_name.value,
                cfg.target_language.value if need_translate else None,
            ),
            # 字幕分割
            max_word_count_cjk=cfg.max_word_count_cjk.value,
            max_word_count_english=cfg.max_word_count_english.value,
            need_split=cfg.need_split.value,
            ideal_duration=cfg.subtitle_ideal_duration.value,
            max_duration=cfg.subtitle_max_duration.value,
            min_duration=cfg.subtitle_min_duration.value,
            pause_threshold=cfg.subtitle_pause_threshold.value,
            max_chars_per_line=cfg.subtitle_max_chars_per_line.value,
            max_lines=cfg.subtitle_max_lines.value,
            max_chars_per_subtitle=cfg.subtitle_max_chars_per_subtitle.value,
            max_chars_per_second=cfg.subtitle_max_chars_per_second.value,
            # 字幕翻译
            target_language=cfg.target_language.value,
            # 字幕提示
            custom_prompt_text=cfg.custom_prompt_text.value,
        )

        task = SubtitleTask(
            queued_at=datetime.datetime.now(),
            subtitle_path=file_path,
            video_path=video_path,
            output_path=output_path,
            subtitle_config=config,
            need_next_task=need_next_task,
        )
        if task_id:
            task.task_id = task_id
        return task

    @staticmethod
    def create_synthesis_task(
        video_path: str,
        subtitle_path: str,
        need_next_task: bool = False,
        task_id: Optional[str] = None,
    ) -> SynthesisTask:
        """创建视频合成任务"""
        output_path = _default_synthesis_path(video_path)

        # 只有启用样式时才传入样式配置
        use_style = cfg.use_subtitle_style.value
        config = SynthesisConfig(
            need_video=cfg.need_video.value,
            soft_subtitle=cfg.soft_subtitle.value,
            render_mode=cfg.subtitle_render_mode.value,
            video_quality=cfg.video_quality.value,
            subtitle_layout=cfg.subtitle_layout.value,
            ass_style=(
                TaskFactory.get_ass_style(
                    cfg.subtitle_style_name.value,
                    cfg.target_language.value if cfg.need_translate.value else None,
                )
                if use_style
                else ""
            ),
            rounded_style=TaskFactory.get_rounded_style() if use_style else None,
        )

        task = SynthesisTask(
            queued_at=datetime.datetime.now(),
            video_path=video_path,
            subtitle_path=subtitle_path,
            output_path=output_path,
            synthesis_config=config,
            need_next_task=need_next_task,
        )
        if task_id:
            task.task_id = task_id
        return task

    @staticmethod
    def create_transcript_and_subtitle_task(
        file_path: str,
        output_path: Optional[str] = None,
        transcribe_config: Optional[TranscribeConfig] = None,
        subtitle_config: Optional[SubtitleConfig] = None,
    ) -> TranscriptAndSubtitleTask:
        """创建转录和字幕任务"""
        if output_path is None:
            file_stem = sanitize_filename_component(Path(file_path).stem)
            output_path = str(
                Path(file_path).parent / f"{file_stem}_processed.srt"
            )

        return TranscriptAndSubtitleTask(
            queued_at=datetime.datetime.now(),
            file_path=file_path,
            output_path=output_path,
        )

    @staticmethod
    def create_full_process_task(
        file_path: str,
        output_path: Optional[str] = None,
        transcribe_config: Optional[TranscribeConfig] = None,
        subtitle_config: Optional[SubtitleConfig] = None,
        synthesis_config: Optional[SynthesisConfig] = None,
    ) -> FullProcessTask:
        """创建完整处理任务（转录+字幕+合成）"""
        if output_path is None:
            output_path = _default_synthesis_path(file_path)

        return FullProcessTask(
            queued_at=datetime.datetime.now(),
            file_path=file_path,
            output_path=output_path,
        )
