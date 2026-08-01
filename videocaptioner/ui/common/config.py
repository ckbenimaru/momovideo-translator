# coding:utf-8
from enum import Enum

from PyQt5.QtCore import QLocale
from PyQt5.QtGui import QColor
from qfluentwidgets import (
    BoolValidator,
    ConfigItem,
    ConfigSerializer,
    EnumSerializer,
    FolderValidator,
    OptionsConfigItem,
    OptionsValidator,
    QConfig,
    RangeConfigItem,
    RangeValidator,
    Theme,
    qconfig,
)

from videocaptioner.config import SETTINGS_PATH, WORK_PATH
from videocaptioner.core.entities import (
    FasterWhisperModelEnum,
    Qwen3ASRModelEnum,
    LLMServiceEnum,
    SubtitleLayoutEnum,
    SubtitleRenderModeEnum,
    TranscribeLanguageEnum,
    TranscribeModelEnum,
    TranscribeOutputFormatEnum,
    TranslatorServiceEnum,
    VadMethodEnum,
    VideoQualityEnum,
    WhisperModelEnum,
)
from videocaptioner.core.translate.types import TargetLanguage
from videocaptioner.core.utils.platform_utils import get_available_transcribe_models


class Language(Enum):
    """软件语言"""

    CHINESE_SIMPLIFIED = QLocale(QLocale.Chinese, QLocale.China)
    CHINESE_TRADITIONAL = QLocale(QLocale.Chinese, QLocale.HongKong)
    ENGLISH = QLocale(QLocale.English)
    AUTO = QLocale()


class LanguageSerializer(ConfigSerializer):
    """Language serializer"""

    def serialize(self, language):
        return language.value.name() if language != Language.AUTO else "Auto"

    def deserialize(self, value: str):
        return Language(QLocale(value)) if value != "Auto" else Language.AUTO


class PlatformAwareTranscribeModelValidator(OptionsValidator):
    """平台相关的转录模型验证器，在 macOS 上自动过滤掉 FasterWhisper"""

    def __init__(self):
        # 不调用父类的 __init__，因为我们要自定义 options
        self._options = get_available_transcribe_models()

    @property
    def options(self):
        return self._options

    def validate(self, value):
        return value in self._options

    def correct(self, value):
        return value if self.validate(value) else self._options[0]


class Config(QConfig):
    """应用配置"""

    # LLM配置
    llm_service = OptionsConfigItem(
        "LLM",
        "LLMService",
        LLMServiceEnum.OLLAMA,
        OptionsValidator(LLMServiceEnum),
        EnumSerializer(LLMServiceEnum),
    )

    openai_model = ConfigItem("LLM", "OpenAI_Model", "gpt-4o-mini")
    openai_api_key = ConfigItem("LLM", "OpenAI_API_Key", "")
    openai_api_base = ConfigItem("LLM", "OpenAI_API_Base", "https://api.openai.com/v1")

    silicon_cloud_model = ConfigItem("LLM", "SiliconCloud_Model", "gpt-4o-mini")
    silicon_cloud_api_key = ConfigItem("LLM", "SiliconCloud_API_Key", "")
    silicon_cloud_api_base = ConfigItem(
        "LLM", "SiliconCloud_API_Base", "https://api.siliconflow.cn/v1"
    )

    deepseek_model = ConfigItem("LLM", "DeepSeek_Model", "deepseek-chat")
    deepseek_api_key = ConfigItem("LLM", "DeepSeek_API_Key", "")
    deepseek_api_base = ConfigItem(
        "LLM", "DeepSeek_API_Base", "https://api.deepseek.com/v1"
    )

    ollama_model = ConfigItem(
        "LLM", "Ollama_Model", "qwen3:4b-instruct-2507-q4_K_M"
    )
    ollama_api_key = ConfigItem("LLM", "Ollama_API_Key", "ollama")
    ollama_api_base = ConfigItem(
        "LLM", "Ollama_API_Base", "http://127.0.0.1:11434/v1"
    )

    lm_studio_model = ConfigItem("LLM", "LmStudio_Model", "qwen2.5:7b")
    lm_studio_api_key = ConfigItem("LLM", "LmStudio_API_Key", "lmstudio")
    lm_studio_api_base = ConfigItem(
        "LLM", "LmStudio_API_Base", "http://localhost:1234/v1"
    )

    gemini_model = ConfigItem("LLM", "Gemini_Model", "gemini-pro")
    gemini_api_key = ConfigItem("LLM", "Gemini_API_Key", "")
    gemini_api_base = ConfigItem(
        "LLM",
        "Gemini_API_Base",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    chatglm_model = ConfigItem("LLM", "ChatGLM_Model", "glm-4")
    chatglm_api_key = ConfigItem("LLM", "ChatGLM_API_Key", "")
    chatglm_api_base = ConfigItem(
        "LLM", "ChatGLM_API_Base", "https://open.bigmodel.cn/api/paas/v4"
    )

    # ------------------- 翻译配置 -------------------
    translator_service = OptionsConfigItem(
        "Translate",
        "TranslatorServiceEnum",
        TranslatorServiceEnum.OPENAI,
        OptionsValidator(TranslatorServiceEnum),
        EnumSerializer(TranslatorServiceEnum),
    )
    need_reflect_translate = ConfigItem(
        "Translate", "NeedReflectTranslate", False, BoolValidator()
    )
    deeplx_endpoint = ConfigItem("Translate", "DeeplxEndpoint", "")
    batch_size = RangeConfigItem("Translate", "BatchSize", 10, RangeValidator(5, 50))
    thread_num = RangeConfigItem("Translate", "ThreadNum", 10, RangeValidator(1, 50))

    # ------------------- 转录配置 -------------------
    transcribe_model = OptionsConfigItem(
        "Transcribe",
        "TranscribeModel",
        TranscribeModelEnum.QWEN3_ASR,
        PlatformAwareTranscribeModelValidator(),
        EnumSerializer(TranscribeModelEnum),
    )
    transcribe_output_format = OptionsConfigItem(
        "Transcribe",
        "OutputFormat",
        TranscribeOutputFormatEnum.SRT,
        OptionsValidator(TranscribeOutputFormatEnum),
        EnumSerializer(TranscribeOutputFormatEnum),
    )
    transcribe_language = OptionsConfigItem(
        "Transcribe",
        "TranscribeLanguage",
        TranscribeLanguageEnum.AUTO,
        OptionsValidator(TranscribeLanguageEnum),
        EnumSerializer(TranscribeLanguageEnum),
    )

    # ------------------- Whisper Cpp 配置 -------------------
    whisper_model = OptionsConfigItem(
        "Whisper",
        "WhisperModel",
        WhisperModelEnum.TINY,
        OptionsValidator(WhisperModelEnum),
        EnumSerializer(WhisperModelEnum),
    )

    # ------------------- Faster Whisper 配置 -------------------
    faster_whisper_program = ConfigItem(
        "FasterWhisper",
        "Program",
        "faster-whisper-xxl.exe",
    )
    faster_whisper_model = OptionsConfigItem(
        "FasterWhisper",
        "Model",
        FasterWhisperModelEnum.LARGE_V3,
        OptionsValidator(FasterWhisperModelEnum),
        EnumSerializer(FasterWhisperModelEnum),
    )
    faster_whisper_model_dir = ConfigItem("FasterWhisper", "ModelDir", "")
    faster_whisper_device = OptionsConfigItem(
        "FasterWhisper", "Device", "cuda", OptionsValidator(["cuda", "cpu"])
    )
    # VAD 参数
    faster_whisper_vad_filter = ConfigItem(
        "FasterWhisper", "VadFilter", True, BoolValidator()
    )
    faster_whisper_vad_threshold = RangeConfigItem(
        "FasterWhisper", "VadThreshold", 0.4, RangeValidator(0, 1)
    )
    faster_whisper_vad_method = OptionsConfigItem(
        "FasterWhisper",
        "VadMethod",
        VadMethodEnum.SILERO_V4,
        OptionsValidator(VadMethodEnum),
        EnumSerializer(VadMethodEnum),
    )
    faster_whisper_vad_min_silence_duration_ms = ConfigItem(
        "FasterWhisper", "VadMinSilenceDurationMs", 350, RangeValidator(100, 2000)
    )
    faster_whisper_vad_speech_pad_ms = ConfigItem(
        "FasterWhisper", "VadSpeechPadMs", 200, RangeValidator(0, 1000)
    )
    # Word-timestamp subtitle re-segmentation. A zero character/rate limit
    # selects language-aware defaults (CJK 20/36/12, English 42/76/20).
    subtitle_ideal_duration = RangeConfigItem(
        "FasterWhisper", "IdealDuration", 3.5, RangeValidator(2.0, 5.0)
    )
    subtitle_max_duration = RangeConfigItem(
        "FasterWhisper", "MaxDuration", 6.0, RangeValidator(2.0, 10.0)
    )
    subtitle_min_duration = RangeConfigItem(
        "FasterWhisper", "MinDuration", 0.8, RangeValidator(0.2, 2.0)
    )
    subtitle_pause_threshold = RangeConfigItem(
        "FasterWhisper", "PauseThreshold", 0.45, RangeValidator(0.1, 2.0)
    )
    subtitle_max_chars_per_line = ConfigItem(
        "FasterWhisper", "MaxCharsPerLine", 0, RangeValidator(0, 120)
    )
    subtitle_max_lines = ConfigItem(
        "FasterWhisper", "MaxLines", 2, RangeValidator(1, 4)
    )
    subtitle_max_chars_per_subtitle = ConfigItem(
        "FasterWhisper", "MaxCharsPerSubtitle", 0, RangeValidator(0, 240)
    )
    subtitle_max_chars_per_second = RangeConfigItem(
        "FasterWhisper", "MaxCharsPerSecond", 0.0, RangeValidator(0.0, 60.0)
    )
    # 人声提取
    faster_whisper_ff_mdx_kim2 = ConfigItem(
        "FasterWhisper", "FfMdxKim2", False, BoolValidator()
    )
    # 文本处理参数
    faster_whisper_one_word = ConfigItem(
        "FasterWhisper", "OneWord", True, BoolValidator()
    )
    # 提示词
    faster_whisper_prompt = ConfigItem("FasterWhisper", "Prompt", "")

    # ------------------- Qwen3-ASR 配置 -------------------
    qwen_runtime_dir = ConfigItem("Qwen3ASR", "RuntimeDir", "")
    qwen_model_dir = ConfigItem("Qwen3ASR", "ModelDir", "")
    qwen_asr_model = OptionsConfigItem(
        "Qwen3ASR",
        "Model",
        Qwen3ASRModelEnum.ASR_1_7B,
        OptionsValidator(Qwen3ASRModelEnum),
        EnumSerializer(Qwen3ASRModelEnum),
    )
    qwen_use_forced_aligner = ConfigItem(
        "Qwen3ASR", "UseForcedAligner", True, BoolValidator()
    )
    qwen_device = OptionsConfigItem(
        "Qwen3ASR", "Device", "cuda", OptionsValidator(["cuda", "cpu"])
    )
    qwen_chunk_seconds = ConfigItem(
        "Qwen3ASR", "ChunkSeconds", 30, RangeValidator(30, 240)
    )

    # ------------------- Whisper API 配置 -------------------
    whisper_api_base = ConfigItem("WhisperAPI", "WhisperApiBase", "")
    whisper_api_key = ConfigItem("WhisperAPI", "WhisperApiKey", "")
    whisper_api_model = OptionsConfigItem("WhisperAPI", "WhisperApiModel", "")
    whisper_api_prompt = ConfigItem("WhisperAPI", "WhisperApiPrompt", "")

    # ------------------- 字幕配置 -------------------
    need_optimize = ConfigItem("Subtitle", "NeedOptimize", False, BoolValidator())
    need_translate = ConfigItem("Subtitle", "NeedTranslate", False, BoolValidator())
    need_split = ConfigItem("Subtitle", "NeedSplit", False, BoolValidator())
    target_language = OptionsConfigItem(
        "Subtitle",
        "TargetLanguage",
        TargetLanguage.SIMPLIFIED_CHINESE,
        OptionsValidator(TargetLanguage),
        EnumSerializer(TargetLanguage),
    )
    max_word_count_cjk = ConfigItem(
        "Subtitle", "MaxWordCountCJK", 28, RangeValidator(8, 100)
    )
    max_word_count_english = ConfigItem(
        "Subtitle", "MaxWordCountEnglish", 20, RangeValidator(8, 100)
    )
    custom_prompt_text = ConfigItem("Subtitle", "CustomPromptText", "")

    # ------------------- 字幕合成配置 -------------------
    soft_subtitle = ConfigItem("Video", "SoftSubtitle", False, BoolValidator())
    need_video = ConfigItem("Video", "NeedVideo", True, BoolValidator())
    video_quality = OptionsConfigItem(
        "Video",
        "VideoQuality",
        VideoQualityEnum.MEDIUM,
        OptionsValidator(VideoQualityEnum),
        EnumSerializer(VideoQualityEnum),
    )
    use_subtitle_style = ConfigItem("Video", "UseSubtitleStyle", True, BoolValidator())

    # ------------------- 字幕样式配置 -------------------
    subtitle_style_name = ConfigItem(
        "SubtitleStyle", "StyleName", "youtube-modern"
    )
    subtitle_layout = OptionsConfigItem(
        "SubtitleStyle",
        "Layout",
        SubtitleLayoutEnum.TRANSLATE_ON_TOP,
        OptionsValidator(SubtitleLayoutEnum),
        EnumSerializer(SubtitleLayoutEnum),
    )
    subtitle_preview_image = ConfigItem("SubtitleStyle", "PreviewImage", "")

    # 字幕渲染模式
    subtitle_render_mode = OptionsConfigItem(
        "SubtitleStyle",
        "RenderMode",
        SubtitleRenderModeEnum.ROUNDED_BG,
        OptionsValidator(SubtitleRenderModeEnum),
        EnumSerializer(SubtitleRenderModeEnum),
    )

    # 圆角背景模式配置
    rounded_bg_font_name = ConfigItem("RoundedBgStyle", "FontName", "Noto Sans TC")
    rounded_bg_font_size = RangeConfigItem(
        "RoundedBgStyle", "FontSize", 42, RangeValidator(16, 120)
    )
    # YouTube Modern：黑色圆角背景，35% 不透明度 (alpha=0x59)
    rounded_bg_color = ConfigItem("RoundedBgStyle", "BgColor", "#00000059")
    rounded_bg_text_color = ConfigItem("RoundedBgStyle", "TextColor", "#FFFFFF")
    rounded_bg_corner_radius = RangeConfigItem(
        "RoundedBgStyle", "CornerRadius", 14, RangeValidator(0, 50)
    )
    rounded_bg_padding_h = RangeConfigItem(
        "RoundedBgStyle", "PaddingH", 24, RangeValidator(4, 100)
    )
    rounded_bg_padding_v = RangeConfigItem(
        "RoundedBgStyle", "PaddingV", 12, RangeValidator(4, 50)
    )
    rounded_bg_margin_bottom = RangeConfigItem(
        "RoundedBgStyle", "MarginBottom", 58, RangeValidator(20, 300)
    )
    rounded_bg_line_spacing = RangeConfigItem(
        "RoundedBgStyle", "LineSpacing", 12, RangeValidator(0, 50)
    )
    rounded_bg_letter_spacing = RangeConfigItem(
        "RoundedBgStyle", "LetterSpacing", 0, RangeValidator(0, 20)
    )

    # ------------------- 保存配置 -------------------
    work_dir = ConfigItem("Save", "Work_Dir", WORK_PATH, FolderValidator())

    # ------------------- 软件页面配置 -------------------
    micaEnabled = ConfigItem("MainWindow", "MicaEnabled", False, BoolValidator())
    dpiScale = OptionsConfigItem(
        "MainWindow",
        "DpiScale",
        "Auto",
        OptionsValidator([1, 1.25, 1.5, 1.75, 2, "Auto"]),
        restart=True,
    )
    language = OptionsConfigItem(
        "MainWindow",
        "Language",
        Language.AUTO,
        OptionsValidator(Language),
        LanguageSerializer(),
        restart=True,
    )

    # ------------------- 更新配置 -------------------
    checkUpdateAtStartUp = ConfigItem(
        "Update", "CheckUpdateAtStartUp", True, BoolValidator()
    )

    # ------------------- 缓存配置 -------------------
    cache_enabled = ConfigItem("Cache", "CacheEnabled", True, BoolValidator())


cfg = Config()
cfg.themeMode.value = Theme.DARK
cfg.themeColor.value = QColor("#ff28f08b")
qconfig.load(SETTINGS_PATH, cfg)
