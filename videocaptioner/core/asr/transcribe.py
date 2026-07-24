from dataclasses import replace

import requests

from videocaptioner.core.asr.asr_data import ASRData
from videocaptioner.core.asr.bcut import BcutASR
from videocaptioner.core.asr.chunked_asr import ChunkedASR
from videocaptioner.core.asr.faster_whisper import FasterWhisperASR
from videocaptioner.core.asr.jianying import JianYingASR
from videocaptioner.core.asr.qwen3_asr import Qwen3ASR
from videocaptioner.core.asr.whisper_api import WhisperAPI
from videocaptioner.core.asr.whisper_cpp import WhisperCppASR
from videocaptioner.core.entities import TranscribeConfig, TranscribeModelEnum


def transcribe(audio_path: str, config: TranscribeConfig, callback=None) -> ASRData:
    """Transcribe audio file using specified configuration.

    Args:
        audio_path: Path to audio file
        config: Transcription configuration
        callback: Progress callback function(progress: int, message: str)

    Returns:
        ASRData: Transcription result data
    """

    def _default_callback(x, y):
        pass

    if callback is None:
        callback = _default_callback

    if config.transcribe_model is None:
        raise ValueError("Transcription model not set")

    # Create ASR instance based on model type
    asr = _create_asr_instance(audio_path, config)

    # Run transcription. Bcut may reject an otherwise valid task result with
    # HTTP 412 (commonly policy/precondition filtering). Retry the full audio
    # once with the independent JianYing/CapCut ASR backend.
    try:
        asr_data = asr.run(callback=callback)
    except requests.exceptions.HTTPError as exc:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        if (
            config.transcribe_model == TranscribeModelEnum.BIJIAN
            and status_code == 412
        ):
            callback(0, "B 接口返回 412，正在自动切换 J 接口重新转录")
            fallback_config = replace(
                config, transcribe_model=TranscribeModelEnum.JIANYING
            )
            fallback_asr = _create_asr_instance(audio_path, fallback_config)
            asr_data = fallback_asr.run(callback=callback)
        else:
            raise

    if config.transcribe_model in {
        TranscribeModelEnum.FASTER_WHISPER,
        TranscribeModelEnum.QWEN3_ASR,
    }:
        from videocaptioner.core.subtitle.resegment import (
            config_from_object,
            resegment_asr_data,
        )

        asr_data = resegment_asr_data(asr_data, config_from_object(config))

    # Optimize legacy sentence-level timing. FasterWhisper timing has already
    # been rebuilt from exact word timestamps and must not be stretched here.
    if (
        not config.need_word_time_stamp
        and config.transcribe_model
        not in {TranscribeModelEnum.FASTER_WHISPER, TranscribeModelEnum.QWEN3_ASR}
    ):
        asr_data.optimize_timing()

    return asr_data


def _create_asr_instance(audio_path: str, config: TranscribeConfig) -> ChunkedASR:
    """Create appropriate ASR instance based on configuration.

    Args:
        audio_path: Path to audio file
        config: Transcription configuration

    Returns:
        ChunkedASR: Chunked ASR instance ready to run
    """
    model_type = config.transcribe_model

    if model_type == TranscribeModelEnum.JIANYING:
        return _create_jianying_asr(audio_path, config)

    elif model_type == TranscribeModelEnum.BIJIAN:
        return _create_bijian_asr(audio_path, config)

    elif model_type == TranscribeModelEnum.WHISPER_CPP:
        return _create_whisper_cpp_asr(audio_path, config)

    elif model_type == TranscribeModelEnum.WHISPER_API:
        return _create_whisper_api_asr(audio_path, config)

    elif model_type == TranscribeModelEnum.FASTER_WHISPER:
        return _create_faster_whisper_asr(audio_path, config)

    elif model_type == TranscribeModelEnum.QWEN3_ASR:
        return _create_qwen3_asr(audio_path, config)

    else:
        raise ValueError(f"Invalid transcription model: {model_type}")


def _create_jianying_asr(audio_path: str, config: TranscribeConfig) -> ChunkedASR:
    """Create JianYing ASR instance with chunking support."""
    asr_kwargs = {
        "use_cache": True,
        "need_word_time_stamp": config.need_word_time_stamp,
    }
    return ChunkedASR(
        asr_class=JianYingASR, audio_path=audio_path, asr_kwargs=asr_kwargs
    )


def _create_bijian_asr(audio_path: str, config: TranscribeConfig) -> ChunkedASR:
    """Create Bijian ASR instance with chunking support."""
    asr_kwargs = {
        "use_cache": True,
        "need_word_time_stamp": config.need_word_time_stamp,
    }
    return ChunkedASR(asr_class=BcutASR, audio_path=audio_path, asr_kwargs=asr_kwargs)


def _create_whisper_cpp_asr(audio_path: str, config: TranscribeConfig) -> ChunkedASR:
    """Create WhisperCpp ASR instance with chunking support."""
    asr_kwargs = {
        "use_cache": True,
        "need_word_time_stamp": config.need_word_time_stamp,
        "language": config.transcribe_language,
        "whisper_model": config.whisper_model.value if config.whisper_model else None,
    }
    return ChunkedASR(
        asr_class=WhisperCppASR,
        audio_path=audio_path,
        asr_kwargs=asr_kwargs,
        chunk_concurrency=1,  # 本地转录使用单线程
        chunk_length=60 * 20,  # 每块20分钟
    )


def _create_whisper_api_asr(audio_path: str, config: TranscribeConfig) -> ChunkedASR:
    """Create Whisper API ASR instance with chunking support."""
    asr_kwargs = {
        "use_cache": True,
        "need_word_time_stamp": config.need_word_time_stamp,
        "language": config.transcribe_language,
        "whisper_model": config.whisper_api_model or "whisper-1",
        "api_key": config.whisper_api_key or "",
        "base_url": config.whisper_api_base or "",
        "prompt": config.whisper_api_prompt or "",
    }
    return ChunkedASR(
        asr_class=WhisperAPI, audio_path=audio_path, asr_kwargs=asr_kwargs
    )


def _create_faster_whisper_asr(audio_path: str, config: TranscribeConfig) -> ChunkedASR:
    """Create FasterWhisper ASR instance with chunking support."""
    asr_kwargs = {
        "use_cache": True,
        "need_word_time_stamp": config.need_word_time_stamp,
        "faster_whisper_program": config.faster_whisper_program or "",
        "language": config.transcribe_language,
        "whisper_model": (
            config.faster_whisper_model.value if config.faster_whisper_model else "base"
        ),
        "model_dir": config.faster_whisper_model_dir or "",
        "device": config.faster_whisper_device,
        "vad_filter": config.faster_whisper_vad_filter,
        "vad_threshold": config.faster_whisper_vad_threshold,
        "vad_method": (
            config.faster_whisper_vad_method.value
            if config.faster_whisper_vad_method
            else ""
        ),
        "vad_min_silence_duration_ms": config.faster_whisper_vad_min_silence_duration_ms,
        "vad_speech_pad_ms": config.faster_whisper_vad_speech_pad_ms,
        "ff_mdx_kim2": config.faster_whisper_ff_mdx_kim2,
        "one_word": config.faster_whisper_one_word,
        "prompt": config.faster_whisper_prompt,
    }
    return ChunkedASR(
        asr_class=FasterWhisperASR,
        audio_path=audio_path,
        asr_kwargs=asr_kwargs,
        chunk_concurrency=1,  # 本地转录使用单线程
        chunk_length=60 * 20,  # 每块20分钟
    )


def _create_qwen3_asr(audio_path: str, config: TranscribeConfig) -> Qwen3ASR:
    """Create the isolated Qwen3-ASR pipeline.

    Qwen performs its own silence-aware chunking because its forced aligner is
    limited to five-minute inputs and the 8 GB target GPU must load ASR and
    aligner sequentially.
    """
    return Qwen3ASR(
        audio_input=audio_path,
        runtime_dir=config.qwen_runtime_dir or "",
        model_dir=config.qwen_model_dir or "",
        asr_model=(
            config.qwen_asr_model.value
            if config.qwen_asr_model
            else "Qwen3-ASR-1.7B-hf"
        ),
        use_forced_aligner=config.qwen_use_forced_aligner,
        language=config.transcribe_language,
        device=config.qwen_device,
        chunk_seconds=config.qwen_chunk_seconds,
        use_cache=True,
    )


if __name__ == "__main__":
    # 示例用法
    from videocaptioner.core.entities import WhisperModelEnum

    # 创建配置
    config = TranscribeConfig(
        transcribe_model=TranscribeModelEnum.WHISPER_CPP,
        transcribe_language="zh",
        whisper_model=WhisperModelEnum.MEDIUM,
    )

    # 转录音频
    audio_file = "test.wav"

    def progress_callback(progress: int, message: str):
        print(f"Progress: {progress}%, Message: {message}")

    result = transcribe(audio_file, config, callback=progress_callback)
    print(result)
