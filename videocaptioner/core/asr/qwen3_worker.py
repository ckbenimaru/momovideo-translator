"""Isolated Qwen3-ASR inference worker.

This module intentionally imports PyTorch/Transformers only after startup so
the main momo environment never needs those heavyweight dependencies.
"""

import argparse
import gc
import json
import re
import tempfile
from pathlib import Path


LANGUAGE_NAMES = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "es": "Spanish",
}


def emit_progress(value: int, message: str) -> None:
    print(f"PROGRESS\t{value}\t{message}", flush=True)


def _hard_split(
    start: int,
    end: int,
    max_samples: int,
    overlap_samples: int = 0,
):
    """Split a continuous utterance, optionally retaining boundary context."""
    cursor = start
    while cursor < end:
        next_end = min(cursor + max_samples, end)
        yield cursor, next_end
        if next_end >= end:
            break
        cursor = max(cursor + 1, next_end - overlap_samples)


def group_speech_intervals(
    intervals,
    total_samples: int,
    sample_rate: int,
    max_seconds: int = 20,
    ideal_min_seconds: float = 5.0,
    ideal_max_seconds: float = 15.0,
    pause_seconds: float = 1.0,
    overlap_seconds: float = 0.25,
) -> list[tuple[int, int]]:
    """Build utterance chunks for per-chunk language identification."""
    if len(intervals) == 0:
        return []

    max_samples = max(1, int(max_seconds * sample_rate))
    ideal_min_samples = int(ideal_min_seconds * sample_rate)
    ideal_max_samples = int(ideal_max_seconds * sample_rate)
    pause_samples = int(pause_seconds * sample_rate)
    overlap_samples = min(int(overlap_seconds * sample_rate), max_samples - 1)
    padding = int(0.2 * sample_rate)

    chunks: list[tuple[int, int]] = []
    first_start, first_end = map(int, intervals[0])
    group_start = max(0, first_start - padding)
    group_end = min(total_samples, first_end + padding)
    previous_raw_end = first_end

    def finish_group(start: int, end: int) -> None:
        chunks.extend(
            _hard_split(start, end, max_samples, overlap_samples=overlap_samples)
        )

    for raw_start_value, raw_end_value in intervals[1:]:
        raw_start, raw_end = int(raw_start_value), int(raw_end_value)
        start = max(0, raw_start - padding)
        end = min(total_samples, raw_end + padding)
        gap = max(0, raw_start - previous_raw_end)
        current_duration = group_end - group_start
        projected_duration = end - group_start

        pause_boundary = gap >= pause_samples and current_duration >= ideal_min_samples
        ideal_boundary = (
            projected_duration > ideal_max_samples
            and current_duration >= ideal_min_samples
        )
        hard_boundary = projected_duration > max_samples
        if pause_boundary or ideal_boundary or hard_boundary:
            finish_group(group_start, group_end)
            group_start, group_end = start, end
        else:
            group_end = max(group_end, end)
        previous_raw_end = raw_end

    finish_group(group_start, group_end)
    return [(start, end) for start, end in chunks if end > start]


def build_speech_chunks(
    audio,
    sample_rate: int,
    max_seconds: int = 120,
    top_db: int = 35,
) -> list[tuple[int, int]]:
    """Split speech into language-detection-friendly utterance chunks."""
    import librosa

    intervals = librosa.effects.split(
        audio, top_db=top_db, frame_length=2048, hop_length=512
    )
    return group_speech_intervals(
        intervals,
        total_samples=len(audio),
        sample_rate=sample_rate,
        max_seconds=max_seconds,
    )


def deduplicate_overlapping_segments(segments: list[dict]) -> list[dict]:
    """Remove duplicated words produced by the 250ms hard-split overlap."""
    ordered = sorted(
        segments,
        key=lambda item: (float(item.get("start_time", 0)), float(item.get("end_time", 0))),
    )
    result: list[dict] = []
    for item in ordered:
        text = re.sub(r"\s+", "", str(item.get("text", ""))).casefold()
        start = float(item.get("start_time", 0))
        end = float(item.get("end_time", start))
        duplicate = False
        for previous in reversed(result[-8:]):
            previous_start = float(previous.get("start_time", 0))
            previous_end = float(previous.get("end_time", previous_start))
            if start - previous_end > 0.5:
                break
            previous_text = re.sub(
                r"\s+", "", str(previous.get("text", ""))
            ).casefold()
            overlap = min(end, previous_end) - max(start, previous_start)
            shorter = max(min(end - start, previous_end - previous_start), 0.001)
            if text and text == previous_text and overlap / shorter >= 0.5:
                duplicate = True
                break
        if not duplicate:
            result.append(item)
    return result


def detected_language_summary(transcripts: list[dict]) -> tuple[str, list[str]]:
    """Return an ordered language set and a non-misleading overall label."""
    languages = list(
        dict.fromkeys(
            str(transcript.get("language", "")).strip()
            for transcript in transcripts
            if str(transcript.get("language", "")).strip()
        )
    )
    overall = languages[0] if len(languages) == 1 else (
        "Multilingual" if languages else ""
    )
    return overall, languages


def fallback_segments(text: str, start: float, end: float) -> list[dict]:
    units = re.findall(r"[A-Za-zÀ-ž']+|\d+|[\u3400-\u9fff]|[^\s]", text)
    if not units:
        return []
    duration = max(end - start, 0.1)
    step = duration / len(units)
    return [
        {
            "text": unit,
            "start_time": start + index * step,
            "end_time": start + (index + 1) * step,
        }
        for index, unit in enumerate(units)
    ]


def release_model(model, processor=None) -> None:
    del model
    if processor is not None:
        del processor
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run(args) -> dict:
    import librosa
    import soundfile as sf
    import torch
    from transformers import (
        AutoModelForMultimodalLM,
        AutoModelForTokenClassification,
        AutoProcessor,
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请检查 NVIDIA 驱动及 Qwen 运行环境")
    device = "cuda:0" if args.device == "cuda" else "cpu"
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32

    emit_progress(2, "读取音频并检测静音边界")
    audio, sample_rate = librosa.load(args.audio, sr=16000, mono=True)
    # Auto language mode uses short utterances so each chunk gets its own
    # language decision. A fixed language may use the configured 30s ceiling.
    chunk_limit = min(args.chunk_seconds, 20) if not args.language else args.chunk_seconds
    chunk_ranges = build_speech_chunks(audio, sample_rate, chunk_limit)
    if not chunk_ranges:
        return {"segments": [], "language": args.language or ""}

    temp_dir = tempfile.TemporaryDirectory(prefix="qwen3-chunks-")
    chunk_files = []
    for index, (start, end) in enumerate(chunk_ranges):
        path = Path(temp_dir.name) / f"chunk-{index:04d}.wav"
        sf.write(path, audio[start:end], sample_rate)
        chunk_files.append(path)

    emit_progress(8, "载入 Qwen3-ASR-1.7B")
    asr_processor = AutoProcessor.from_pretrained(args.asr_model)
    asr_model = AutoModelForMultimodalLM.from_pretrained(
        args.asr_model,
        dtype=dtype,
        device_map=device,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).eval()

    transcripts = []
    total = len(chunk_files)
    for index, path in enumerate(chunk_files):
        progress = 10 + int(45 * index / max(total, 1))
        emit_progress(progress, f"Qwen3-ASR 转录 {index + 1}/{total}")
        request = asr_processor.apply_transcription_request(
            audio=str(path), language=args.language or None
        ).to(asr_model.device, asr_model.dtype)
        duration = (chunk_ranges[index][1] - chunk_ranges[index][0]) / sample_rate
        max_new_tokens = max(256, min(2048, int(duration * 8)))
        with torch.inference_mode():
            output_ids = asr_model.generate(
                **request, max_new_tokens=max_new_tokens, do_sample=False
            )
        generated_ids = output_ids[:, request["input_ids"].shape[1] :]
        parsed = asr_processor.decode(generated_ids, return_format="parsed")[0]
        transcripts.append(
            {
                "text": str(parsed.get("transcription", "")).strip(),
                "language": parsed.get("language")
                or LANGUAGE_NAMES.get(args.language, args.language or "English"),
            }
        )

    release_model(asr_model, asr_processor)

    segments = []
    if not args.aligner_model:
        for (start, end), transcript in zip(chunk_ranges, transcripts):
            segments.extend(
                {
                    **item,
                    "language": transcript["language"],
                }
                for item in fallback_segments(
                    transcript["text"], start / sample_rate, end / sample_rate
                )
            )
    else:
        emit_progress(58, "载入 Qwen3 Forced Aligner")
        aligner_processor = AutoProcessor.from_pretrained(args.aligner_model)
        aligner_model = AutoModelForTokenClassification.from_pretrained(
            args.aligner_model,
            dtype=dtype,
            device_map=device,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).eval()

        for index, (path, time_range, transcript) in enumerate(
            zip(chunk_files, chunk_ranges, transcripts)
        ):
            progress = 60 + int(35 * index / max(total, 1))
            emit_progress(progress, f"字幕时间轴对齐 {index + 1}/{total}")
            if not transcript["text"]:
                continue
            offset = time_range[0] / sample_rate
            try:
                aligner_inputs, word_lists = (
                    aligner_processor.prepare_forced_aligner_inputs(
                        audio=str(path),
                        transcript=transcript["text"],
                        language=transcript["language"],
                    )
                )
                aligner_inputs = aligner_inputs.to(
                    aligner_model.device, aligner_model.dtype
                )
                with torch.inference_mode():
                    outputs = aligner_model(**aligner_inputs)
                timestamps = aligner_processor.decode_forced_alignment(
                    logits=outputs.logits,
                    input_ids=aligner_inputs["input_ids"],
                    word_lists=word_lists,
                    timestamp_token_id=aligner_model.config.timestamp_token_id,
                )[0]
                segments.extend(
                    {
                        "text": item["text"],
                        "start_time": float(item["start_time"]) + offset,
                        "end_time": float(item["end_time"]) + offset,
                        "language": transcript["language"],
                    }
                    for item in timestamps
                )
            except Exception:
                segments.extend(
                    {
                        **item,
                        "language": transcript["language"],
                    }
                    for item in fallback_segments(
                        transcript["text"],
                        time_range[0] / sample_rate,
                        time_range[1] / sample_rate,
                    )
                )
        release_model(aligner_model, aligner_processor)

    temp_dir.cleanup()
    emit_progress(98, "整理 Qwen3-ASR 时间戳")
    segments = deduplicate_overlapping_segments(segments)
    overall_language, detected_languages = detected_language_summary(transcripts)
    return {
        "segments": segments,
        "language": overall_language,
        "languages": detected_languages,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--asr-model", required=True)
    parser.add_argument("--aligner-model", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--language", default="")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--chunk-seconds", type=int, default=30)
    args = parser.parse_args()
    result = run(args)
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    emit_progress(100, "Qwen3-ASR 完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
