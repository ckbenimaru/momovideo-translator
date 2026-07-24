"""Rounded background subtitle renderer"""

import os
import re
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

from PIL import Image, ImageDraw

from videocaptioner.core.entities import SubtitleLayoutEnum
from videocaptioner.core.utils.encoding_policy import (
    build_mac_encoding_args,
    get_target_bitrate_kbps,
)
from videocaptioner.core.utils.logger import setup_logger

from .font_utils import FontType, get_font
from .geometry import calculate_subtitle_geometry, probe_display_geometry
from .styles import RoundedBgStyle
from .text_utils import hex_to_rgba, is_mainly_cjk, wrap_text

if TYPE_CHECKING:
    from videocaptioner.core.asr.asr_data import ASRData

logger = setup_logger("subtitle.rounded")


def _scale_style_for_geometry(
    style: RoundedBgStyle,
    width: int,
    height: int,
    reference_height: int = 720,
) -> RoundedBgStyle:
    """Scale typography by the constraining axis and margins by frame size."""
    geometry = calculate_subtitle_geometry(
        width,
        height,
        reference_width=round(reference_height * 16 / 9),
        reference_height=reference_height,
    )
    scale = geometry.typography_scale
    return replace(
        style,
        font_size=max(1, int(round(style.font_size * scale))),
        corner_radius=max(1, int(round(style.corner_radius * scale))),
        padding_h=max(1, int(round(style.padding_h * scale))),
        padding_v=max(1, int(round(style.padding_v * scale))),
        margin_bottom=geometry.bottom_margin,
        line_spacing=max(1, int(round(style.line_spacing * scale))),
        letter_spacing=max(0, int(round(style.letter_spacing * scale))),
        # Keep the protective outline at the requested physical 1 px.
        outline_width=style.outline_width,
        shadow_offset_x=max(1, int(round(style.shadow_offset_x * scale))),
        shadow_offset_y=max(1, int(round(style.shadow_offset_y * scale))),
        max_chars_cjk=min(style.max_chars_cjk, geometry.max_chars_cjk),
        max_chars_english=min(
            style.max_chars_english, geometry.max_chars_english
        ),
    )

_TRADITIONAL_HINTS = set(
    "這個們來為與說時會後裡點體臺灣國學實發現開關過還進讓從對將於長見門間"
    "風電車書話語網畫頭聲應處無萬專業廣東龍麼麼麼愛歸區號檔預設顯示邊陰"
)


def _font_name_for_text(text: str, style: RoundedBgStyle) -> str:
    """Choose region-correct Noto Sans family for the subtitle text."""
    if not is_mainly_cjk(text):
        return style.font_name_en
    if any(char in _TRADITIONAL_HINTS for char in text):
        return style.font_name_tc
    return style.font_name_sc


def _split_by_character_limit(text: str, limit: int, cjk: bool) -> List[str]:
    """Split without truncating, respecting explicit newlines first."""
    result: List[str] = []
    for paragraph in (line.strip() for line in text.splitlines()):
        if not paragraph:
            continue
        if cjk:
            result.extend(
                paragraph[index : index + limit]
                for index in range(0, len(paragraph), limit)
            )
            continue

        current = ""
        for word in paragraph.split():
            if len(word) > limit:
                if current:
                    result.append(current)
                    current = ""
                result.extend(
                    word[index : index + limit]
                    for index in range(0, len(word), limit)
                )
                continue
            candidate = f"{current} {word}".strip()
            if current and len(candidate) > limit:
                result.append(current)
                current = word
            else:
                current = candidate
        if current:
            result.append(current)
    return result or ([text] if text else [])


def _fit_text_lines(
    text: str,
    width: int,
    style: RoundedBgStyle,
    max_lines: int,
) -> Tuple[FontType, List[str]]:
    """Fit text into at most ``max_lines`` while preserving every character."""
    font_name = _font_name_for_text(text, style)
    cjk = is_mainly_cjk(text)
    char_limit = style.max_chars_cjk if cjk else style.max_chars_english
    min_size = min(style.font_size, max(12, int(style.font_size * 0.65)))
    last_font = get_font(style.font_size, font_name, style.font_weight)
    last_lines: List[str] = []

    for size in range(style.font_size, min_size - 1, -2):
        font = get_font(size, font_name, style.font_weight)
        candidate_lines: List[str] = []
        for chunk in _split_by_character_limit(text, char_limit, cjk):
            candidate_lines.extend(
                wrap_text(
                    chunk,
                    font,
                    width,
                    style.padding_h,
                    extra_margin=int(width * 0.1),
                    spacing=style.letter_spacing,
                )
            )
        last_font, last_lines = font, candidate_lines
        if len(candidate_lines) <= max_lines:
            return font, candidate_lines

    # Extremely long externally supplied cues cannot be split in time here.
    # Preserve their full text and keep the visual contract at two lines.
    if len(last_lines) > max_lines:
        separator = "" if cjk else " "
        last_lines = last_lines[: max_lines - 1] + [
            separator.join(last_lines[max_lines - 1 :])
        ]
    return last_font, last_lines


def _get_video_info(video_path: str) -> Tuple[int, int, float]:
    """获取视频分辨率和时长"""
    result = subprocess.run(
        ["ffmpeg", "-i", video_path],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=(getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0),
    )

    # Prefer display dimensions so phone rotation metadata is respected.
    try:
        width, height = probe_display_geometry(video_path)
    except (OSError, RuntimeError, ValueError):
        width, height = 0, 0
        if match := re.search(r"Stream.*Video:.* (\d{2,5})x(\d{2,5})", result.stderr):
            width, height = int(match.group(1)), int(match.group(2))
    else:
        raise ValueError(f"Cannot get video resolution: {video_path}")

    # 解析时长
    duration = 0.0
    if match := re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr):
        h, m, s = match.groups()
        duration = int(h) * 3600 + int(m) * 60 + float(s)

    return width, height, duration


def render_text_block(
    draw: ImageDraw.ImageDraw,
    texts: List[str],
    font: FontType,
    center_x: int,
    top_y: float,
    style: RoundedBgStyle,
) -> float:
    """
    渲染多行文本块（共享圆角背景）

    Args:
        draw: PIL ImageDraw 对象
        texts: 文本行列表
        font: 字体对象
        center_x: 水平中心位置
        top_y: 顶部 y 坐标
        style: 样式配置

    Returns:
        背景框高度
    """
    if not texts:
        return 0

    bg_color = hex_to_rgba(style.bg_color)
    text_color = hex_to_rgba(style.text_color)
    outline_color = hex_to_rgba(style.outline_color)
    shadow_color = hex_to_rgba(style.shadow_color)

    # 计算All行的尺寸和垂直偏移
    line_sizes = []
    line_offsets = []
    for text in texts:
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        # 如果有字符间距，需要加上额外的宽度
        if style.letter_spacing > 0 and len(text) > 1:
            text_width += style.letter_spacing * (len(text) - 1)
        line_sizes.append(
            (
                text_width + style.outline_width * 2,
                bbox[3] - bbox[1] + style.outline_width * 2,
            )
        )
        line_offsets.append(bbox[1])  # 记录垂直偏移，用于居中对齐

    max_width = max(w for w, h in line_sizes)
    line_height = max(h for w, h in line_sizes)
    line_step = max(
        line_height + style.line_spacing,
        int(round(line_height * style.line_height_ratio)),
    )
    total_height = line_height + line_step * (len(texts) - 1)

    # 绘制共享背景
    bg_width = max_width + style.padding_h * 2
    bg_height = total_height + style.padding_v * 2
    bg_left = center_x - bg_width // 2
    bg_top = top_y

    draw.rounded_rectangle(
        [bg_left, bg_top, bg_left + bg_width, bg_top + bg_height],
        radius=style.corner_radius,
        fill=bg_color,
    )

    # 绘制文本（补偿字体垂直偏移）
    y = bg_top + style.padding_v
    for i, text in enumerate(texts):
        w, h = line_sizes[i]
        x = center_x - w // 2
        y_offset = line_offsets[i]
        text_y = y - y_offset  # 补偿垂直偏移，使文本视觉居中

        # 如果有字符间距，逐字符绘制
        def draw_character(position, value):
            shadow_position = (
                position[0] + style.shadow_offset_x,
                position[1] + style.shadow_offset_y,
            )
            draw.text(
                shadow_position,
                value,
                font=font,
                fill=shadow_color,
                stroke_width=style.outline_width,
                stroke_fill=shadow_color,
            )
            draw.text(
                position,
                value,
                font=font,
                fill=text_color,
                stroke_width=style.outline_width,
                stroke_fill=outline_color,
            )

        if style.letter_spacing > 0 and len(text) > 1:
            current_x = x
            for char in text:
                draw_character((current_x, text_y), char)
                char_width = font.getbbox(char)[2] - font.getbbox(char)[0]
                current_x += char_width + style.letter_spacing
        else:
            # 无字符间距，一次性绘制（性能更好）
            draw_character((x, text_y), text)

        y += line_step

    return bg_height


def render_subtitle_image(
    primary_text: str,
    secondary_text: str,
    width: int,
    height: int,
    style: RoundedBgStyle,
) -> Image.Image:
    """
    渲染单帧字幕图像（透明背景）

    Args:
        primary_text: 主字幕文本
        secondary_text: 副字幕文本
        width: 图像宽度
        height: 图像高度
        style: 样式配置

    Returns:
        PIL Image 对象（RGBA 格式）
    """
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # One language may occupy two lines; bilingual subtitles use one line each.
    primary_limit = 1 if primary_text and secondary_text else style.max_lines
    secondary_limit = 1 if primary_text and secondary_text else style.max_lines
    primary_font, primary_lines = (
        _fit_text_lines(primary_text, width, style, primary_limit)
        if primary_text
        else (get_font(style.font_size, style.font_name, style.font_weight), [])
    )
    secondary_font, secondary_lines = (
        _fit_text_lines(secondary_text, width, style, secondary_limit)
        if secondary_text
        else (primary_font, [])
    )

    center_x = width // 2

    # 计算总高度
    def calc_block_height(lines: List[str], font: FontType) -> float:
        if not lines:
            return 0
        bbox = font.getbbox("测试Ag")
        line_h = bbox[3] - bbox[1] + style.outline_width * 2
        line_step = max(
            line_h + style.line_spacing,
            int(round(line_h * style.line_height_ratio)),
        )
        return line_h + line_step * (len(lines) - 1) + style.padding_v * 2

    primary_height = calc_block_height(primary_lines, primary_font)
    secondary_height = calc_block_height(secondary_lines, secondary_font)
    gap = style.line_spacing if primary_lines and secondary_lines else 0
    total_height = primary_height + gap + secondary_height

    # 从底部计算起始位置
    bottom_y = height - style.margin_bottom
    start_y = bottom_y - total_height

    # 渲染文本块
    current_y = start_y
    if primary_lines:
        h = render_text_block(
            draw, primary_lines, primary_font, center_x, current_y, style
        )
        current_y += h + gap
    if secondary_lines:
        render_text_block(
            draw, secondary_lines, secondary_font, center_x, current_y, style
        )

    return image


def render_preview(
    primary_text: str,
    secondary_text: str = "",
    width: Optional[int] = None,
    height: Optional[int] = None,
    style: Optional[RoundedBgStyle] = None,
    bg_image_path: Optional[str] = None,
    reference_height: int = 720,
) -> str:
    """
    渲染圆角背景字幕预览图

    Args:
        primary_text: 主字幕文本
        secondary_text: 副字幕文本
        width: 图片宽度（None=从bg_image_path自动获取）
        height: 图片高度（None=从bg_image_path自动获取）
        style: 圆角背景样式（包含reference_height，会根据height自动缩放）
        bg_image_path: 背景图片路径
        reference_height: 参考高度（固定720P）
    Returns:
        生成的预览图路径
    """
    if style is None:
        style = RoundedBgStyle()

    # 加载或创建背景
    if bg_image_path and Path(bg_image_path).exists():
        background = Image.open(bg_image_path).convert("RGB")
        # 如果未提供尺寸，从图片获取
        if width is None or height is None:
            width, height = background.size
    else:
        # 没有背景图片，使用默认尺寸或提供的尺寸
        if width is None:
            width = 1920
        if height is None:
            height = 1080
        background = Image.new("RGB", (width, height), (20, 20, 20))

    # 确保 width 和 height 不为 None（类型收窄）
    assert width is not None and height is not None

    # 从样式中获取参考高度，根据图片高度自动缩放样式
    style = _scale_style_for_geometry(style, width, height, reference_height)

    # 渲染字幕并叠加
    subtitle_img = render_subtitle_image(primary_text, secondary_text, width, height, style)
    background.paste(subtitle_img, (0, 0), subtitle_img)

    # 保存到临时目录
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".png", delete=False) as tmp_file:
        background.save(tmp_file, "PNG")
        return tmp_file.name


def render_rounded_video(
    video_path: str,
    asr_data: "ASRData",
    output_path: str,
    rounded_style: Optional[dict] = None,
    layout: SubtitleLayoutEnum = SubtitleLayoutEnum.ONLY_ORIGINAL,
    crf: int = 23,
    preset: str = "medium",
    progress_callback: Optional[Callable] = None,
    reference_height: int = 720,
) -> None:
    """
    渲染圆角背景字幕到视频（分批overlay方案）

    核心流程: 直接分批overlay字幕PNG到原视频
    每批50个字幕，避免FFmpeg文件数量限制

    Args:
        video_path: 输入视频路径
        asr_data: 字幕数据
        output_path: 输出视频路径
        rounded_style: 圆角背景样式配置字典
        layout: 字幕布局
        crf: 视频质量参数
        preset: FFmpeg编码预设
        progress_callback: 进度回调 (progress: int, message: str)
        reference_height: 参考高度（固定720P）
    """
    # 检查字幕数据
    if not asr_data or not asr_data.segments:
        raise ValueError("Empty subtitle data, cannot render video")

    # 检查布局合理性
    if layout == SubtitleLayoutEnum.ONLY_TRANSLATE:
        has_translation = any(
            seg.translated_text and seg.translated_text.strip() for seg in asr_data.segments
        )
        if not has_translation:
            layout = SubtitleLayoutEnum.ONLY_ORIGINAL
    elif (
        layout == SubtitleLayoutEnum.TRANSLATE_ON_TOP
        or layout == SubtitleLayoutEnum.ORIGINAL_ON_TOP
    ):
        has_translation = any(
            seg.translated_text and seg.translated_text.strip() for seg in asr_data.segments
        )
        if not has_translation:
            layout = SubtitleLayoutEnum.ONLY_ORIGINAL

    # 获取视频信息
    width, height, video_duration = _get_video_info(video_path)
    final_encoding_args, source_info, max_rate = build_mac_encoding_args(
        video_path, output_path, crf, preset
    )
    using_nvenc = "h264_nvenc" in final_encoding_args
    target_rate_k = get_target_bitrate_kbps(crf)
    logger.info(
        "Mac-compatible encode: encoder=%s, source=%s %dx%d %.3ffps "
        "%.2fMbps, target=%dkbps, maxrate=%.2fMbps",
        "h264_nvenc P5 VBR" if using_nvenc else f"libx264 {preset} VBR",
        source_info.codec,
        source_info.width,
        source_info.height,
        source_info.fps,
        source_info.bit_rate / 1_000_000,
        target_rate_k,
        max_rate / 1_000_000,
    )

    # 构建并缩放样式
    style_config = rounded_style or {}
    style_config["layout"] = layout
    style = RoundedBgStyle(**style_config)

    style = _scale_style_for_geometry(style, width, height, reference_height)

    with tempfile.TemporaryDirectory(prefix="rounded_subtitle_") as temp_dir:
        temp_path = Path(temp_dir)

        # 步骤1: 生成All字幕PNG (0-30%)
        logger.debug(f"Generating subtitle PNGs图片（共{len(asr_data.segments)}个，布局: {layout.value}）")
        subtitle_frames = []

        for i, seg in enumerate(asr_data.segments):
            # 根据布局确定主副文本
            if layout == SubtitleLayoutEnum.ONLY_ORIGINAL:
                primary, secondary = seg.text, ""
            elif layout == SubtitleLayoutEnum.ONLY_TRANSLATE:
                primary, secondary = seg.translated_text or "", ""
            elif layout == SubtitleLayoutEnum.ORIGINAL_ON_TOP:
                primary, secondary = seg.text, seg.translated_text or ""
            else:  # TRANSLATE_ON_TOP
                primary, secondary = seg.translated_text or "", seg.text

            # 渲染字幕图片
            img = render_subtitle_image(primary, secondary, width, height, style)
            png_path = temp_path / f"subtitle_{i:06d}.png"
            img.save(png_path, "PNG")

            # 记录时间戳
            start_time = seg.start_time / 1000.0
            end_time = seg.end_time / 1000.0
            subtitle_frames.append((start_time, end_time, png_path))

            # 进度回调
            if progress_callback:
                progress = int((i + 1) / len(asr_data.segments) * 30)
                progress_callback(progress, f"生成字幕图片 {i + 1}/{len(asr_data.segments)}")

        if not subtitle_frames:
            raise ValueError("No valid subtitle images generated")

        # 步骤2: 分批overlay到视频 (30-100%)
        logger.debug("Overlaying subtitle batches onto video")
        BATCH_SIZE = 50
        current_video = video_path
        total_batches = (len(subtitle_frames) + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_idx in range(total_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min((batch_idx + 1) * BATCH_SIZE, len(subtitle_frames))
            batch_frames = subtitle_frames[start_idx:end_idx]

            # 构建overlay滤镜链
            input_args = ["-i", current_video]
            filter_parts = []

            for local_idx, (start, end, png_path) in enumerate(batch_frames):
                input_args.extend(["-i", str(png_path)])
                prev = f"[v{local_idx}]" if local_idx > 0 else "[0:v]"
                curr = f"[{local_idx + 1}:v]"
                out = f"[v{local_idx + 1}]"
                filter_parts.append(
                    f"{prev}{curr}overlay=0:0:enable='between(t,{start},{end})'{out}"
                )

            filter_complex = ";".join(filter_parts)
            final_output = f"[v{len(batch_frames)}]"

            # 判断是否是最后一批
            is_last_batch = batch_idx == total_batches - 1
            batch_output = (
                output_path if is_last_batch else temp_path / f"batch_{batch_idx:03d}.mp4"
            )

            logger.debug(f"Processing batch {batch_idx + 1}/{total_batches}（{len(batch_frames)}个字幕）")
            # 构建 ffmpeg Command
            # -t 参数强制保持原视频时长，防止因 overlay ended而截断视频
            encoding_args = (
                final_encoding_args
                if is_last_batch
                else [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-crf",
                    "0",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "copy",
                ]
            )
            cmd = [
                "ffmpeg",
                "-y",
                *input_args,
                "-filter_complex",
                filter_complex,
                "-map",
                final_output,
                "-map",
                "0:a?",
                "-t",
                str(video_duration),  # 强制保持原视频时长
                *encoding_args,
                str(batch_output),
            ]

            if batch_idx == 0 or is_last_batch:
                cmd_str = subprocess.list2cmdline(cmd)
                logger.debug(f"FFmpeg cmd: {cmd_str}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=(
                    getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
                ),
            )

            if result.returncode != 0 and is_last_batch and using_nvenc:
                # This callback raises immediately when the UI requested stop,
                # preventing a cancelled NVENC process from starting CPU retry.
                if progress_callback:
                    progress_callback(30, "NVENC失敗，正在切換CPU編碼")
                logger.warning(
                    "NVENC encoding failed (code %s); retrying final batch with libx264",
                    result.returncode,
                )
                fallback_args, _, _ = build_mac_encoding_args(
                    video_path,
                    output_path,
                    crf,
                    preset,
                    prefer_hardware=False,
                )
                fallback_cmd = [
                    "ffmpeg",
                    "-y",
                    *input_args,
                    "-filter_complex",
                    filter_complex,
                    "-map",
                    final_output,
                    "-map",
                    "0:a?",
                    "-t",
                    str(video_duration),
                    *fallback_args,
                    str(batch_output),
                ]
                result = subprocess.run(
                    fallback_cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=(
                        getattr(subprocess, "CREATE_NO_WINDOW", 0)
                        if os.name == "nt"
                        else 0
                    ),
                )
                if result.returncode == 0:
                    logger.info("NVENC fallback completed successfully with libx264")

            if result.returncode != 0:
                logger.error(f"批次 {batch_idx + 1} 失败: {result.stderr}")
                raise RuntimeError(f"Subtitle processing failed（批次 {batch_idx + 1}）")

            # 更新进度 (30-100%)
            if progress_callback:
                progress = 30 + int((batch_idx + 1) / total_batches * 70)
                progress_callback(progress, f"合成视频 {batch_idx + 1}/{total_batches}")

            # 更新当前视频
            current_video = str(batch_output)

        logger.debug("Video synthesis complete")
