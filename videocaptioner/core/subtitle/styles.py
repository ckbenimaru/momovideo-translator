"""Subtitle style configurations"""

from dataclasses import dataclass

from videocaptioner.core.entities import SubtitleLayoutEnum


@dataclass
class RoundedBgStyle:
    """YouTube-style modern rounded subtitle defaults (720p reference)."""

    font_name: str = "Noto Sans TC"
    font_name_tc: str = "Noto Sans TC"
    font_name_sc: str = "Noto Sans SC"
    font_name_en: str = "Noto Sans"
    font_weight: int = 500
    font_size: int = 42

    # 颜色配置（支持 hex 格式，如 #RRGGBB 或 #RRGGBBAA）
    bg_color: str = "#00000059"  # 黑色，35% 不透明度
    text_color: str = "#FFFFFF"  # 文字颜色
    outline_color: str = "#202020FF"
    outline_width: int = 1
    shadow_color: str = "#00000040"
    shadow_offset_x: int = 1
    shadow_offset_y: int = 2

    # 圆角和间距
    corner_radius: int = 14  # 圆角半径
    padding_h: int = 24  # 水平内边距
    padding_v: int = 12  # 垂直内边距
    margin_bottom: int = 58  # 720p 的约 8%
    line_spacing: int = 12  # 约 1.3 倍行高
    line_height_ratio: float = 1.3
    letter_spacing: int = 0  # 字符间距
    max_lines: int = 2
    max_chars_cjk: int = 20
    max_chars_english: int = 42

    # 字幕布局
    layout: SubtitleLayoutEnum = SubtitleLayoutEnum.ONLY_ORIGINAL
