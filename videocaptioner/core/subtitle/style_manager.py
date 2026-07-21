"""Unified subtitle style management.

Single source of truth for loading, converting, and listing subtitle styles.
Both CLI and UI import from here.
"""

import json
import logging
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class StyleMode(Enum):
    ASS = "ass"
    ROUNDED = "rounded"


@dataclass
class SecondaryStyle:
    """Secondary (bilingual) subtitle style for ASS mode."""

    font_name: str = "Noto Sans TC"
    font_size: int = 30
    color: str = "#ffffff"
    outline_color: str = "#20202037"
    outline_width: float = 1.0
    spacing: float = 0.0


@dataclass
class SubtitleStyle:
    """Unified subtitle style definition.

    A single dataclass that covers both ASS and rounded modes.
    Fields irrelevant to the current mode are simply ignored.
    """

    # -- Metadata --
    name: str = ""
    description: str = ""
    mode: StyleMode = StyleMode.ASS

    # -- Common --
    font_name: str = "Noto Sans TC"
    font_size: int = 42

    # -- ASS mode fields --
    primary_color: str = "#ffffff"
    # With ASS BorderStyle=3, libass paints the box from OutlineColour.
    # Its alpha therefore needs to match the translucent background.
    outline_color: str = "#20202037"
    # ASS BorderStyle=3 draws an opaque-box style background.  A lighter
    # 22% fill avoids the visually over-dark result produced by libass.
    outline_width: float = 1.0
    back_color: str = "#00000037"
    border_style: int = 3
    shadow: float = 0.0
    bold: bool = False
    spacing: float = 0.0
    margin_horizontal: int = 20
    margin_bottom: int = 58
    secondary: Optional[SecondaryStyle] = None

    # -- Rounded mode fields --
    text_color: str = "#ffffff"
    bg_color: str = "#00000059"
    corner_radius: int = 14
    padding_h: int = 24
    padding_v: int = 18
    line_spacing: int = 12
    letter_spacing: int = 1
    margin_bottom_rounded: int = 58

    # ------------------------------------------------------------------ #
    # Conversion helpers
    # ------------------------------------------------------------------ #

    def to_ass_string(self) -> str:
        """Render as ASS V4+ Styles section (for FFmpeg)."""
        primary = _hex_to_ass(self.primary_color)
        outline = _hex_to_ass(self.outline_color)
        back = _hex_to_ass(self.back_color)
        bold_flag = -1 if self.bold else 0

        sec = self.secondary or SecondaryStyle(
            font_name=self.font_name,
            font_size=int(self.font_size * 0.7),
        )
        sec_color = _hex_to_ass(sec.color)
        sec_outline = _hex_to_ass(sec.outline_color)

        header = (
            "[V4+ Styles]\n"
            "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,"
            "OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,"
            "ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
            "Alignment,MarginL,MarginR,MarginV,Encoding"
        )
        default_line = (
            f"Style: Default,{self.font_name},{self.font_size},"
            f"{primary},&H000000FF,{outline},{back},"
            f"{bold_flag},0,0,0,100,100,{self.spacing},0,{self.border_style},"
            f"{self.outline_width},{self.shadow},2,{self.margin_horizontal},"
            f"{self.margin_horizontal},{self.margin_bottom},1"
        )
        secondary_line = (
            f"Style: Secondary,{sec.font_name},{sec.font_size},"
            f"{sec_color},&H000000FF,{sec_outline},{back},"
            f"{bold_flag},0,0,0,100,100,{sec.spacing},0,{self.border_style},"
            f"{sec.outline_width},{self.shadow},2,{self.margin_horizontal},"
            f"{self.margin_horizontal},{self.margin_bottom},1"
        )
        return f"{header}\n{default_line}\n{secondary_line}"

    def to_rounded_dict(self) -> dict:
        """Return the dict expected by the rounded renderer."""
        return {
            "font_name": self.font_name,
            "font_size": self.font_size,
            "text_color": self.text_color,
            "bg_color": self.bg_color,
            "corner_radius": self.corner_radius,
            "padding_h": self.padding_h,
            "padding_v": self.padding_v,
            "margin_bottom": self.margin_bottom_rounded,
            "line_spacing": self.line_spacing,
            "letter_spacing": self.letter_spacing,
        }

    def to_json_dict(self) -> dict:
        """Serialize to a JSON-friendly dict (for saving)."""
        d: dict = {"name": self.name, "description": self.description, "mode": self.mode.value}
        if self.mode == StyleMode.ROUNDED:
            d.update(self.to_rounded_dict())
        else:
            d.update({
                "font_name": self.font_name,
                "font_size": self.font_size,
                "primary_color": self.primary_color,
                "outline_color": self.outline_color,
                "outline_width": self.outline_width,
                "back_color": self.back_color,
                "border_style": self.border_style,
                "shadow": self.shadow,
                "bold": self.bold,
                "spacing": self.spacing,
                "margin_horizontal": self.margin_horizontal,
                "margin_bottom": self.margin_bottom,
            })
            if self.secondary:
                d["secondary"] = asdict(self.secondary)
        return d

    # ------------------------------------------------------------------ #
    # Factory methods
    # ------------------------------------------------------------------ #

    @classmethod
    def from_json(cls, data: dict) -> "SubtitleStyle":
        """Create from a parsed JSON dict. Auto-detects mode if not specified."""
        if "mode" in data:
            mode = StyleMode(data["mode"])
        elif any(k in data for k in ("bg_color", "text_color", "corner_radius")):
            mode = StyleMode.ROUNDED
        else:
            mode = StyleMode.ASS
        sec_data = data.get("secondary")
        secondary = SecondaryStyle(**sec_data) if isinstance(sec_data, dict) else None

        if mode == StyleMode.ROUNDED:
            return cls(
                name=data.get("name", ""),
                description=data.get("description", ""),
                mode=mode,
                font_name=data.get("font_name", cls.font_name),
                font_size=data.get("font_size", cls.font_size),
                text_color=data.get("text_color", cls.text_color),
                bg_color=data.get("bg_color", cls.bg_color),
                corner_radius=data.get("corner_radius", cls.corner_radius),
                padding_h=data.get("padding_h", cls.padding_h),
                padding_v=data.get("padding_v", cls.padding_v),
                margin_bottom_rounded=data.get("margin_bottom", cls.margin_bottom_rounded),
                line_spacing=data.get("line_spacing", cls.line_spacing),
                letter_spacing=data.get("letter_spacing", cls.letter_spacing),
            )

        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            mode=mode,
            font_name=data.get("font_name", cls.font_name),
            font_size=data.get("font_size", cls.font_size),
            primary_color=data.get("primary_color", cls.primary_color),
            outline_color=data.get("outline_color", cls.outline_color),
            outline_width=data.get("outline_width", cls.outline_width),
            back_color=data.get("back_color", cls.back_color),
            border_style=data.get("border_style", cls.border_style),
            shadow=data.get("shadow", cls.shadow),
            bold=data.get("bold", cls.bold),
            spacing=data.get("spacing", cls.spacing),
            margin_horizontal=data.get("margin_horizontal", cls.margin_horizontal),
            margin_bottom=data.get("margin_bottom", cls.margin_bottom),
            secondary=secondary,
        )

    @classmethod
    def from_file(cls, path: Path) -> "SubtitleStyle":
        """Load from a file (.json or legacy .txt)."""
        content = path.read_text(encoding="utf-8")

        if path.suffix == ".json":
            return cls.from_json(json.loads(content))

        # Legacy .txt fallback: parse ASS V4+ Style lines
        if "[V4+ Styles]" in content or "Style:" in content:
            return _parse_ass_txt(content, path.stem)

        raise ValueError(f"Unrecognized style file format: {path}")

    @classmethod
    def from_rounded_dict(cls, data: dict) -> "SubtitleStyle":
        """Create a rounded style from a flat dict (used by UI config)."""
        data_with_mode = {**data, "mode": "rounded"}
        return cls.from_json(data_with_mode)


# ------------------------------------------------------------------ #
# Module-level helpers
# ------------------------------------------------------------------ #

YOUTUBE_MODERN_NAME = "youtube-modern"


def ensure_youtube_modern_styles(styles_dir: Optional[Path] = None) -> None:
    """Install the built-in modern presets into the user's style selector."""
    if styles_dir is None:
        styles_dir = _default_styles_dir()
    styles_dir.mkdir(parents=True, exist_ok=True)

    ass_path = styles_dir / "ass-youtube-modern.json"
    rounded_path = styles_dir / "rounded-youtube-modern.json"
    ass_data = SubtitleStyle(
        name=YOUTUBE_MODERN_NAME,
        description="YouTube-style modern subtitle",
    ).to_json_dict()
    rounded_data = {
        "name": YOUTUBE_MODERN_NAME,
        "description": "YouTube-style modern subtitle",
        "mode": "rounded",
        "font_name": "Noto Sans TC",
        "font_name_tc": "Noto Sans TC",
        "font_name_sc": "Noto Sans SC",
        "font_name_en": "Noto Sans",
        "font_weight": 500,
        "font_size": 42,
        "text_color": "#FFFFFF",
        "bg_color": "#00000059",
        "outline_color": "#202020FF",
        "outline_width": 1,
        "shadow_color": "#00000040",
        "shadow_offset_x": 1,
        "shadow_offset_y": 2,
        "corner_radius": 14,
        "padding_h": 24,
        "padding_v": 12,
        "line_spacing": 12,
        "line_height_ratio": 1.3,
        "letter_spacing": 0,
        "margin_bottom": 58,
        "max_lines": 2,
        "max_chars_cjk": 20,
        "max_chars_english": 42,
    }

    for path, data in ((ass_path, ass_data), (rounded_path, rounded_data)):
        serialized = json.dumps(data, ensure_ascii=False, indent=2)
        if not path.exists() or path.read_text(encoding="utf-8") != serialized:
            path.write_text(serialized, encoding="utf-8")


def style_id_from_filename(filename: str) -> str:
    """Extract user-facing style ID from filename.

    'ass-default.json' -> 'default', 'rounded-dark.json' -> 'dark'
    """
    stem = Path(filename).stem
    for prefix in ("ass-", "rounded-"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def list_styles(styles_dir: Optional[Path] = None) -> List[SubtitleStyle]:
    """List all available styles in the directory."""
    if styles_dir is None:
        styles_dir = _default_styles_dir()
    ensure_youtube_modern_styles(styles_dir)
    if not styles_dir.exists():
        return []

    result: List[SubtitleStyle] = []
    for f in sorted(styles_dir.glob("*.json")):
        try:
            style = SubtitleStyle.from_file(f)
            # Use filename-derived ID if name not set in JSON
            if not style.name:
                style.name = style_id_from_filename(f.name)
            result.append(style)
        except Exception:
            logger.warning("Failed to load style %s", f)
    return result


def load_style(
    name: str,
    styles_dir: Optional[Path] = None,
    mode: Optional[str] = None,
) -> Optional[SubtitleStyle]:
    """Load a style by name (e.g. 'default', 'anime', 'rounded').

    Args:
        name: Style preset name.
        styles_dir: Directory containing style JSON files.
        mode: Preferred render mode ('ass' or 'rounded'). When set, the
              matching prefix is tried first so that ``load_style("default",
              mode="rounded")`` finds ``rounded-default.json`` before
              ``ass-default.json``.

    Searches by: exact filename match, prefixed filename (ass-X, rounded-X),
    or JSON 'name' field.
    """
    if styles_dir is None:
        styles_dir = _default_styles_dir()
    ensure_youtube_modern_styles(styles_dir)
    if not styles_dir.exists():
        return None

    # Try exact filename: <name>.json
    exact = styles_dir / f"{name}.json"
    if exact.exists():
        try:
            return SubtitleStyle.from_file(exact)
        except Exception:
            pass

    # Try prefixed filenames: ass-<name>.json, rounded-<name>.json
    # When *mode* is given, try the preferred prefix first.
    prefixes = ["ass-", "rounded-"]
    if mode == "rounded":
        prefixes = ["rounded-", "ass-"]
    for prefix in prefixes:
        prefixed = styles_dir / f"{prefix}{name}.json"
        if prefixed.exists():
            try:
                return SubtitleStyle.from_file(prefixed)
            except Exception:
                pass

    # Fallback: scan all files and match by JSON 'name' field
    for f in styles_dir.glob("*.json"):
        try:
            style = SubtitleStyle.from_file(f)
            if style.name == name or style_id_from_filename(f.name) == name:
                return style
        except Exception:
            pass

    return None


def available_style_names(styles_dir: Optional[Path] = None) -> List[str]:
    """Return sorted list of unique style names."""
    names = []
    for s in list_styles(styles_dir):
        names.append(s.name)
    return sorted(set(names))


# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #

def _default_styles_dir() -> Path:
    from videocaptioner.config import SUBTITLE_STYLE_PATH
    return SUBTITLE_STYLE_PATH


def _hex_to_ass(hex_color: str) -> str:
    """Convert #RRGGBB[AA] to ASS &HAABBGGRR (ASS alpha is inverted)."""
    h = hex_color.lstrip("#")
    if len(h) == 8:
        # Application colors use #RRGGBBAA, while ASS uses inverted alpha.
        r, g, b, a = h[0:2], h[2:4], h[4:6], h[6:8]
        ass_alpha = 255 - int(a, 16)
        return f"&H{ass_alpha:02X}{b}{g}{r}"
    if len(h) == 6:
        r, g, b = h[0:2], h[2:4], h[4:6]
        return f"&H00{b}{g}{r}"
    return "&H00ffffff"


def _ass_color_to_hex(ass_color: str) -> str:
    """Convert ASS &HAABBGGRR to #RRGGBB hex."""
    c = ass_color.strip().lstrip("&Hh")
    if len(c) == 8:
        b, g, r = c[2:4], c[4:6], c[6:8]
    elif len(c) == 6:
        b, g, r = c[0:2], c[2:4], c[4:6]
    else:
        return "#ffffff"
    return f"#{r}{g}{b}"


def _parse_ass_txt(content: str, stem: str = "") -> SubtitleStyle:
    """Parse a legacy .txt file containing ASS V4+ Style lines."""
    kwargs: dict = {"name": stem, "mode": StyleMode.ASS}
    secondary_kwargs: dict = {}

    for line in content.splitlines():
        line = line.strip()
        if line.startswith("Style: Default,"):
            parts = line.split(",")
            kwargs["font_name"] = parts[1]
            kwargs["font_size"] = int(parts[2])
            kwargs["primary_color"] = _ass_color_to_hex(parts[3])
            kwargs["outline_color"] = _ass_color_to_hex(parts[5])
            kwargs["bold"] = parts[7].strip() == "-1"
            kwargs["spacing"] = float(parts[13])
            kwargs["outline_width"] = float(parts[16])
            kwargs["margin_bottom"] = int(parts[21])

        elif line.startswith("Style: Secondary,"):
            parts = line.split(",")
            secondary_kwargs["font_name"] = parts[1]
            secondary_kwargs["font_size"] = int(parts[2])
            secondary_kwargs["color"] = _ass_color_to_hex(parts[3])
            secondary_kwargs["outline_color"] = _ass_color_to_hex(parts[5])
            secondary_kwargs["spacing"] = float(parts[13])
            secondary_kwargs["outline_width"] = float(parts[16])

    if secondary_kwargs:
        kwargs["secondary"] = SecondaryStyle(**secondary_kwargs)

    return SubtitleStyle(**kwargs)
