import hashlib
import re
import unicodedata


WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def sanitize_filename_component(
    name: str,
    replacement: str = "_",
    max_length: int = 120,
    fallback: str = "media",
) -> str:
    """Return one filesystem-safe path component."""

    sanitized = unicodedata.normalize("NFC", str(name or ""))
    sanitized = re.sub(r'[<>:"/\\|?*]', replacement, sanitized)
    sanitized = re.sub(r"[\x00-\x1f]", "", sanitized)
    sanitized = sanitized.strip(" .")

    if not sanitized:
        sanitized = fallback

    base_name, separator, extension = sanitized.partition(".")
    if base_name.upper() in WINDOWS_RESERVED_NAMES:
        sanitized = (
            f"{base_name}_.{extension}" if separator else f"{base_name}_"
        )

    max_length = max(16, int(max_length))
    if len(sanitized) > max_length:
        digest = hashlib.sha1(sanitized.encode("utf-8")).hexdigest()[:8]
        prefix_length = max_length - len(digest) - 1
        sanitized = f"{sanitized[:prefix_length].rstrip(' .')}-{digest}"

    return sanitized
