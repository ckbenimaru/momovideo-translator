#!/bin/bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
logo="$repo_dir/videocaptioner/resources/logo.png"
app="${1:-/Applications/momo 字幕助手.app}"
app="$(cd "$(dirname "$app")" && pwd)/$(basename "$app")"
plist="$app/Contents/Info.plist"
bundle_id="$(/usr/libexec/PlistBuddy -c 'Print CFBundleIdentifier' "$plist")"
icon_file="$(/usr/libexec/PlistBuddy -c 'Print CFBundleIconFile' "$plist")"
if [[ "$bundle_id" != com.ckbenimaru.momovideo-translator.local-launcher ||
      "$icon_file" != Momo.icns ]]; then
    printf '%s\n' '找不到相容的 momo macOS 啟動器。' >&2
    exit 1
fi

work_dir="$(mktemp -d "$(dirname "$app")/.momo-icon.XXXXXXXX")"
previous_app="$work_dir/previous/$(basename "$app")"
cleanup() {
    if [[ -d "$previous_app" && ! -e "$app" ]]; then
        if ! mv "$previous_app" "$app"; then
            printf '原啟動器位於 %s\n' "$previous_app" >&2
            return 1
        fi
    fi
    rm -rf "$work_dir"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

iconset="$work_dir/Momo.iconset"
mkdir "$iconset" "$work_dir/previous"
for size in 16 32 128 256 512; do
    /usr/bin/sips -z "$size" "$size" "$logo" \
        --out "$iconset/icon_${size}x${size}.png" >/dev/null
    /usr/bin/sips -z "$((size * 2))" "$((size * 2))" "$logo" \
        --out "$iconset/icon_${size}x${size}@2x.png" >/dev/null
done

staged_app="$work_dir/$(basename "$app")"
/usr/bin/ditto "$app" "$staged_app"
/usr/bin/iconutil -c icns "$iconset" -o "$staged_app/Contents/Resources/$icon_file"
/usr/bin/codesign --force --sign - "$staged_app"
/usr/bin/codesign --verify --strict "$staged_app"
mv "$app" "$previous_app"
mv "$staged_app" "$app"
touch "$app"
printf '已更新 %s 的圖示。\n' "$app"
