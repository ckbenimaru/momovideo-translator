# Getting Started

If momo 字幕助手 is already installed on your Mac, open it from Applications. To run the source checkout, install Git, uv and FFmpeg with libass, then run:

```bash
git clone https://github.com/ckbenimaru/momovideo-translator.git
cd momovideo-translator
uv sync --locked
uv run --no-sync videocaptioner-gui
```

See the [Chinese getting-started guide](/guide/getting-started) for local Qwen3-ASR and Ollama setup.
