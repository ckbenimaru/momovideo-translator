# momo 字幕助手

以 [VideoCaptioner](https://github.com/WEIFENG2333/VideoCaptioner) 為基礎修改的本地影片轉錄、字幕翻譯及影片合成工具。

本倉庫只保存核心原始碼及必要字型，不包含 Python 執行環境、FFmpeg、Faster-Whisper-XXL 程式、Qwen3-ASR 執行環境、Ollama、CUDA 元件或 AI 模型。這些大型元件由使用者按需要自行安裝或在軟體內下載。

## 主要改動

- 支援 Faster-Whisper-XXL `large-v3` 本地語音轉錄。
- 新增 Qwen3-ASR-1.7B 高準確率轉錄，以及 Qwen3-ForcedAligner-0.6B 詞／字級字幕時間軸對齊。
- 啟用詞級時間戳與 VAD，並按標點、停頓、時長、字數及閱讀速度重新切分字幕。
- 建立任務後自動完成語音轉錄及字幕優化／翻譯，影片合成改為使用者手動啟動。
- 合成期間可停止任務；輸出名稱採用 `原檔名.done.mp4`。
- 影片輸出為 MP4、H.264/AVC、AAC、`yuv420p` 並加入 Fast Start，方便 macOS 播放。
- NVIDIA 顯示卡優先使用 NVENC，失敗時自動回退至 `libx264`。
- 影片品質預設碼率：極高 8 Mbps、高 5 Mbps、中 3 Mbps、低 1.5 Mbps。
- 內置 momo／YouTube-style Modern Subtitle 字幕樣式及 Noto Sans 字型。

## 系統需求

- Windows 10／11 64 位元
- Python 3.10、3.11 或 3.12
- Git
- FFmpeg（需要可從命令列執行 `ffmpeg` 與 `ffprobe`）
- 建議使用 NVIDIA 顯示卡及近期驅動；沒有 NVIDIA 顯示卡時可使用 CPU／`libx264`

目前 Faster-Whisper-XXL 與 Qwen3-ASR CUDA 流程主要針對 Windows。輸出的 H.264／AAC MP4 影片可在 macOS 使用，但這不代表本專案已提供完整的 macOS 語音轉錄環境。

## 從原始碼安裝

建議先安裝 [uv](https://docs.astral.sh/uv/)，然後執行：

```powershell
git clone https://github.com/ckbenimaru/momovideo-translator.git
cd momovideo-translator
uv sync
uv run videocaptioner
```

也可以明確啟動圖形介面：

```powershell
uv run videocaptioner-gui
```

本倉庫目前是原始碼版本，不是可攜式免安裝軟體包。

## Faster Whisper 與模型

Faster-Whisper-XXL 執行檔及模型不會提交至 GitHub，避免倉庫體積過大。

啟動軟體後，在 Faster Whisper 設定頁面：

1. 下載或指定 Faster-Whisper-XXL 程式。
2. 下載 `faster-whisper-large-v3` 模型。
3. NVIDIA 顯示卡選擇 `cuda`；沒有相容顯示卡時選擇 `cpu`。
4. 確認程式及模型狀態顯示已安裝後再開始任務。

開發模式的預設位置：

```text
resource\bin\Faster-Whisper-XXL\faster-whisper-xxl.exe
AppData\models\faster-whisper-large-v3\
```

這些目錄已由 `.gitignore` 排除，不會被意外提交。

## Qwen3-ASR 與字幕時間軸

在「轉錄模型」選擇 `Qwen3-ASR ✨` 後，模型頁面提供一鍵管理：

1. 安裝／修復獨立 Qwen 執行環境。
2. 下載 `Qwen3-ASR-1.7B-hf`（約 4.09 GB）。
3. 下載 `Qwen3-ForcedAligner-0.6B-hf`（約 1.85 GB）。
4. RTX 4060 8GB 建議使用 `cuda`、30 秒單段上限，並保持 Forced Aligner 開啟。自動語言模式會按停頓建立約 5～15 秒片段，最長 20 秒；無停頓長句會保留 0.25 秒重疊後硬切並在合併時去重。

Qwen 使用獨立 Python 環境，不會把 PyTorch／Transformers 安裝到 momo 主環境。推理時先完成 ASR，再釋放顯存並載入 Forced Aligner，避免兩個模型同時佔用 8GB 顯存。若 Faster Whisper 模型已放在外接或 D 槽的 AI 目錄，Qwen 預設會沿用同一個 AI 根目錄：

```text
D:\AI\Qwen3-ASR-Runtime\
D:\AI\QwenModels\Qwen3-ASR-1.7B-hf\
D:\AI\QwenModels\Qwen3-ForcedAligner-0.6B-hf\
```

Qwen3-ASR 只負責離線語音辨識；辨識後的字幕會沿用 momo 現有的字幕重切分與翻譯流程。Forced Aligner 支援中文、英文、粵語、法文、德文、意大利文、日文、韓文、葡萄牙文、俄文及西班牙文。其他語言會使用估算時間戳回退。

## 預設離線翻譯

momo 預設使用 Ollama 的 `qwen3:4b-instruct-2507-q4_K_M` 翻譯字幕，建議將便攜環境與模型放在：

```text
D:\AI\Ollama\ollama.exe
D:\AI\OllamaModels\
```

翻譯服務地址為 `http://127.0.0.1:11434/v1`。程式開始翻譯時會自動啟動本機 Ollama、檢查模型是否完整，任務結束後自動卸載模型以釋放顯存。RTX 4060 8 GB 預設使用單一翻譯佇列及每批最多 8 條字幕，避免同時載入造成顯存不足。

簡體／繁體輸出會再經 OpenCC 統一；Google 免費翻譯接口仍保留為可手動選擇的快速備用。本專案不使用或要求 Google Cloud 付費 API。

## 預設字幕切分

- 理想時長：3.5 秒
- 最大時長：6 秒
- 最短時長：0.8 秒
- 優先停頓閾值：0.45 秒
- VAD 最短靜音：350 ms
- VAD 語音前後留白：200 ms
- 最多兩行
- 中文每行最多約 20 字，英文每行最多約 42 字

以上參數可在軟體介面調整。

## 隱私說明

- 使用 Faster Whisper 或 Qwen3-ASR 本地轉錄時，語音檔案不會上傳至 Bilibili、Google 或其他語音平台；模型權重只在安裝／下載階段連線取得。
- 使用 Google 翻譯時，字幕文字仍會傳送至 Google 翻譯服務。
- 使用預設 Qwen3-4B／Ollama 翻譯時，字幕文字只在本機處理，不會上傳第三方。
- 使用其他翻譯或 LLM 服務時，字幕文字及所提供的上下文會按該服務的流程傳送。

如需完全離線處理，請同時選用本地轉錄及本地翻譯方案。

## 輸出影片

硬字幕合成預設使用：

```text
容器：MP4
影像：H.264 / AVC (avc1)
音訊：AAC (mp4a)
像素格式：yuv420p
Fast Start：開啟
輸出名稱：原檔名.done.mp4
```

## 上游及授權

本專案衍生自 WEIFENG2333 的 [VideoCaptioner](https://github.com/WEIFENG2333/VideoCaptioner)，並保留原專案的 GPL-3.0 授權。

本專案依照 [GNU General Public License v3.0](LICENSE) 發佈。Noto Sans 字型使用 SIL Open Font License，授權文件位於 `videocaptioner/resources/fonts/`。

如要回報此修改版問題，請使用本倉庫的 [Issues](https://github.com/ckbenimaru/momovideo-translator/issues)。上游原始功能問題可先參考原專案文件及 Issues。
