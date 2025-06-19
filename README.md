# 即時語音克隆翻譯系統 (Real-Time Voice Cloning Translation System)

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Status](https://img.shields.io/badge/Status-Active-green.svg)

這是一個革命性的即時語音翻譯系統，能夠**邊聽邊翻譯邊輸出**，並在保留說話者原始音色的情況下，將語音內容實時翻譯成多種語言。系統採用最先進的AI模型來實現語音識別、翻譯和合成的無縫整合。

## ✨ 主要功能

- **🎤 即時語音處理**：真正的邊聽邊翻譯邊輸出，支持同時進行音頻捕獲、翻譯處理和語音播放
- **🎭 高精度音色克隆**：僅需3-5秒音頻樣本即可克隆用戶音色
- **🌍 多語言支持**：支持 9 種語言互譯（中文、英文、日文、韓文、西班牙文、法文、德文、意大利文、葡萄牙文）
- **🤖 斷點檢測**：自動識別語音開始和停頓，無需手動控制
- **⚡ 多線程架構**：音頻捕獲、翻譯處理、語音播放並行進行
- **🔒 安全性**：API Key 安全輸入，臨時文件自動清理

## 🎯 核心特色

### 真正的即時翻譯
- 用戶說話時系統同時進行音頻捕獲
- 檢測到語音停頓（1秒）後立即開始翻譯處理
- 翻譯完成後使用用戶克隆的語音立即播放
- 用戶可以在系統播放翻譯的同時繼續說話

### 智能語音活動檢測
- 自動檢測語音開始和結束
- 最小語音長度過濾（0.3秒）避免雜音干擾
- 可調節的靜音閾值和持續時間

## 🏗️ 技術架構

本專案結合了多個先進的AI模型和技術：

### 1. **XTTS-v2 (Coqui TTS)**
- 用於語音合成和音色克隆
- 支持多語言語音合成（包括日語、韓語等）
- 僅需短音頻樣本即可克隆音色

### 2. **Gemini 2.0 Flash API**
- 用於語音識別和翻譯
- 支持多模態輸入（音頻 + 文本）
- 高精度的語音轉文字和翻譯

### 3. **MeCab + unidic-lite**
- 日語文本分析和處理
- 支持高質量的日語語音合成
- 自動配置和錯誤處理
- start.sh 檢查 MeCab 檢測

## 💻 系統要求

- **作業系統**：macOS (推薦 M4 Pro)、Linux
- **Python**：3.10
- **記憶體**：至少 8GB RAM
- **硬碟空間**：至少 5GB（用於模型存儲）
- **網絡**：穩定的網絡連接（用於 API 請求）
- **音頻設備**：麥克風和揚聲器

## 🚀 快速安裝

### 1. 克隆專案
```bash
git clone https://github.com/yourusername/Deep-Simultaneous-Translation.git
cd Deep-Simultaneous-Translation
```

### 2. 創建虛擬環境
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. 安裝依賴項
```bash
pip install -r requirements.txt
```

### 4. macOS 用戶額外設置（日語支持）
```bash
brew install mecab mecab-ipadic
```

## 📋 依賴項清單

```txt
sounddevice
numpy
TTS==0.22.0
openai-whisper
google-generativeai
torch==2.1
pyaudio
pygame
cutlet
fugashi              # 日語文本處理庫
unidic-lite         # 日語詞典，支援 XTTS 日語語音合成
```

## 🎯 使用方法

### 啟動系統

**方法一：使用啟動腳本（推薦）**
```bash
chmod +x start.sh
./start.sh
```

**方法二：直接運行**
```bash
python main.py
```

### 操作流程

1. **📡 設置 Gemini API Key**
   - 系統會提示您安全輸入 API Key
   - 從 [Google AI Studio](https://makersuite.google.com/app/apikey) 獲取

2. **🌍 選擇語言**
   ```
   支持的語言:
   zh: 中文    en: 英文    ja: 日文
   ko: 韓文    es: 西班牙文  fr: 法文
   de: 德文    it: 意大利文  pt: 葡萄牙文
   ```

3. **🎭 克隆語音（按 c）**
   - 用自然語調說一段話（3-5秒）
   - 系統會自動檢測語音開始和結束
   - 語音樣本將保存在 `cloned_voices/` 目錄

4. **⚡ 開始即時翻譯（按 Enter）**
   - 系統進入持續監聽模式
   - 開始說話，停頓1秒後自動翻譯
   - 使用您的克隆語音即時播放翻譯結果

## 🎛️ 操作界面

```
📋 操作選項:
  c - 克隆語音（必須先完成）
  enter - 開始即時翻譯
  q - 退出程序

✅ 語音已克隆，可以開始即時翻譯
```

## 🔧 核心代碼架構

### 多線程即時處理
```python
# 三個並行工作線程
1. audio_capture_worker()    # 音頻捕獲線程
2. translation_worker()      # 翻譯處理線程  
3. playback_worker()         # 音頻播放線程
```

### 智能語音檢測
```python
def detect_voice_activity(self, audio_data):
    """語音活動檢測"""
    rms = self.calculate_rms(audio_data)
    # 音量閾值檢測 + 時間窗口分析
    # 自動識別說話開始和停頓
```

### 多語言語音合成
```python
def synthesize_speech(self, text):
    """使用克隆語音合成多語言語音"""
    language_map = {
        'zh': 'zh-cn', 'en': 'en', 'ja': 'ja',
        'ko': 'ko', 'es': 'es', 'fr': 'fr',
        'de': 'de', 'it': 'it', 'pt': 'pt'
    }
    # 自動語言檢測 + 錯誤處理
```

## 🛠️ 高級功能

### 自動 MeCab 配置
系統會自動檢測並配置 MeCab（日語處理）：
```python
def setup_mecab():
    # 自動檢測 unidic-lite 詞典
    # 備用系統 MeCab 配置
    # 智能錯誤處理
```

### 智能錯誤恢復
- 日語處理失敗時自動使用英語合成
- API 錯誤重試機制
- 臨時文件自動清理

## 🎵 支持的語言組合

| 原始語言 | 目標語言 | 狀態 |
|---------|---------|------|
| 中文 | 日文 | ✅ 完全支持 |
| 中文 | 英文 | ✅ 完全支持 |
| 英文 | 中文 | ✅ 完全支持 |
| 英文 | 日文 | ✅ 完全支持 |
| 其他語言 | 任意支持語言 | ✅ 完全支持 |

## 🔍 故障排除

### 常見問題

**1. MeCab 初始化錯誤**
```bash
# 解決方案：系統已自動處理，會使用 unidic-lite 備用詞典
```

**2. 語音合成失敗**
```
⚠️ 日語處理組件問題，嘗試使用英語合成...
🔊 使用英語語音合成完成
```

**3. API Key 無效**
```bash
❌ API Key 無效: 400 API key not valid
是否重新輸入？(y/n): y
```

**4. 音頻設備問題**
- 檢查麥克風權限
- 確認音頻設備工作正常
- 在安靜環境中使用

### 性能優化

**記憶體優化**
- 臨時文件自動清理
- 音頻緩衝區大小限制
- 佇列管理

## 📁 專案結構

```
Deep-Simultaneous-Translation/
├── main.py              # 主程序
├── start.sh             # 啟動腳本
├── requirements.txt     # 依賴項
├── README.md           # 使用說明
├── XTTS-v2/            # XTTS 模型文件
│   ├── config.json
│   ├── model.pth
│   └── ...
├── cloned_voices/      # 克隆語音存儲
└── voice_output/       # 音頻輸出（臨時）
```

## 🔐 安全與隱私

- **API Key 安全輸入**：使用 `getpass` 避免明文顯示
- **本地音頻處理**：語音克隆完全在本地進行
- **臨時文件清理**：自動清理所有臨時音頻文件
- **網絡隱私**：僅翻譯時向 Gemini API 發送音頻

## 🤝 貢獻指南

歡迎提交 Issue 和 Pull Request！

1. Fork 本專案
2. 創建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

## 📜 授權信息

- **本專案**：MIT License
- **XTTS-v2 模型**：[Coqui Public Model License](https://huggingface.co/coqui/XTTS-v2)
- **Gemini API**：遵循 Google AI 使用條款

## 📞 聯繫方式

- **Email**：[rayc57429@gmail.com]
- **GitHub Issues**：[提交問題](https://github.com/yourusername/Deep-Simultaneous-Translation/issues)

## 🙏 致謝

感謝以下開源專案：
- [Coqui TTS](https://github.com/coqui-ai/TTS) - XTTS-v2 語音合成
- [Google Gemini](https://ai.google.dev/) - 語音識別和翻譯
- [MeCab](https://taku910.github.io/mecab/) - 日語文本分析

---

<div align="center">

**🎤 即時語音克隆翻譯系統 | Deep-Simultaneous-Translation**

*讓語言不再是溝通的障礙*

© 2024 Deep Simultaneous Translation

</div>

