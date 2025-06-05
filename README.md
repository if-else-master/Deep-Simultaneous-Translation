# 即時克隆語音翻譯 (Deep Simultaneous Translation)
[MIT License](LICENSE)
這是一個強大的即時語音翻譯系統，能夠在保留說話者原始音色的情況下，將語音內容翻譯成另一種語言。系統使用最先進的AI模型來實現語音識別、翻譯和合成的無縫整合。

## 主要功能

- **即時語音識別**：使用先進的語音識別技術捕捉用戶語音
- **音色克隆**：保留說話者的獨特音色和語調
- **高品質翻譯**：提供準確、自然的翻譯結果
- **多語言支持**：支持中文和英文的雙向翻譯
- **實時處理**：快速響應的語音處理系統

## 技術架構

本專案結合了三個強大的AI模型：

### 1. XTTS-v2 (Coqui TTS)
用於語音合成和音色克隆。這是一個革命性的語音合成模型，能夠僅通過短短幾秒的音頻樣本，就能複製說話者的聲音特徵，並將其應用於不同語言的文本合成。

### 2. Gemini 2.0 Flash API
用於語音識別和翻譯。Google的最新多模態AI模型能夠處理音頻輸入，進行語音識別並翻譯成目標語言，提供高精度的文本輸出。

### 3. Whisper (OpenAI)
作為備選的語音識別模型。這個強大的開源語音識別模型可以識別和轉錄多種語言的語音內容，為系統提供穩定的後備選項。

## 系統要求

- Python 3.10
- PyTorch (最新版本)
- 足夠的硬碟空間（至少5GB用於模型存儲）
- 支持CUDA的GPU（推薦，但不是必須）
- 穩定的網絡連接（用於API請求）
- 使用Macbook M4pro 作為測試平台

## 安裝指南

1. 克隆此代碼庫：
```bash
git clone https://github.com/yourusername/Deep-Simultaneous-Translation.git
cd Deep-Simultaneous-Translation
```

2. 創建並激活虛擬環境：
```bash
python -m venv venv
source venv/bin/activate  # Windows上使用: venv\Scripts\activate
```

3. 安裝必要的依賴：
```bash
pip install -r requirements.txt
```

4. 下載XTTS-v2模型（如果尚未下載）：
```bash
# 模型將被自動下載到XTTS-v2目錄
# 或者從https://huggingface.co/coqui/XTTS-v2手動下載
```

## 使用方法

1. 首先，在main.py中設置您的Gemini API密鑰：
```python
GEMINI_API_KEY = "your_api_key_here"
```

2. 運行主程序：
```bash
python main.py
```

3. 遵循命令行界面的指示：
   - 輸入`start`或`s`開始錄音和翻譯流程
   - 輸入`lang en`設置翻譯為英文
   - 輸入`lang zh`設置翻譯為中文
   - 輸入`voices`查看已克隆的語音
   - 輸入`clean`清理所有克隆語音
   - 輸入`quit`或`q`退出程序

4. 完整流程說明：
   - 系統會先錄製您的語音
   - 自動克隆您的音色
   - 識別並翻譯您說的內容
   - 使用您的音色說出翻譯後的內容

## 核心代碼說明

以下是系統中一些重要的部分：

### 語音錄制
系統使用PyAudio來捕獲麥克風輸入，並保存為臨時WAV文件：
```python
def start_recording(self):
    """開始錄音"""
    self.is_recording = True
    self.audio_frames = []
    
    audio = pyaudio.PyAudio()
    stream = audio.open(format=self.format,
                      channels=self.channels,
                      rate=self.rate,
                      input=True,
                      frames_per_buffer=self.chunk)
    
    print("🎤 錄音中... 按 Enter 停止錄音")
    
    while self.is_recording:
        data = stream.read(self.chunk)
        self.audio_frames.append(data)
```

### 語音克隆
系統將錄製的音頻保存為參考語音，用於後續合成：
```python
def clone_voice(self, audio_file_path):
    """克隆語音 - 將錄音文件複製為語音參考"""
    try:
        print("🎭 正在克隆語音...")
        
        # 創建克隆語音存儲目錄
        if not os.path.exists("cloned_voices"):
            os.makedirs("cloned_voices")
        
        # 將原始錄音複製作為語音克隆參考
        cloned_voice_filename = f"cloned_voices/cloned_voice_{int(time.time())}.wav"
        shutil.copy2(audio_file_path, cloned_voice_filename)
        
        self.cloned_voice_path = cloned_voice_filename
        print(f"✅ 語音克隆完成，參考文件: {cloned_voice_filename}")
        
        return cloned_voice_filename
    except Exception as e:
        print(f"❌ 語音克隆過程發生錯誤: {e}")
        return None
```

### 語音識別與翻譯
使用Gemini API進行語音識別和翻譯：
```python
def transcribe_and_translate(self, audio_file_path, target_language="en"):
    """使用 Gemini 進行語音轉文字和翻譯"""
    try:
        print("🤖 正在使用 Gemini 進行語音識別和翻譯...")
        
        # 上傳音頻文件
        audio_file = genai.upload_file(path=audio_file_path)
        
        # 根據目標語言設定提示詞
        if target_language.lower() == "en":
            prompt = "請將這段音頻中的語音內容轉換為英文文字，如果原本就是英文就直接轉錄，如果是其他語言請翻譯成英文。只回傳最終的英文文字內容，不要包含其他說明。"
        elif target_language.lower() == "zh":
            prompt = "請將這段音頻中的語音內容轉換為繁體中文文字，如果原本就是中文就直接轉錄，如果是其他語言請翻譯成繁體中文。只回傳最終的繁體中文文字內容，不要包含其他說明。"
        
        # 發送請求到 Gemini
        response = self.model.generate_content([audio_file, prompt])
        translated_text = response.text.strip()
        
        return translated_text
    except Exception as e:
        print(f"❌ 翻譯過程發生錯誤: {e}")
        return None
```

### 語音合成
使用XTTS-v2模型和克隆的語音進行語音合成：
```python
def synthesize_speech_with_cloned_voice(self, text, output_file="output_speech.wav"):
    """使用克隆的語音合成語音"""
    try:
        print("🔊 正在使用克隆語音合成語音...")
        
        if not self.cloned_voice_path or not os.path.exists(self.cloned_voice_path):
            print("❌ 沒有可用的克隆語音，請先錄音進行語音克隆")
            return None
        
        # 檢測語言
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
        language = "zh-cn" if has_chinese else "en"
        
        outputs = self.xtts_model.synthesize(
            text,
            self.config,
            speaker_wav=self.cloned_voice_path,  # 使用克隆的語音
            gpt_cond_len=3,
            language=language,
        )
        
        # 保存音頻
        scipy.io.wavfile.write(output_file, rate=24000, data=outputs["wav"])
        
        return output_file
    except Exception as e:
        print(f"❌ 語音合成發生錯誤: {e}")
        return None
```

## requirements.txt

```
sounddevice
numpy
TTS==0.22.0
openai-whisper
google-generativeai
torch==2.1  # XTTS 與 Whisper 都依賴它
pyaudio
pygame
```

## 注意事項

1. **Gemini API密鑰**：您需要從[Google AI Studio](https://makersuite.google.com/app/apikey)獲取Gemini API密鑰。

2. **XTTS-v2模型**：模型文件應放在`XTTS-v2`目錄中，包括`config.json`和模型權重文件。您可以從[Hugging Face](https://huggingface.co/coqui/XTTS-v2)下載這些文件。

3. **Transformers版本**：XTTS-v2模型對transformers庫的版本有特定要求，請確保安裝的是4.49.0版本，以避免`GPT2InferenceModel`相關的錯誤。

4. **GPU加速**：如果您的系統有支持CUDA的GPU，本系統將自動使用GPU加速語音合成過程，大幅提高處理速度。

## 故障排除

- **語音識別錯誤**：確保您的麥克風工作正常，並在安靜的環境中錄音。
- **合成失敗**：檢查XTTS-v2模型文件是否完整，並確認transformers庫版本是否正確。
- **API錯誤**：確認您的Gemini API密鑰是否有效，以及網絡連接是否穩定。

## 授權信息

本專案中使用的XTTS-v2模型受[Coqui Public Model License](https://coqui.ai/cpml)約束。

## 聯繫方式

如有任何問題，請聯繫：[rayc57429@gmail.com]

---

© 2024 即時克隆語音翻譯 | Deep Simultaneous Translation

