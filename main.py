import os
import sounddevice as sd
import whisper
import google.generativeai as genai
from TTS.api import TTS
import numpy as np
import torch
import subprocess
import sys
from pathlib import Path

# ==== 環境初始化 ====
# 請將此處替換為你的 Google API Key
GOOGLE_API_KEY = ""
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

# ==== 檢查並安裝必要套件 ====
def check_disk_space(required_gb=3):
    """檢查磁碟空間（模型約需要 2GB）"""
    import shutil
    try:
        total, used, free = shutil.disk_usage("/")
        free_gb = free // (1024**3)
        if free_gb < required_gb:
            print(f"⚠️ 剩餘磁碟空間不足: {free_gb}GB，建議至少保留 {required_gb}GB 空間")
            return False
        else:
            print(f"✅ 磁碟空間充足: {free_gb}GB 可用")
            return True
    except:
        return True  # 如果無法檢查，假設空間足夠

def check_internet_connection():
    """檢查網路連線"""
    import urllib.request
    try:
        urllib.request.urlopen('https://www.google.com', timeout=5)
        print("✅ 網路連線正常")
        return True
    except:
        print("❌ 網路連線失敗，無法下載模型")
        return False

def check_dependencies():
    """檢查必要的套件是否已安裝"""
    required_packages = {
        'sounddevice': 'sounddevice',
        'whisper': 'openai-whisper',
        'google.generativeai': 'google-generativeai',
        'TTS': 'TTS',
        'soundfile': 'soundfile'
    }
    
    missing_packages = []
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"⚠️ 缺少必要套件: {', '.join(missing_packages)}")
        print("請執行以下命令安裝：")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    return True

# ==== 模型載入 ====
def check_whisper_model_exists(model_name="base"):
    """檢查 Whisper 模型是否已存在"""
    import whisper
    import os
    from pathlib import Path
    
    # 取得 Whisper 模型的下載目錄
    try:
        # 嘗試多種方式取得模型路徑
        download_root = os.path.expanduser(
            os.getenv(
                "XDG_CACHE_HOME", 
                os.path.join(os.path.expanduser("~"), ".cache")
            )
        )
        model_dir = Path(download_root) / "whisper"
        model_path = model_dir / f"{model_name}.pt"
        
        if model_path.exists():
            print(f"✅ 發現已下載的 Whisper {model_name} 模型: {model_path}")
            return True
        else:
            # 檢查其他可能的路徑
            alternative_paths = [
                Path.home() / ".cache" / "whisper" / f"{model_name}.pt",
                Path.home() / "whisper" / f"{model_name}.pt",
                Path("/tmp") / "whisper" / f"{model_name}.pt"
            ]
            
            for alt_path in alternative_paths:
                if alt_path.exists():
                    print(f"✅ 發現已下載的 Whisper {model_name} 模型: {alt_path}")
                    return True
            
            print(f"🔍 未發現 Whisper {model_name} 模型，需要下載")
            return False
            
    except Exception as e:
        print(f"⚠️ 檢查 Whisper 模型時發生錯誤: {e}")
        print(f"🔍 將嘗試載入 Whisper {model_name} 模型...")
        return False

def download_whisper_model(model_name="base"):
    """智慧載入 Whisper 模型（避免重複下載）"""
    print(f"🔄 準備載入 Whisper {model_name} 模型...")
    
    # 先檢查模型是否已存在
    try:
        model_exists = check_whisper_model_exists(model_name)
    except:
        model_exists = False
    
    import whisper
    try:
        # 針對 M4 晶片，建議使用 CPU 模式以避免 MPS 兼容性問題
        device = "cpu"  # 暫時使用 CPU，避免 MPS 兼容性問題
        print(f"🎯 使用設備: {device}（M4 優化模式）")
        
        if model_exists:
            print(f"⚡ 載入本地 Whisper {model_name} 模型...")
        else:
            print(f"📥 下載 Whisper {model_name} 模型（約 142MB）...")
            print("   首次下載會需要一些時間，請稍候...")
        
        # 載入模型時禁用 fp16 以確保兼容性
        model = whisper.load_model(model_name, device=device)
        
        if not model_exists:
            print(f"✅ Whisper {model_name} 模型下載並載入完成")
        else:
            print(f"✅ Whisper {model_name} 模型載入完成")
            
        return model
    except Exception as e:
        print(f"❌ Whisper 模型載入失敗: {e}")
        print("💡 常見解決方案：")
        print("   1. 檢查網路連線")
        print("   2. 重新安裝 whisper：pip uninstall openai-whisper && pip install git+https://github.com/openai/whisper.git")
        print("   3. 安裝 ffmpeg：brew install ffmpeg")
        return None

def check_xtts_model_exists():
    """檢查 XTTS 模型是否已存在"""
    import os
    from pathlib import Path
    
    # XTTS 模型的常見快取路徑
    possible_paths = [
        Path.home() / ".cache" / "tts" / "tts_models--multilingual--multi-dataset--xtts_v2",
        Path.home() / ".local" / "share" / "tts" / "tts_models--multilingual--multi-dataset--xtts_v2",
        Path("/tmp") / "tts" / "tts_models--multilingual--multi-dataset--xtts_v2"
    ]
    
    for path in possible_paths:
        if path.exists() and any(path.glob("*.pth")):
            print(f"✅ 發現已下載的 XTTS 模型: {path}")
            return True
    
    print("🔍 未發現 XTTS 模型，需要下載")
    return False

def download_xtts_model():
    """修正版的 TTS 模型載入函數"""
    print("🔄 載入 TTS 模型...")
    
    try:
        from TTS.api import TTS
        import torch
        
        # 輸出調試信息
        print(f"🔍 PyTorch 版本: {torch.__version__}")
        print(f"🔍 使用設備: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        
        # 修正：改用正確的方式取得模型列表
        print("🔍 查詢可用的 TTS 模型...")
        
        # 嘗試使用預定義的穩定模型（避免查詢所有模型）
        stable_models = [
            "tts_models/en/ljspeech/tacotron2-DDC",
            "tts_models/en/ljspeech/glow-tts", 
            "tts_models/en/ljspeech/fast_pitch",
            "tts_models/en/vctk/vits",
            "tts_models/zh-CN/baker/tacotron2-DDC-GST"
        ]
        
        print("🔍 嘗試載入穩定的 TTS 模型...")
        
        # 首先嘗試載入常見的穩定模型
        for model_name in stable_models:
            print(f"🔍 嘗試載入: {model_name}")
            try:
                tts = TTS(model_name=model_name)
                print(f"✅ 成功載入模型: {model_name}")
                return tts
            except Exception as e:
                print(f"⚠️ 載入 {model_name} 失敗: {str(e)[:100]}...")
                continue
        
        # 如果預定義模型都失敗，嘗試其他方法
        print("🔍 嘗試其他 TTS 模型載入方式...")
        
        # 方法1：嘗試載入默認英文模型
        try:
            print("🔍 嘗試載入默認英文模型...")
            tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
            print("✅ 成功載入默認英文模型")
            return tts
        except Exception as e:
            print(f"⚠️ 默認英文模型載入失敗: {str(e)[:100]}...")
        
        # 方法2：嘗試最簡單的初始化
        try:
            print("🔍 嘗試最簡單的 TTS 初始化...")
            # 不指定模型，讓 TTS 自動選擇
            tts = TTS()
            print("✅ 使用自動選擇的 TTS 模型")
            return tts
        except Exception as e:
            print(f"⚠️ 自動 TTS 模型載入失敗: {str(e)[:100]}...")
        
        # 方法3：嘗試指定設備
        try:
            print("🔍 嘗試指定設備載入 TTS...")
            device = "cpu"  # M4 使用 CPU
            tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC").to(device)
            print(f"✅ 成功載入 TTS 模型並設定到 {device}")
            return tts
        except Exception as e:
            print(f"⚠️ 指定設備載入失敗: {str(e)[:100]}...")
        
        # 如果所有方法都失敗
        raise Exception("所有 TTS 模型載入方法都失敗")
        
    except Exception as e:
        print(f"❌ TTS 模型載入失敗: {e}")
        print("💡 建議解決方案：")
        print("   1. 更新 TTS 庫：pip install -U TTS")
        print("   2. 重新安裝 TTS：pip uninstall TTS && pip install TTS")
        print("   3. 檢查網路連線（首次下載需要網路）")
        print("   4. 確保有足夠磁碟空間（約2GB）")
        raise

def load_models():
    """載入所有 AI 模型"""
    print("🚀 開始載入 AI 模型...")
    
    # 智慧載入 Whisper 模型
    asr_model = download_whisper_model("base")
    if asr_model is None:
        raise Exception("Whisper 模型載入失敗")
    
    # 智慧載入 TTS 模型  
    try:
        tts_model = download_xtts_model()
        if tts_model is None:
            raise Exception("TTS 模型返回為空")
    except Exception as e:
        print(f"❌ TTS 模型載入失敗: {e}")
        raise Exception("TTS 模型載入失敗")
    
    print("🎉 所有模型載入完成！")
    return asr_model, tts_model

# ==== 參數設定 ====
LANG_ASR = "zh"           # 語音辨識語言：中文
LANG_TRANSLATE = "English"  # 翻譯目標語言
LANG_TTS = "en"           # TTS 輸出語言代碼
SAMPLE_RATE = 16000       # 採樣率
DEFAULT_DURATION = 5      # 預設錄音時長（秒）
SPEAKER_WAV = "reference.wav"  # 語音合成參考音頻（放在voice_output目錄下）

# 建立輸出目錄
OUTPUT_DIR = Path("voice_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ==== 函式區 ====

def record_audio(duration=DEFAULT_DURATION, samplerate=SAMPLE_RATE):
    """錄製音訊"""
    print(f"🎙️ 錄音中...（{duration} 秒）")
    try:
        audio = sd.rec(int(duration * samplerate), 
                      samplerate=samplerate, 
                      channels=1, 
                      dtype='float32')
        sd.wait()
        print("✅ 錄音完成")
        return audio.flatten()
    except Exception as e:
        print(f"❌ 錄音失敗: {e}")
        return None

def transcribe_audio(audio, model):
    """語音轉文字"""
    try:
        if audio is None:
            return ""
        # 針對 M4 優化：使用 fp16=False 確保穩定性
        result = model.transcribe(audio, language=LANG_ASR, fp16=False)
        return result['text']
    except Exception as e:
        print(f"❌ 語音辨識失敗: {e}")
        return ""

def translate_text_with_gemini(text, target_language=LANG_TRANSLATE):
    """使用 Gemini 翻譯文字"""
    try:
        if not text.strip():
            return ""
        
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"請將下列繁體中文內容翻譯成 {target_language}，只回傳翻譯結果：\n{text}"
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"❌ 翻譯失敗: {e}")
        return text

def speak_text(text, tts_model, lang=LANG_TTS, output_file="output.wav"):
    """修正版的文字轉語音函數"""
    try:
        if not text.strip():
            print("⚠️ 沒有文字可以轉換為語音")
            return
        
        print("🗣️ 語音合成中...")
        output_path = OUTPUT_DIR / output_file
        
        # 修正版的 TTS 調用
        print(f"🔊 目標語言: {lang}")
        
        success = False
        error_messages = []
        
        # 方法1: 最簡單的調用（對大多數模型有效）
        try:
            tts_model.tts_to_file(text=text, file_path=str(output_path))
            success = True
            print("✅ 語音合成成功")
        except Exception as e:
            error_messages.append(f"基本 API 失敗: {str(e)[:100]}")
        
        # 方法2: 嘗試指定語言（如果模型支持）
        if not success:
            try:
                if hasattr(tts_model, 'languages') and lang in tts_model.languages:
                    tts_model.tts_to_file(
                        text=text,
                        file_path=str(output_path),
                        language=lang
                    )
                    success = True
                    print("✅ 使用指定語言成功")
            except Exception as e:
                error_messages.append(f"指定語言 API 失敗: {str(e)[:100]}")
        
        # 方法3: 嘗試使用說話者（多說話者模型）
        if not success:
            try:
                # 安全地檢查說話者
                speakers = None
                if hasattr(tts_model, 'speakers') and tts_model.speakers:
                    speakers = tts_model.speakers
                elif hasattr(tts_model, 'synthesizer') and hasattr(tts_model.synthesizer, 'tts_speakers_file'):
                    # 嘗試獲取說話者列表
                    try:
                        speakers = list(tts_model.synthesizer.tts_config.speakers)
                    except:
                        pass
                
                if speakers and len(speakers) > 0:
                    speaker = speakers[0] if isinstance(speakers, list) else str(speakers[0])
                    print(f"🎤 使用說話者: {speaker}")
                    tts_model.tts_to_file(
                        text=text,
                        file_path=str(output_path),
                        speaker=speaker
                    )
                    success = True
                    print("✅ 使用說話者參數成功")
            except Exception as e:
                error_messages.append(f"說話者 API 失敗: {str(e)[:100]}")
        
        if not success:
            raise Exception(f"所有語音合成方法都失敗: {'; '.join(error_messages)}")
        
        # 確認檔案已生成
        if not output_path.exists():
            raise Exception("語音檔案未成功生成")
        
        # 在 macOS 上播放音訊
        play_audio(str(output_path))
        
    except Exception as e:
        print(f"❌ 語音合成失敗: {e}")
        print("💡 建議解決方案：")
        print("   1. 檢查模型是否正確載入")
        print("   2. 嘗試較短的文字")
        print("   3. 更新 TTS 庫: pip install -U TTS")

def play_audio(file_path):
    """播放音訊檔案"""
    try:
        # macOS 使用 afplay
        result = subprocess.run(["afplay", file_path], 
                              check=True, 
                              capture_output=True, 
                              text=True)
        print("🔊 音訊播放完成")
    except subprocess.CalledProcessError as e:
        print(f"❌ 播放音訊失敗: {e}")
        print("💡 您也可以手動播放檔案：", file_path)
    except FileNotFoundError:
        print("❌ 找不到 afplay 命令")
        print("💡 請確認在 macOS 系統上執行，或手動播放檔案：", file_path)

def get_audio_devices():
    """列出可用的音訊設備"""
    try:
        print("🎧 可用的音訊設備：")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"  {i}: {device['name']} (輸入)")
    except Exception as e:
        print(f"⚠️ 無法列出音訊設備: {e}")

# ==== 主流程 ====

def main_loop():
    """主要執行迴圈"""
    # 檢查系統環境
    print("🔍 檢查系統環境...")
    
    # 檢查依賴套件
    if not check_dependencies():
        return
    
    # 先檢查是否有現有模型來決定磁碟空間需求
    try:
        whisper_exists = check_whisper_model_exists("base")
        xtts_exists = check_xtts_model_exists()
    except Exception as e:
        print(f"⚠️ 檢查模型時發生錯誤: {e}")
        print("🔄 將繼續執行並嘗試載入模型...")
        whisper_exists = False
        xtts_exists = False
    
    # 根據需要下載的模型調整磁碟空間需求
    required_space = 0
    if not whisper_exists:
        required_space += 1  # Whisper base 約 142MB，算作 1GB
    if not xtts_exists:
        required_space += 2  # TTS 模型約 1-2GB，算作 2GB
    
    if required_space > 0:
        print(f"📥 需要下載約 {required_space}GB 的模型檔案")
        if not check_disk_space(required_space):
            response = input("是否繼續執行？(y/N): ")
            if response.lower() != 'y':
                return
        
        # 檢查網路連線（只有需要下載時才檢查）
        if not check_internet_connection():
            print("💡 提示：需要網路連線來下載模型")
            response = input("是否繼續執行？(y/N): ")
            if response.lower() != 'y':
                return
    else:
        print("⚡ 所有模型已存在，可離線使用")
    
    # 檢查 API Key
    if GOOGLE_API_KEY == "your_google_api_key_here":
        print("❌ 請先設定您的 Google API Key")
        print("請到 https://makersuite.google.com/app/apikey 取得 API Key")
        return
    
    # 載入模型（智慧載入，避免重複下載）
    try:
        asr_model, tts_model = load_models()
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        return
    
    # 顯示音訊設備
    get_audio_devices()
    
    print("\n🎯 語音翻譯程式已準備就緒！")
    print("📝 使用說明：")
    print("  - 按 Enter 開始錄音（5秒）")
    print("  - 輸入數字可設定錄音時長")
    print("  - 輸入 'q' 退出程式")
    print("  - 輸入 'test' 測試語音合成")
    print("  - 輸入 'voice' 錄製並保存參考音頻樣本")
    
    while True:
        try:
            user_input = input("\n🔁 請選擇操作: ").strip()
            
            if user_input.lower() == 'q':
                print("👋 程式結束")
                break
            elif user_input.lower() == 'test':
                test_text = "This is a test of the text-to-speech system. If you can hear this message, the system is working properly."
                speak_text(test_text, tts_model, lang="en")
                continue
            elif user_input.lower() == 'voice':
                # 錄製參考音頻
                print("🎙️ 請錄製5秒鐘的參考音頻（請清晰說話）...")
                ref_audio = record_audio(duration=5)
                if ref_audio is None:
                    continue
                
                # 保存為WAV文件
                import soundfile as sf
                ref_path = OUTPUT_DIR / SPEAKER_WAV
                sf.write(str(ref_path), ref_audio, SAMPLE_RATE)
                print(f"✅ 參考音頻已保存至: {ref_path}")
                print("📝 下次語音合成將使用您的聲音特徵")
                continue
            elif user_input.isdigit():
                duration = int(user_input)
                if duration > 30:
                    print("⚠️ 錄音時長限制為30秒")
                    duration = 30
            else:
                duration = DEFAULT_DURATION
            
            # 錄音
            audio = record_audio(duration=duration)
            if audio is None:
                continue
            
            # 語音辨識
            text = transcribe_audio(audio, asr_model)
            if not text.strip():
                print("⚠️ 沒有辨識到語音內容")
                continue
            
            print(f"📄 辨識結果: {text}")
            
            # 翻譯
            translated = translate_text_with_gemini(text)
            if translated:
                print(f"🌍 翻譯結果: {translated}")
                
                # 語音合成
                speak_text(translated, tts_model)
            else:
                print("⚠️ 翻譯失敗")
                
        except KeyboardInterrupt:
            print("\n👋 程式中斷")
            break
        except Exception as e:
            print(f"❌ 發生錯誤: {e}")

# ==== 執行程式 ====
if __name__ == "__main__":
    print("🚀 MacBook M4 語音翻譯程式 (修正版)")
    print("=" * 50)
    print("📋 程式功能：")
    print("  • 中文語音辨識 → 英文翻譯 → 英文語音")
    print("  • 支援自訂語音合成聲音 (輸入 'voice' 錄製參考音頻)")
    print("  • 使用本地 TTS 模型")
    print("  • 針對 M4 晶片優化")
    print("⚠️ 安裝提醒：")
    print("  • 請確保已安裝 ffmpeg：brew install ffmpeg")
    print("  • 請安裝所需套件：pip install soundfile TTS openai-whisper sounddevice google-generativeai")
    print("  • 如遇到問題，請重新安裝 whisper：")
    print("    pip uninstall openai-whisper")
    print("    pip install git+https://github.com/openai/whisper.git")
    print("-" * 50)
    main_loop()