import pyaudio
import wave
import threading
import time
import io
import google.generativeai as genai
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch
import scipy.io.wavfile
import numpy as np
import pygame
import tempfile
import os
import shutil
import queue
from collections import deque
import getpass

def setup_mecab():
    """設置 MeCab 配置以支持日語處理"""
    try:
        import unidic_lite
        dicdir = unidic_lite.dicdir
        mecabrc_path = os.path.join(dicdir, 'mecabrc')
        if os.path.exists(mecabrc_path):
            os.environ['MECABRC'] = mecabrc_path
            print("✅ 使用 unidic-lite 詞典")
            return True
        else:
            print("✅ unidic-lite 可用，使用預設配置")
            return True
    except (ImportError, AttributeError):
        pass
    possible_paths = [
        '/opt/homebrew/etc/mecabrc',  # Homebrew Apple Silicon
        '/usr/local/etc/mecabrc',     # Homebrew Intel
        '/usr/etc/mecabrc',           # 系統安裝
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            os.environ['MECABRC'] = path
            print(f"✅ 使用系統 MeCab 配置: {path}")
            return True
    
    print("⚠️ 未找到 MeCab 配置，日語處理可能受限")
    return False

# 初始化 MeCab 配置
mecab_available = setup_mecab()

class RealTimeVoiceTranslationSystem:
    def __init__(self):
        # 系統狀態
        self.gemini_api_key = None
        self.model = None
        self.xtts_model = None
        self.config = None
        
        # 語言設置
        self.source_language = None
        self.target_language = None
        self.supported_languages = {
            'zh': '中文',
            'en': '英文', 
            'ja': '日文',
            'ko': '韓文',
            'es': '西班牙文',
            'fr': '法文',
            'de': '德文',
            'it': '意大利文',
            'pt': '葡萄牙文'
        }
        
        # 音頻參數
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        
        # 語音克隆
        self.cloned_voice_path = None
        self.is_voice_cloned = False
        
        # 即時翻譯控制
        self.is_real_time_active = False
        self.should_stop = False
        
        # 音頻處理隊列
        self.audio_segments_queue = queue.Queue()
        self.translation_queue = queue.Queue()
        self.playback_queue = queue.Queue()
        
        # 語音活動檢測
        self.silence_threshold = 500
        self.silence_duration = 1.0  # 降低到1秒以提高響應速度
        self.min_speech_duration = 0.3
        
        # 音頻緩衝
        self.audio_buffer = deque(maxlen=int(self.rate * 10))
        self.current_segment = []
        self.last_speech_time = 0
        self.is_speech_detected = False
        
        # 初始化pygame
        pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=1024)
        
        print("🎤 即時語音克隆翻譯系統已啟動！")
    
    def setup_system(self):
        """系統初始化設置"""
        print("=" * 60)
        print("🚀 即時語音克隆翻譯系統設置")
        print("=" * 60)
        
        # 1. 輸入API Key
        if not self.setup_gemini_api():
            return False
        
        # 2. 選擇語言
        if not self.setup_languages():
            return False
        
        # 3. 載入XTTS模型
        if not self.load_xtts_model():
            return False
        
        return True
    
    def setup_gemini_api(self):
        """設置Gemini API"""
        print("\n📡 設置 Gemini API")
        print("-" * 30)
        
        while True:
            try:
                api_key = getpass.getpass("請輸入您的 Gemini API Key: ").strip()
                
                if not api_key:
                    print("❌ API Key 不能為空，請重新輸入")
                    continue
                
                # 測試API Key
                genai.configure(api_key=api_key)
                test_model = genai.GenerativeModel('gemini-2.0-flash-exp')
                test_response = test_model.generate_content("測試")
                
                self.gemini_api_key = api_key
                self.model = test_model
                print("✅ Gemini API 設置成功！")
                return True
                
            except Exception as e:
                print(f"❌ API Key 無效: {e}")
                retry = input("是否重新輸入？(y/n): ").strip().lower()
                if retry != 'y':
                    return False
    
    def setup_languages(self):
        """設置語言選項"""
        print("\n🌍 語言設置")
        print("-" * 30)
        
        # 顯示支持的語言
        print("支持的語言:")
        for code, name in self.supported_languages.items():
            print(f"  {code}: {name}")
        
        # 選擇原始語言
        while True:
            source = input("\n請選擇您的原始語言代碼 (例: zh): ").strip().lower()
            if source in self.supported_languages:
                self.source_language = source
                print(f"✅ 原始語言: {self.supported_languages[source]}")
                break
            else:
                print("❌ 不支持的語言代碼，請重新選擇")
        
        # 選擇目標語言
        while True:
            target = input("請選擇目標語言代碼 (例: ja): ").strip().lower()
            if target in self.supported_languages:
                if target == self.source_language:
                    print("⚠️ 目標語言與原始語言相同，將直接轉錄語音")
                self.target_language = target
                print(f"✅ 目標語言: {self.supported_languages[target]}")
                break
            else:
                print("❌ 不支持的語言代碼，請重新選擇")
        
        return True
    
    def load_xtts_model(self):
        """載入XTTS模型"""
        print("\n🤖 載入 XTTS 語音合成模型...")
        print("-" * 30)
        
        try:
            config = XttsConfig()
            config.load_json("XTTS-v2/config.json")
            self.xtts_model = Xtts.init_from_config(config)
            self.xtts_model.load_checkpoint(config, checkpoint_dir="XTTS-v2/", eval=True)
            
            if torch.cuda.is_available():
                self.xtts_model.cuda()
                print("✅ 使用 GPU 加速")
            else:
                print("✅ 使用 CPU 運算")
            
            self.config = config
            print("✅ XTTS 模型載入成功！")
            return True
            
        except Exception as e:
            print(f"❌ 載入 XTTS 模型失敗: {e}")
            return False
    
    def clone_voice_step(self):
        """語音克隆步驟"""
        print("\n🎭 語音克隆步驟")
        print("-" * 30)
        print("📋 說明：系統將錄製您的聲音樣本用於語音克隆")
        print("💡 請用您的自然語調說一段話（建議3-5秒）")
        
        input("準備好後按 Enter 開始錄音...")
        
        # 錄音
        print("🎤 正在錄音... 請開始說話")
        audio_frames = []
        
        audio = pyaudio.PyAudio()
        stream = audio.open(format=self.format,
                          channels=self.channels,
                          rate=self.rate,
                          input=True,
                          frames_per_buffer=self.chunk)
        
        start_time = time.time()
        silence_start = None
        
        while True:
            data = stream.read(self.chunk)
            audio_frames.append(data)
            
            # 檢測音量
            rms = self.calculate_rms(data)
            current_time = time.time()
            
            if rms > self.silence_threshold:
                silence_start = None
                if current_time - start_time >= 0.5:  # 至少錄音0.5秒
                    print("🔊 檢測到語音...")
            else:
                if silence_start is None:
                    silence_start = current_time
                elif current_time - silence_start >= 2.0 and current_time - start_time >= 1.0:
                    # 靜音2秒且總錄音時間超過1秒
                    break
            
            # 最大錄音時間限制
            if current_time - start_time >= 10:
                break
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        print("📝 錄音完成，正在處理...")
        
        # 保存語音克隆樣本
        try:
            if not os.path.exists("cloned_voices"):
                os.makedirs("cloned_voices")
            
            clone_file = f"cloned_voices/voice_clone_{int(time.time())}.wav"
            
            wf = wave.open(clone_file, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(audio_frames))
            wf.close()
            
            self.cloned_voice_path = clone_file
            self.is_voice_cloned = True
            print(f"✅ 語音克隆完成！樣本已保存: {clone_file}")
            return True
            
        except Exception as e:
            print(f"❌ 語音克隆失敗: {e}")
            return False
    
    def calculate_rms(self, audio_data):
        """計算音頻RMS值"""
        if not audio_data or len(audio_data) == 0:
            return 0
        
        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            if len(audio_np) == 0:
                return 0
            
            # 計算RMS值，避免數學警告
            mean_square = np.mean(audio_np.astype(np.float64)**2)
            if mean_square < 0:
                return 0
            
            return np.sqrt(mean_square)
        except (ValueError, OverflowError):
            return 0
    
    def detect_voice_activity(self, audio_data):
        """語音活動檢測"""
        rms = self.calculate_rms(audio_data)
        current_time = time.time()
        
        if rms > self.silence_threshold:
            if not self.is_speech_detected:
                self.is_speech_detected = True
                print("🎤 開始說話...")
            self.last_speech_time = current_time
            return True
        else:
            if self.is_speech_detected:
                silence_duration = current_time - self.last_speech_time
                if silence_duration >= self.silence_duration:
                    self.is_speech_detected = False
                    print("⏸️ 檢測到停頓，處理語音...")
                    return False
            return self.is_speech_detected
    
    def audio_capture_worker(self):
        """音頻捕獲工作線程"""
        audio = pyaudio.PyAudio()
        stream = audio.open(format=self.format,
                          channels=self.channels,
                          rate=self.rate,
                          input=True,
                          frames_per_buffer=self.chunk)
        
        print("🎤 開始即時語音監聽...")
        
        while self.is_real_time_active:
            try:
                data = stream.read(self.chunk, exception_on_overflow=False)
                self.audio_buffer.extend(np.frombuffer(data, dtype=np.int16))
                
                is_speech = self.detect_voice_activity(data)
                
                if is_speech:
                    self.current_segment.extend(np.frombuffer(data, dtype=np.int16))
                elif self.current_segment and len(self.current_segment) > int(self.rate * self.min_speech_duration):
                    # 語音段結束，發送處理
                    segment_audio = np.array(self.current_segment, dtype=np.int16)
                    self.audio_segments_queue.put(segment_audio)
                    self.current_segment = []
                
                if self.should_stop:
                    break
                    
            except Exception as e:
                print(f"❌ 音頻捕獲錯誤: {e}")
                break
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        print("🎤 音頻捕獲已停止")
    
    def translation_worker(self):
        """翻譯處理工作線程"""
        print("🔄 翻譯處理線程已啟動")
        
        while self.is_real_time_active or not self.audio_segments_queue.empty():
            try:
                audio_segment = self.audio_segments_queue.get(timeout=1)
                
                # 保存音頻段到臨時文件
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                scipy.io.wavfile.write(temp_file.name, self.rate, audio_segment)
                
                # 翻譯
                translated_text = self.transcribe_and_translate(temp_file.name)
                
                if translated_text and translated_text.strip():
                    # 語音合成
                    output_file = self.synthesize_speech(translated_text)
                    if output_file:
                        self.playback_queue.put(output_file)
                
                # 清理臨時文件
                os.unlink(temp_file.name)
                self.audio_segments_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ 翻譯處理錯誤: {e}")
        
        print("🔄 翻譯處理線程已停止")
    
    def playback_worker(self):
        """音頻播放工作線程"""
        print("🔊 音頻播放線程已啟動")
        
        while self.is_real_time_active or not self.playback_queue.empty():
            try:
                audio_file = self.playback_queue.get(timeout=1)
                
                # 播放音頻
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                
                # 等待播放完成
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                    if self.should_stop:
                        pygame.mixer.music.stop()
                        break
                
                # 清理音頻文件
                try:
                    os.unlink(audio_file)
                except:
                    pass
                
                self.playback_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ 音頻播放錯誤: {e}")
        
        print("🔊 音頻播放線程已停止")
    
    def transcribe_and_translate(self, audio_file_path):
        """語音轉文字和翻譯"""
        try:
            audio_file = genai.upload_file(path=audio_file_path)
            
            # 構建翻譯提示詞
            source_lang_name = self.supported_languages[self.source_language]
            target_lang_name = self.supported_languages[self.target_language]
            
            if self.source_language == self.target_language:
                prompt = f"請將這段音頻中的{source_lang_name}語音內容轉換為文字。只回傳轉錄的文字內容，不要包含其他說明。"
            else:
                prompt = f"請將這段音頻中的{source_lang_name}語音內容翻譯為{target_lang_name}。只回傳翻譯後的文字內容，不要包含其他說明。"
            
            response = self.model.generate_content([audio_file, prompt])
            genai.delete_file(audio_file.name)
            
            result = response.text.strip()
            print(f"📝 翻譯: {result}")
            return result
            
        except Exception as e:
            print(f"❌ 翻譯錯誤: {e}")
            return None
    
    def synthesize_speech(self, text):
        """使用克隆語音合成語音"""
        try:
            if not self.cloned_voice_path or not os.path.exists(self.cloned_voice_path):
                print("❌ 沒有可用的克隆語音")
                return None
            
            # 檢測目標語言
            language_map = {
                'zh': 'zh-cn',
                'en': 'en',
                'ja': 'ja',
                'ko': 'ko',
                'es': 'es',
                'fr': 'fr',
                'de': 'de',
                'it': 'it',
                'pt': 'pt'
            }
            
            xtts_language = language_map.get(self.target_language, 'en')
            
            # 針對日語處理，添加特殊處理
            if xtts_language == 'ja':
                print("🇯🇵 正在合成日語語音...")
                # 檢查是否有可用的日語詞典
                if not mecab_available:
                    print("⚠️ 日語詞典不可用，將使用英語合成")
                    xtts_language = 'en'
            
            outputs = self.xtts_model.synthesize(
                text,
                self.config,
                speaker_wav=self.cloned_voice_path,
                gpt_cond_len=3,
                language=xtts_language,
            )
            
            # 保存音頻
            output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            scipy.io.wavfile.write(output_file.name, rate=24000, data=outputs["wav"])
            
            language_name = {'zh-cn': '中文', 'en': '英語', 'ja': '日語'}.get(xtts_language, xtts_language)
            print(f"🔊 {language_name}語音合成完成")
            return output_file.name
            
        except Exception as e:
            error_msg = str(e)
            if any(keyword in error_msg for keyword in ["MeCab", "fugashi", "dictionary format", "GenericTagger"]):
                print("⚠️ 日語處理組件問題，嘗試使用英語合成...")
                try:
                    # 嘗試用英語合成
                    outputs = self.xtts_model.synthesize(
                        text,
                        self.config,
                        speaker_wav=self.cloned_voice_path,
                        gpt_cond_len=3,
                        language='en',
                    )
                    output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    scipy.io.wavfile.write(output_file.name, rate=24000, data=outputs["wav"])
                    print("🔊 使用英語語音合成完成")
                    return output_file.name
                except Exception as fallback_e:
                    print(f"❌ 備用語音合成也失敗: {fallback_e}")
                    return None
            else:
                print(f"❌ 語音合成錯誤: {e}")
                return None
    
    def start_real_time_translation(self):
        """開始即時翻譯模式"""
        if not self.is_voice_cloned:
            print("❌ 請先完成語音克隆步驟！")
            return
        
        print("\n🚀 即時翻譯模式")
        print("-" * 30)
        print(f"🌍 {self.supported_languages[self.source_language]} → {self.supported_languages[self.target_language]}")
        print("💡 系統將持續監聽您的語音並即時翻譯")
        print("⏹️ 按 Enter 停止即時翻譯")
        
        input("準備好後按 Enter 開始即時翻譯...")
        
        # 重置狀態
        self.is_real_time_active = True
        self.should_stop = False
        
        # 啟動工作線程
        threads = []
        
        # 音頻捕獲線程
        capture_thread = threading.Thread(target=self.audio_capture_worker)
        capture_thread.daemon = True
        capture_thread.start()
        threads.append(capture_thread)
        
        # 翻譯處理線程
        translation_thread = threading.Thread(target=self.translation_worker)
        translation_thread.daemon = True
        translation_thread.start()
        threads.append(translation_thread)
        
        # 音頻播放線程
        playback_thread = threading.Thread(target=self.playback_worker)
        playback_thread.daemon = True
        playback_thread.start()
        threads.append(playback_thread)
        
        print("\n🔥 即時翻譯已啟動！開始說話吧...")
        print("按 Enter 停止...")
        
        # 等待用戶停止
        try:
            input()
        except KeyboardInterrupt:
            pass
        
        print("\n⏹️ 正在停止即時翻譯...")
        
        # 停止所有線程
        self.should_stop = True
        self.is_real_time_active = False
        
        # 等待線程結束
        for thread in threads:
            thread.join(timeout=3)
        
        print("✅ 即時翻譯已停止")
    
    def run(self):
        """運行主程序"""
        print("🎤 歡迎使用即時語音克隆翻譯系統！")
        
        # 系統設置
        if not self.setup_system():
            print("❌ 系統設置失敗，程序退出")
            return
        
        print("\n" + "=" * 60)
        print("🎉 系統設置完成！")
        print("=" * 60)
        
        while True:
            print("\n📋 操作選項:")
            print("  c - 克隆語音（必須先完成）")
            print("  enter - 開始即時翻譯")
            print("  q - 退出程序")
            
            if self.is_voice_cloned:
                print("✅ 語音已克隆，可以開始即時翻譯")
            else:
                print("⚠️ 請先按 'c' 完成語音克隆")
            
            try:
                choice = input("\n請選擇操作: ").strip().lower()
                
                if choice == 'q':
                    print("👋 再見！")
                    break
                elif choice == 'c':
                    if self.clone_voice_step():
                        print("✅ 語音克隆完成，現在可以開始即時翻譯了！")
                    else:
                        print("❌ 語音克隆失敗，請重試")
                elif choice == '' or choice == 'enter':
                    self.start_real_time_translation()
                else:
                    print("❌ 無效選項，請重新選擇")
                    
            except KeyboardInterrupt:
                print("\n👋 程序被中斷，再見！")
                break
            except Exception as e:
                print(f"❌ 發生錯誤: {e}")

# 主程序入口
if __name__ == "__main__":
    system = RealTimeVoiceTranslationSystem()
    system.run()