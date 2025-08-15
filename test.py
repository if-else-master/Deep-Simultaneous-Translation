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

class VoiceTranslationSystem:
    def __init__(self, gemini_api_key):
        # 初始化 Gemini API
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # 初始化 XTTS 模型
        print("正在載入 XTTS 模型...")
        config = XttsConfig()
        config.load_json("XTTS-v2/config.json")
        self.xtts_model = Xtts.init_from_config(config)
        self.xtts_model.load_checkpoint(config, checkpoint_dir="XTTS-v2/", eval=True)
        
        if torch.cuda.is_available():
            self.xtts_model.cuda()
            print("使用 GPU 加速")
        
        self.config = config
        
        # 音頻參數
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000  # 降低採樣率以提高處理效率
        
        # 初始化 pygame 用於播放音頻
        pygame.mixer.init()
        
        # 持續錄音控制
        self.is_continuous_recording = False
        self.should_stop = False
        self.audio_queue = queue.Queue()
        self.processing_queue = queue.Queue()
        
        # 語音活動檢測參數
        self.silence_threshold = 500  # 音量閾值
        self.silence_duration = 1.5   # 靜音持續時間（秒）
        self.min_speech_duration = 0.5  # 最小語音持續時間（秒）
        
        # 語音克隆相關
        self.cloned_voice_path = None
        self.is_voice_cloned = False
        
        # 音頻緩衝區
        self.audio_buffer = deque(maxlen=int(self.rate * 10))  # 10秒緩衝區
        self.current_segment = []
        self.last_speech_time = 0
        self.is_speech_detected = False
        
        print("系統初始化完成！")
    
    def calculate_rms(self, audio_data):
        """計算音頻的RMS值用於語音活動檢測"""
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        return np.sqrt(np.mean(audio_np**2))
    
    def detect_voice_activity(self, audio_data):
        """語音活動檢測"""
        rms = self.calculate_rms(audio_data)
        current_time = time.time()
        
        if rms > self.silence_threshold:
            # 檢測到語音
            if not self.is_speech_detected:
                self.is_speech_detected = True
                print("🎤 檢測到語音開始")
            self.last_speech_time = current_time
            return True
        else:
            # 靜音
            if self.is_speech_detected:
                silence_duration = current_time - self.last_speech_time
                if silence_duration >= self.silence_duration:
                    # 靜音持續時間足夠，認為語音結束
                    self.is_speech_detected = False
                    return False
            return self.is_speech_detected
    
    def continuous_audio_capture(self):
        """持續音頻捕獲線程"""
        audio = pyaudio.PyAudio()
        stream = audio.open(format=self.format,
                          channels=self.channels,
                          rate=self.rate,
                          input=True,
                          frames_per_buffer=self.chunk)
        
        print("🎤 開始持續語音監聽...")
        
        while self.is_continuous_recording:
            try:
                data = stream.read(self.chunk, exception_on_overflow=False)
                self.audio_buffer.extend(np.frombuffer(data, dtype=np.int16))
                
                # 語音活動檢測
                is_speech = self.detect_voice_activity(data)
                
                if is_speech:
                    # 正在說話，收集音頻
                    self.current_segment.extend(np.frombuffer(data, dtype=np.int16))
                elif self.current_segment and len(self.current_segment) > int(self.rate * self.min_speech_duration):
                    # 語音結束且長度足夠，發送處理
                    print("📝 檢測到語音結束，發送處理...")
                    segment_audio = np.array(self.current_segment, dtype=np.int16)
                    self.processing_queue.put(segment_audio)
                    self.current_segment = []
                
                # 檢查是否需要停止
                if self.should_stop:
                    break
                    
            except Exception as e:
                print(f"❌ 音頻捕獲錯誤: {e}")
                break
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        print("🎤 音頻捕獲已停止")
    
    def save_audio_segment(self, audio_data, suffix='_segment'):
        """將音頻段保存到臨時文件"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'{suffix}.wav')
        
        # 確保音頻數據是正確的格式
        if isinstance(audio_data, np.ndarray):
            audio_data = audio_data.astype(np.int16)
        
        scipy.io.wavfile.write(temp_file.name, self.rate, audio_data)
        return temp_file.name
    
    def clone_voice_from_segment(self, audio_file_path):
        """從音頻段克隆語音"""
        try:
            if self.is_voice_cloned:
                return self.cloned_voice_path
            
            print("🎭 正在進行首次語音克隆...")
            
            # 創建克隆語音存儲目錄
            if not os.path.exists("cloned_voices"):
                os.makedirs("cloned_voices")
            
            # 將音頻段複製作為語音克隆參考
            cloned_voice_filename = f"cloned_voices/cloned_voice_{int(time.time())}.wav"
            shutil.copy2(audio_file_path, cloned_voice_filename)
            
            self.cloned_voice_path = cloned_voice_filename
            self.is_voice_cloned = True
            print(f"✅ 語音克隆完成，參考文件: {cloned_voice_filename}")
            
            return cloned_voice_filename
            
        except Exception as e:
            print(f"❌ 語音克隆過程發生錯誤: {e}")
            return None
    
    def transcribe_and_translate(self, audio_file_path, target_language="en"):
        """使用 Gemini 進行語音轉文字和翻譯"""
        try:
            print("🤖 正在進行語音識別和翻譯...")
            
            # 上傳音頻文件
            audio_file = genai.upload_file(path=audio_file_path)
            
            # 根據目標語言設定提示詞
            if target_language.lower() == "en":
                prompt = "請將這段音頻中的語音內容轉換為英文文字，如果原本就是英文就直接轉錄，如果是其他語言請翻譯成英文。只回傳最終的英文文字內容，不要包含其他說明。"
            elif target_language.lower() == "zh":
                prompt = "請將這段音頻中的語音內容轉換為繁體中文文字，如果原本就是中文就直接轉錄，如果是其他語言請翻譯成繁體中文。只回傳最終的繁體中文文字內容，不要包含其他說明。"
            else:
                prompt = f"請將這段音頻中的語音內容轉換為{target_language}文字，如果需要翻譯請翻譯成{target_language}。只回傳最終的文字內容，不要包含其他說明。"
            
            # 發送請求到 Gemini
            response = self.model.generate_content([audio_file, prompt])
            
            # 清理上傳的文件
            genai.delete_file(audio_file.name)
            
            translated_text = response.text.strip()
            print(f"📝 翻譯結果: {translated_text}")
            
            return translated_text
            
        except Exception as e:
            print(f"❌ 翻譯過程發生錯誤: {e}")
            return None
    
    def synthesize_speech_with_cloned_voice(self, text, output_file=None):
        """使用克隆的語音合成語音"""
        try:
            if not output_file:
                output_file = f"output_speech_{int(time.time())}.wav"
            
            print("🔊 正在使用克隆語音合成語音...")
            
            if not self.cloned_voice_path or not os.path.exists(self.cloned_voice_path):
                print("❌ 沒有可用的克隆語音")
                return None
            
            # 檢測語言
            has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
            language = "zh-cn" if has_chinese else "en"
            
            print(f"🌐 檢測到語言: {'中文' if language == 'zh-cn' else '英文'}")
            
            outputs = self.xtts_model.synthesize(
                text,
                self.config,
                speaker_wav=self.cloned_voice_path,
                gpt_cond_len=3,
                language=language,
            )
            
            # 保存音頻
            scipy.io.wavfile.write(output_file, rate=24000, data=outputs["wav"])
            print(f"✅ 語音合成完成")
            
            return output_file
            
        except AttributeError as e:
            if "'GPT2InferenceModel' object has no attribute 'generate'" in str(e):
                print("❌ 錯誤：transformers庫版本過高，請降級到4.49.0版本")
                print("請運行: pip install transformers==4.49.0")
                return None
            else:
                raise e
        except Exception as e:
            print(f"❌ 語音合成發生錯誤: {e}")
            return None
    
    def play_audio(self, audio_file):
        """播放音頻文件"""
        try:
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            # 等待播放完成
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
        except Exception as e:
            print(f"❌ 播放音頻時發生錯誤: {e}")
    
    def process_audio_segment(self, audio_segment, target_language="en"):
        """處理單個音頻段"""
        temp_audio_file = None
        output_file = None
        
        try:
            # 保存音頻段到臨時文件
            temp_audio_file = self.save_audio_segment(audio_segment)
            
            # 如果還沒有克隆語音，先進行克隆
            if not self.is_voice_cloned:
                self.clone_voice_from_segment(temp_audio_file)
            
            # 翻譯音頻內容
            translated_text = self.transcribe_and_translate(temp_audio_file, target_language)
            
            if translated_text and translated_text.strip():
                # 使用克隆語音合成翻譯後的內容
                output_file = self.synthesize_speech_with_cloned_voice(translated_text)
                
                if output_file:
                    # 播放結果
                    self.play_audio(output_file)
                    print("🎉 語音片段處理完成\n" + "="*50)
            else:
                print("⚠️ 沒有檢測到有效的語音內容")
            
        except Exception as e:
            print(f"❌ 處理音頻段時發生錯誤: {e}")
        
        finally:
            # 清理臨時文件
            for file_path in [temp_audio_file, output_file]:
                if file_path and os.path.exists(file_path):
                    try:
                        os.unlink(file_path)
                    except:
                        pass
    
    def audio_processing_worker(self, target_language="en"):
        """音頻處理工作線程"""
        print("🔄 音頻處理線程已啟動")
        
        while self.is_continuous_recording or not self.processing_queue.empty():
            try:
                # 獲取待處理的音頻段
                audio_segment = self.processing_queue.get(timeout=1)
                print(f"\n{'='*50}")
                print("🎯 開始處理新的語音片段...")
                
                # 處理音頻段
                self.process_audio_segment(audio_segment, target_language)
                
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ 處理線程錯誤: {e}")
        
        print("🔄 音頻處理線程已停止")
    
    def start_continuous_translation(self, target_language="en"):
        """開始持續翻譯模式"""
        print("🚀 啟動深度同步翻譯模式！")
        print("📢 系統將持續監聽你的語音並實時翻譯")
        print("💡 說話後停頓1.5秒，系統會自動處理並播放翻譯")
        print("⏹️ 輸入 'stop' 結束翻譯模式")
        print(f"🌐 目標語言: {'英文' if target_language == 'en' else '中文'}")
        
        # 等待用戶按 Enter 開始
        input("\n按 Enter 開始深度同步翻譯...")
        
        # 重置狀態
        self.should_stop = False
        self.is_continuous_recording = True
        self.is_voice_cloned = False
        self.cloned_voice_path = None
        
        # 啟動音頻捕獲線程
        capture_thread = threading.Thread(target=self.continuous_audio_capture)
        capture_thread.daemon = True
        capture_thread.start()
        
        # 啟動音頻處理線程
        processing_thread = threading.Thread(target=self.audio_processing_worker, args=(target_language,))
        processing_thread.daemon = True
        processing_thread.start()
        
        print("\n🔥 深度同步翻譯已啟動！開始說話吧...")
        
        # 等待用戶輸入 stop
        while True:
            try:
                user_input = input().strip().lower()
                if user_input == 'stop':
                    print("\n⏹️ 正在停止深度同步翻譯...")
                    break
            except KeyboardInterrupt:
                print("\n⏹️ 用戶中斷，正在停止...")
                break
        
        # 停止所有線程
        self.should_stop = True
        self.is_continuous_recording = False
        
        # 等待線程結束
        capture_thread.join(timeout=2)
        processing_thread.join(timeout=5)
        
        print("✅ 深度同步翻譯已停止")
    
    def run_translation_loop(self, target_language="en"):
        """運行翻譯循環"""
        print("🚀 語音克隆翻譯系統啟動！")
        print("\n=== 系統模式 ===")
        print("1. 🔥 深度同步翻譯 - 持續監聽並實時翻譯 (推薦)")
        print("2. 🎤 傳統模式 - 手動錄音翻譯")
        
        print("\n=== 指令說明 ===")
        print("- 輸入 'continuous' 或 'c' 開始深度同步翻譯")
        print("- 輸入 'start' 或 's' 開始傳統模式")
        print("- 輸入 'lang en' 設定翻譯為英文")
        print("- 輸入 'lang zh' 設定翻譯為中文")
        print("- 輸入 'clean' 清理所有克隆語音")
        print("- 輸入 'quit' 或 'q' 退出程式")
        
        print(f"\n目前翻譯語言: {'英文' if target_language == 'en' else '中文'}")
        
        while True:
            try:
                command = input("\n請輸入指令: ").strip().lower()
                
                if command in ['quit', 'q']:
                    print("👋 再見！")
                    break
                
                elif command.startswith('lang '):
                    new_lang = command.split(' ')[1]
                    if new_lang in ['en', 'zh']:
                        target_language = new_lang
                        print(f"✅ 翻譯語言已設定為: {'英文' if target_language == 'en' else '中文'}")
                    else:
                        print("❌ 支援的語言: en (英文), zh (中文)")
                
                elif command == 'clean':
                    self.clean_cloned_voices()
                
                elif command in ['continuous', 'c']:
                    self.start_continuous_translation(target_language)
                
                elif command in ['start', 's']:
                    print("\n🎬 開始傳統語音翻譯流程...")
                    success = self.process_voice_translation(target_language)
                    
                    if success:
                        print("\n🎉 語音翻譯流程完成！")
                    else:
                        print("\n❌ 流程執行失敗，請重試")
                
                else:
                    print("❌ 未知指令，請輸入 'continuous' 開始深度同步翻譯或 'quit' 退出")
                    
            except KeyboardInterrupt:
                print("\n👋 程式被中斷，再見！")
                break
            except Exception as e:
                print(f"❌ 發生錯誤: {e}")

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
        
        print("📝 錄音結束")
        stream.stop_stream()
        stream.close()
        audio.terminate()
    
    def stop_recording(self):
        """停止錄音"""
        self.is_recording = False
    
    def save_audio_to_temp(self, suffix='_original'):
        """將錄音保存到臨時文件"""
        if not self.audio_frames:
            return None
        
        # 創建臨時文件
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'{suffix}.wav')
        
        audio = pyaudio.PyAudio()
        wf = wave.open(temp_file.name, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(audio.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.audio_frames))
        wf.close()
        audio.terminate()
        
        return temp_file.name
    
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
    
    def process_voice_translation(self, target_language="en"):
        """完整的語音翻譯流程：錄音 → 克隆語音 → 翻譯 → 合成"""
        # 初始化錄音變量
        self.is_recording = True
        self.audio_frames = []
        
        # 開始錄音
        recording_thread = threading.Thread(target=self.start_recording)
        recording_thread.start()
        
        # 等待用戶按 Enter 停止錄音
        input()
        self.stop_recording()
        recording_thread.join()
        
        # 保存錄音到臨時文件
        temp_audio_file = self.save_audio_to_temp()
        
        if not temp_audio_file:
            print("❌ 沒有錄音數據")
            return False
        
        try:
            # 步驟1: 克隆語音
            print("\n=== 步驟 1: 語音克隆 ===")
            cloned_voice_file = self.clone_voice(temp_audio_file)
            
            if not cloned_voice_file:
                return False
            
            # 步驟2: 翻譯音頻內容
            print("\n=== 步驟 2: 語音翻譯 ===")
            translated_text = self.transcribe_and_translate(temp_audio_file, target_language)
            
            if not translated_text:
                return False
            
            # 步驟3: 使用克隆語音合成翻譯後的內容
            print("\n=== 步驟 3: 語音合成 ===")
            output_file = self.synthesize_speech_with_cloned_voice(translated_text)
            
            if not output_file:
                return False
            
            # 步驟4: 播放結果
            print("\n=== 步驟 4: 播放結果 ===")
            self.play_audio(output_file)
            
            return True
            
        finally:
            # 清理臨時文件
            try:
                if temp_audio_file and os.path.exists(temp_audio_file):
                    os.unlink(temp_audio_file)
            except:
                pass
    
    def show_cloned_voices(self):
        """顯示已克隆的語音文件"""
        if not os.path.exists("cloned_voices"):
            print("📁 還沒有克隆的語音文件")
            return
        
        voices = [f for f in os.listdir("cloned_voices") if f.endswith('.wav')]
        if not voices:
            print("📁 還沒有克隆的語音文件")
            return
        
        print("📁 已克隆的語音文件:")
        for i, voice in enumerate(voices, 1):
            print(f"  {i}. {voice}")
    
    def clean_cloned_voices(self):
        """清理所有克隆的語音文件"""
        if os.path.exists("cloned_voices"):
            try:
                shutil.rmtree("cloned_voices")
                print("🗑️ 已清理所有克隆的語音文件")
                self.cloned_voice_path = None
                self.is_voice_cloned = False
            except Exception as e:
                print(f"❌ 清理失敗: {e}")
        else:
            print("📁 沒有需要清理的文件")

# 使用範例
if __name__ == "__main__":
    # 請替換為你的 Gemini API Key
    GEMINI_API_KEY = "########################"
    
    # 初始化系統（不需要預設的參考語音，因為會用克隆的語音）
    system = VoiceTranslationSystem(gemini_api_key=GEMINI_API_KEY)
    
    # 運行翻譯循環，預設翻譯為英文
    system.run_translation_loop(target_language="en")