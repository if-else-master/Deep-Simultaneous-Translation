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
        self.rate = 44100
        self.record_seconds = 10  # 最大錄音時間
        
        # 初始化 pygame 用於播放音頻
        pygame.mixer.init()
        
        # 錄音控制
        self.is_recording = False
        self.audio_frames = []
        
        # 語音克隆相關
        self.cloned_voice_path = None
        
        print("系統初始化完成！")
    
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
    
    def synthesize_speech_with_cloned_voice(self, text, output_file="output_speech.wav"):
        """使用克隆的語音合成語音"""
        try:
            print("🔊 正在使用克隆語音合成語音...")
            
            if not self.cloned_voice_path or not os.path.exists(self.cloned_voice_path):
                print("❌ 沒有可用的克隆語音，請先錄音進行語音克隆")
                return None
            
            # 檢測語言
            # 簡單的語言檢測：如果包含中文字符就用中文，否則用英文
            has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
            language = "zh-cn" if has_chinese else "en"
            
            print(f"🌐 檢測到語言: {'中文' if language == 'zh-cn' else '英文'}")
            
            outputs = self.xtts_model.synthesize(
                text,
                self.config,
                speaker_wav=self.cloned_voice_path,  # 使用克隆的語音
                gpt_cond_len=3,
                language=language,
            )
            
            # 保存音頻
            scipy.io.wavfile.write(output_file, rate=24000, data=outputs["wav"])
            print(f"✅ 語音合成完成，已保存到 {output_file}")
            
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
            print("🔊 正在播放音頻...")
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            # 等待播放完成
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
            print("✅ 播放完成")
            
        except Exception as e:
            print(f"❌ 播放音頻時發生錯誤: {e}")
    
    def process_voice_translation(self, target_language="en"):
        """完整的語音翻譯流程：錄音 → 克隆語音 → 翻譯 → 合成"""
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
            except Exception as e:
                print(f"❌ 清理失敗: {e}")
        else:
            print("📁 沒有需要清理的文件")
    
    def run_translation_loop(self, target_language="en"):
        """運行翻譯循環"""
        print("🚀 語音克隆翻譯系統啟動！")
        print("\n=== 系統流程 ===")
        print("1. 🎤 錄音 - 錄製你的語音")
        print("2. 🎭 克隆 - 克隆你的音色")
        print("3. 🤖 翻譯 - 翻譯語音內容")
        print("4. 🔊 合成 - 用你的音色說出翻譯結果")
        
        print("\n=== 指令說明 ===")
        print("- 輸入 'start' 或 's' 開始完整流程")
        print("- 輸入 'lang en' 設定翻譯為英文")
        print("- 輸入 'lang zh' 設定翻譯為中文")
        print("- 輸入 'voices' 查看已克隆的語音")
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
                
                elif command == 'voices':
                    self.show_cloned_voices()
                
                elif command == 'clean':
                    self.clean_cloned_voices()
                
                elif command in ['start', 's']:
                    print("\n🎬 開始語音克隆翻譯流程...")
                    print("📝 準備錄音，錄音期間請清楚說話，錄音將用於語音克隆和翻譯")
                    
                    success = self.process_voice_translation(target_language)
                    
                    if success:
                        print("\n🎉 語音克隆翻譯流程完成！")
                    else:
                        print("\n❌ 流程執行失敗，請重試")
                
                else:
                    print("❌ 未知指令，請輸入 'start' 開始流程或 'quit' 退出")
                    
            except KeyboardInterrupt:
                print("\n👋 程式被中斷，再見！")
                break
            except Exception as e:
                print(f"❌ 發生錯誤: {e}")

# 使用範例
if __name__ == "__main__":
    # 請替換為你的 Gemini API Key
    GEMINI_API_KEY = "XXXX"
    
    # 初始化系統（不需要預設的參考語音，因為會用克隆的語音）
    system = VoiceTranslationSystem(gemini_api_key=GEMINI_API_KEY)
    
    # 運行翻譯循環，預設翻譯為英文
    system.run_translation_loop(target_language="en")