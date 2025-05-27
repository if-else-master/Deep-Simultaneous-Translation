import os
import sounddevice as sd
import whisper
import google.generativeai as genai
from TTS.api import TTS
import numpy as np

# ==== 環境初始化 ====
os.environ["GOOGLE_API_KEY"] = "你的 Gemini API 金鑰"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# ==== 模型載入 ====
print("🔄 載入 Whisper 模型...")
asr_model = whisper.load_model("base")  # 可改成 "small"、"medium"、"faster-whisper"
print("🔄 載入 XTTS 模型...")
tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

# ==== 參數設定 ====
LANG_ASR = "zh"           # 語音辨識語言：中文
LANG_TRANSLATE = "English"  # 翻譯目標語言
LANG_TTS = "en"           # XTTS 輸出語言代碼
SPEAKER_SAMPLE = "your_voice_sample.wav"  # XTTS 語音樣本

# ==== 函式區 ====

def record_audio(duration=5, samplerate=16000):
    print("🎙️ 錄音中...（{} 秒）".format(duration))
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    print("✅ 錄音完成")
    return audio.flatten()

def transcribe_audio(audio):
    result = asr_model.transcribe(audio, language=LANG_ASR)
    return result['text']

def translate_text_with_gemini(text, target_language=LANG_TRANSLATE):
    model = genai.GenerativeModel("gemini-pro")
    prompt = f"請將下列內容翻譯成 {target_language}：\n{text}"
    response = model.generate_content(prompt)
    return response.text.strip()

def speak_text(text, lang=LANG_TTS, speaker_wav=SPEAKER_SAMPLE):
    print("🗣️ 語音合成中...")
    tts_model.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        language=lang,
        file_path="output.wav"
    )
    os.system("afplay output.wav")  # 對 macOS 使用者，Windows 可改用 playsound、winsound

# ==== 主流程 ====

def main_loop():
    while True:
        user_input = input("\n🔁 按下 Enter 開始錄音（或輸入 q 離開）: ")
        if user_input.lower() == 'q':
            print("👋 已結束")
            break

        audio = record_audio(duration=5)
        text = transcribe_audio(audio)
        print("📄 認出語音文字:", text)

        translated = translate_text_with_gemini(text)
        print("🌍 翻譯結果:", translated)

        speak_text(translated)

# ==== 執行程式 ====
if __name__ == "__main__":
    main_loop()
