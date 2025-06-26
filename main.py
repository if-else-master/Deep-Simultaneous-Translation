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
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import glob
from datetime import datetime

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

class VoiceTranslationGUI:
    def __init__(self):
        # 創建主窗口
        self.root = tk.Tk()
        self.root.title("🎤 即時語音克隆翻譯系統")
        self.root.geometry("1200x900")
        self.root.configure(bg='#2c3e50')
        self.root.minsize(800, 600)
        
        # 系統後端
        self.backend = RealTimeVoiceTranslationSystem()
        
        # GUI 狀態
        self.is_recording = False
        self.recording_animation_id = None
        
        # 創建界面
        self.create_widgets()
        
        # 設置按鈕懸停效果
        self.setup_hover_effects()
        
        # 載入已存在的語音文件
        self.load_existing_voices()
        
    def create_widgets(self):
        """創建所有GUI元件"""
        # 深色主題配色
        self.colors = {
            'bg_primary': '#2c3e50',      # 主背景 - 深藍灰
            'bg_secondary': '#34495e',     # 次要背景 - 藍灰
            'bg_card': '#3c4043',         # 卡片背景 - 深灰
            'text_primary': '#ecf0f1',    # 主文字 - 淺灰白
            'text_secondary': '#bdc3c7',  # 次要文字 - 灰色
            'accent_blue': '#3498db',     # 藍色強調
            'accent_green': '#27ae60',    # 綠色強調
            'accent_orange': '#f39c12',   # 橙色強調
            'accent_red': '#e74c3c',      # 紅色強調
            'input_bg': '#2c3e50',        # 輸入框背景
            'input_text': '#ecf0f1',      # 輸入框文字
            'button_bg': '#3c4043',       # 按鈕背景 - 深灰
            'button_hover': '#4a4a4a'     # 按鈕懸停 - 更深的灰
        }
        
        # 標題
        title_frame = tk.Frame(self.root, bg=self.colors['bg_primary'])
        title_frame.pack(pady=15)
        
        title_label = tk.Label(title_frame, text="🎤 即時語音克隆翻譯系統", 
                              font=('SF Pro Display', 24, 'bold'), 
                              bg=self.colors['bg_primary'], 
                              fg=self.colors['text_primary'])
        title_label.pack()
        
        subtitle_label = tk.Label(title_frame, text="Deep Simultaneous Translation", 
                                 font=('SF Pro Display', 12), 
                                 bg=self.colors['bg_primary'], 
                                 fg=self.colors['text_secondary'])
        subtitle_label.pack(pady=(5, 0))
        
        # API Key 設置區域
        api_frame = tk.LabelFrame(self.root, text="📡 API 設置", 
                                 font=('SF Pro Display', 14, 'bold'), 
                                 bg=self.colors['bg_secondary'], 
                                 fg=self.colors['text_primary'],
                                 bd=2, relief='groove',
                                 padx=15, pady=15)
        api_frame.pack(fill='x', padx=25, pady=8)
        
        tk.Label(api_frame, text="Gemini API Key:", 
                bg=self.colors['bg_secondary'], 
                fg=self.colors['text_primary'],
                font=('SF Pro Display', 11)).grid(row=0, column=0, sticky='w', pady=5)
        
        self.api_key_entry = tk.Entry(api_frame, width=55, show='*', 
                                     font=('SF Pro Display', 11),
                                     bg=self.colors['input_bg'], 
                                     fg=self.colors['input_text'],
                                     insertbackground=self.colors['text_primary'],
                                     bd=2, relief='solid')
        self.api_key_entry.grid(row=0, column=1, padx=10, sticky='w')
        
        self.test_api_btn = tk.Button(api_frame, text="✅ 測試 API", command=self.test_api,
                                     bg=self.colors['button_bg'], fg='black', 
                                     font=('SF Pro Display', 11, 'bold'),
                                     bd=0, relief='flat', cursor='hand2',
                                     padx=15, pady=8)
        self.test_api_btn.grid(row=0, column=2, padx=10)
        
        # 語言設置區域
        lang_frame = tk.LabelFrame(self.root, text="🌍 語言設置", 
                                  font=('SF Pro Display', 14, 'bold'), 
                                  bg=self.colors['bg_secondary'], 
                                  fg=self.colors['text_primary'],
                                  bd=2, relief='groove',
                                  padx=15, pady=15)
        lang_frame.pack(fill='x', padx=25, pady=8)
        
        # 語言選項
        languages = {
            'zh': '中文', 'en': '英文', 'ja': '日文', 'ko': '韓文',
            'es': '西班牙文', 'fr': '法文', 'de': '德文', 'it': '意大利文', 'pt': '葡萄牙文'
        }
        
        tk.Label(lang_frame, text="原始語言:", 
                bg=self.colors['bg_secondary'], 
                fg=self.colors['text_primary'],
                font=('SF Pro Display', 11)).grid(row=0, column=0, sticky='w', pady=5)
        
        self.source_lang_var = tk.StringVar(value='zh')
        
        # 創建自定義樣式的 Combobox
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Custom.TCombobox',
                       fieldbackground=self.colors['input_bg'],
                       background=self.colors['bg_secondary'],
                       foreground=self.colors['text_primary'],
                       arrowcolor=self.colors['text_primary'],
                       bordercolor=self.colors['accent_blue'],
                       lightcolor=self.colors['bg_secondary'],
                       darkcolor=self.colors['bg_secondary'])
        
        self.source_lang_combo = ttk.Combobox(lang_frame, textvariable=self.source_lang_var, 
                                             values=list(languages.keys()), state='readonly', 
                                             width=12, style='Custom.TCombobox',
                                             font=('SF Pro Display', 11))
        self.source_lang_combo.grid(row=0, column=1, padx=10)
        
        tk.Label(lang_frame, text="目標語言:", 
                bg=self.colors['bg_secondary'], 
                fg=self.colors['text_primary'],
                font=('SF Pro Display', 11)).grid(row=0, column=2, sticky='w', padx=(25,5))
        
        self.target_lang_var = tk.StringVar(value='en')
        self.target_lang_combo = ttk.Combobox(lang_frame, textvariable=self.target_lang_var, 
                                             values=list(languages.keys()), state='readonly', 
                                             width=12, style='Custom.TCombobox',
                                             font=('SF Pro Display', 11))
        self.target_lang_combo.grid(row=0, column=3, padx=10)
        
        # 語音克隆區域
        voice_frame = tk.LabelFrame(self.root, text="🎭 語音克隆", 
                                   font=('SF Pro Display', 14, 'bold'), 
                                   bg=self.colors['bg_secondary'], 
                                   fg=self.colors['text_primary'],
                                   bd=2, relief='groove',
                                   padx=15, pady=15)
        voice_frame.pack(fill='x', padx=25, pady=8)
        
        # 語音選擇
        tk.Label(voice_frame, text="選擇語音:", 
                bg=self.colors['bg_secondary'], 
                fg=self.colors['text_primary'],
                font=('SF Pro Display', 11)).grid(row=0, column=0, sticky='w', pady=5)
        
        self.voice_var = tk.StringVar()
        self.voice_combo = ttk.Combobox(voice_frame, textvariable=self.voice_var, 
                                       state='readonly', width=35, 
                                       style='Custom.TCombobox',
                                       font=('SF Pro Display', 11))
        self.voice_combo.grid(row=0, column=1, padx=10, sticky='w')
        
        # 語音克隆按鈕
        self.clone_voice_btn = tk.Button(voice_frame, text="🎤 錄製新語音", command=self.clone_voice,
                                        bg=self.colors['button_bg'], fg='black', 
                                        font=('SF Pro Display', 11, 'bold'),
                                        bd=0, relief='flat', cursor='hand2',
                                        padx=15, pady=8)
        self.clone_voice_btn.grid(row=0, column=2, padx=12)
        
        self.load_model_btn = tk.Button(voice_frame, text="🤖 載入模型", command=self.load_xtts_model,
                                       bg=self.colors['button_bg'], fg='black', 
                                       font=('SF Pro Display', 11, 'bold'),
                                       bd=0, relief='flat', cursor='hand2',
                                       padx=15, pady=8)
        self.load_model_btn.grid(row=0, column=3, padx=8)
        
        # 逐字稿顯示區域
        transcript_frame = tk.Frame(self.root, bg=self.colors['bg_primary'])
        transcript_frame.pack(fill='both', expand=True, padx=25, pady=15)
        
        # 原文逐字稿
        original_label_frame = tk.Frame(transcript_frame, bg=self.colors['bg_primary'])
        original_label_frame.pack(fill='x', pady=(0, 5))
        
        tk.Label(original_label_frame, text="📝 原文逐字稿", 
                bg=self.colors['bg_primary'], 
                fg=self.colors['text_primary'],
                font=('SF Pro Display', 14, 'bold')).pack(side='left')
        
        tk.Label(original_label_frame, text="Real-time Speech Recognition", 
                bg=self.colors['bg_primary'], 
                fg=self.colors['text_secondary'],
                font=('SF Pro Display', 10)).pack(side='right')
        
        self.original_text = scrolledtext.ScrolledText(transcript_frame, height=9, wrap=tk.WORD,
                                                      font=('SF Pro Display', 12), 
                                                      bg=self.colors['bg_card'], 
                                                      fg=self.colors['text_primary'],
                                                      insertbackground=self.colors['text_primary'],
                                                      selectbackground=self.colors['accent_blue'],
                                                      selectforeground='white',
                                                      bd=2, relief='solid')
        self.original_text.pack(fill='both', expand=True, pady=(0, 15))
        
        # 翻譯逐字稿
        translated_label_frame = tk.Frame(transcript_frame, bg=self.colors['bg_primary'])
        translated_label_frame.pack(fill='x', pady=(0, 5))
        
        tk.Label(translated_label_frame, text="🌍 翻譯逐字稿", 
                bg=self.colors['bg_primary'], 
                fg=self.colors['text_primary'],
                font=('SF Pro Display', 14, 'bold')).pack(side='left')
        
        tk.Label(translated_label_frame, text="Real-time Translation", 
                bg=self.colors['bg_primary'], 
                fg=self.colors['text_secondary'],
                font=('SF Pro Display', 10)).pack(side='right')
        
        self.translated_text = scrolledtext.ScrolledText(transcript_frame, height=9, wrap=tk.WORD,
                                                        font=('SF Pro Display', 12), 
                                                        bg=self.colors['bg_card'], 
                                                        fg=self.colors['accent_green'],
                                                        insertbackground=self.colors['accent_green'],
                                                        selectbackground=self.colors['accent_green'],
                                                        selectforeground='white',
                                                        bd=2, relief='solid')
        self.translated_text.pack(fill='both', expand=True, pady=(0, 15))
        
        # 控制區域
        control_frame = tk.Frame(self.root, bg=self.colors['bg_secondary'], relief='groove', bd=2)
        control_frame.pack(fill='x', padx=25, pady=(0, 20))
        
        # 內部控制框架
        inner_control = tk.Frame(control_frame, bg=self.colors['bg_secondary'])
        inner_control.pack(fill='x', padx=20, pady=15)
        
        # 語音輸出控制
        self.enable_voice_var = tk.BooleanVar(value=True)
        self.enable_voice_check = tk.Checkbutton(inner_control, text="🔊 啟用語音輸出", 
                                                variable=self.enable_voice_var, 
                                                bg=self.colors['bg_secondary'],
                                                fg=self.colors['text_primary'],
                                                selectcolor=self.colors['bg_card'],
                                                activebackground=self.colors['bg_secondary'],
                                                activeforeground=self.colors['text_primary'],
                                                font=('SF Pro Display', 12), 
                                                cursor='hand2')
        self.enable_voice_check.pack(side='left')
        
        # 狀態顯示
        self.status_label = tk.Label(inner_control, text="📋 系統就緒", 
                                    bg=self.colors['bg_secondary'], 
                                    fg=self.colors['text_secondary'],
                                    font=('SF Pro Display', 12))
        self.status_label.pack(side='left', padx=30)
        
        # 開始翻譯按鈕
        self.start_btn = tk.Button(inner_control, text="🚀 開始即時翻譯", command=self.toggle_translation,
                                  bg=self.colors['button_bg'], fg='black', 
                                  font=('SF Pro Display', 16, 'bold'),
                                  bd=0, relief='flat', cursor='hand2',
                                  padx=25, pady=12)
        self.start_btn.pack(side='right')
    
    def setup_hover_effects(self):
        """設置按鈕懸停效果"""
        def on_enter(event, original_bg, hover_bg):
            event.widget.config(bg=hover_bg)
        
        def on_leave(event, original_bg, hover_bg):
            event.widget.config(bg=original_bg)
        
        # API 測試按鈕懸停效果
        self.test_api_btn.bind("<Enter>", lambda e: on_enter(e, self.colors['button_bg'], self.colors['button_hover']))
        self.test_api_btn.bind("<Leave>", lambda e: on_leave(e, self.colors['button_bg'], self.colors['button_hover']))
        
        # 語音克隆按鈕懸停效果
        self.clone_voice_btn.bind("<Enter>", lambda e: on_enter(e, self.colors['button_bg'], self.colors['button_hover']))
        self.clone_voice_btn.bind("<Leave>", lambda e: on_leave(e, self.colors['button_bg'], self.colors['button_hover']))
        
        # 載入模型按鈕懸停效果
        self.load_model_btn.bind("<Enter>", lambda e: on_enter(e, self.colors['button_bg'], self.colors['button_hover']))
        self.load_model_btn.bind("<Leave>", lambda e: on_leave(e, self.colors['button_bg'], self.colors['button_hover']))
        
        # 開始翻譯按鈕懸停效果
        self.start_btn.bind("<Enter>", lambda e: on_enter(e, self.colors['button_bg'], self.colors['button_hover']))
        self.start_btn.bind("<Leave>", lambda e: on_leave(e, self.colors['button_bg'], self.colors['button_hover']))
        
    def load_existing_voices(self):
        """載入已存在的語音文件"""
        try:
            if not os.path.exists("cloned_voices"):
                os.makedirs("cloned_voices")
            
            voice_files = glob.glob("cloned_voices/*.wav")
            voice_names = [os.path.basename(f) for f in voice_files]
            
            if voice_names:
                self.voice_combo['values'] = voice_names
                self.voice_combo.set(voice_names[0])
                self.backend.cloned_voice_path = os.path.join("cloned_voices", voice_names[0])
                self.backend.is_voice_cloned = True
                self.update_status("✅ 已載入語音文件")
            else:
                self.voice_combo['values'] = ["無可用語音"]
                self.update_status("⚠️ 請先錄製語音")
                
        except Exception as e:
            self.update_status(f"❌ 載入語音失敗: {e}")
    
    def test_api(self):
        """測試 API Key"""
        api_key = self.api_key_entry.get().strip()
        if not api_key:
            messagebox.showerror("錯誤", "請輸入 API Key")
            return
        
        try:
            self.update_status("🔄 測試 API Key...")
            genai.configure(api_key=api_key)
            test_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            test_response = test_model.generate_content("測試")
            
            self.backend.gemini_api_key = api_key
            self.backend.model = test_model
            self.update_status("✅ API Key 有效！")
            messagebox.showinfo("成功", "API Key 測試成功！")
            
        except Exception as e:
            self.update_status("❌ API Key 無效")
            messagebox.showerror("錯誤", f"API Key 無效: {e}")
    
    def clone_voice(self):
        """語音克隆功能"""
        if not self.backend.gemini_api_key:
            messagebox.showerror("錯誤", "請先設置並測試 API Key")
            return
        
        # 創建錄音對話框
        self.show_recording_dialog()
    
    def show_recording_dialog(self):
        """顯示錄音對話框"""
        dialog = tk.Toplevel(self.root)
        dialog.title("🎤 語音錄製")
        dialog.geometry("480x280")
        dialog.configure(bg=self.colors['bg_primary'])
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)
        
        # 居中顯示
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        # 主標題
        tk.Label(dialog, text="🎭 語音克隆錄製", 
                font=('SF Pro Display', 18, 'bold'), 
                bg=self.colors['bg_primary'], 
                fg=self.colors['text_primary']).pack(pady=25)
        
        # 說明文字
        instruction = tk.Label(dialog, text="請用自然語調說一段話（建議3-5秒）\n錄音將在您停止說話2秒後自動結束", 
                              font=('SF Pro Display', 12), 
                              bg=self.colors['bg_primary'], 
                              fg=self.colors['text_secondary'], 
                              justify='center')
        instruction.pack(pady=15)
        
        # 錄音狀態顯示
        self.recording_status = tk.Label(dialog, text="準備錄音...", 
                                        font=('SF Pro Display', 14, 'bold'), 
                                        bg=self.colors['bg_primary'], 
                                        fg=self.colors['accent_orange'])
        self.recording_status.pack(pady=20)
        
        # 按鈕框架
        btn_frame = tk.Frame(dialog, bg=self.colors['bg_primary'])
        btn_frame.pack(pady=25)
        
        start_record_btn = tk.Button(btn_frame, text="🎤 開始錄音", 
                                    command=lambda: self.start_recording(dialog),
                                    bg=self.colors['button_bg'], fg='black', 
                                    font=('SF Pro Display', 12, 'bold'), 
                                    bd=0, relief='flat', cursor='hand2',
                                    padx=25, pady=10)
        start_record_btn.pack(side='left', padx=15)
        
        cancel_btn = tk.Button(btn_frame, text="❌ 取消", command=dialog.destroy,
                              bg=self.colors['button_bg'], fg='white', 
                              font=('SF Pro Display', 12, 'bold'),
                              bd=0, relief='flat', cursor='hand2',
                              padx=25, pady=10)
        cancel_btn.pack(side='left', padx=15)
    
    def start_recording(self, dialog):
        """開始錄音"""
        try:
            self.recording_status.config(text="🎤 正在錄音...", fg=self.colors['accent_red'])
            dialog.update()
            
            # 執行錄音
            result = self.backend.clone_voice_step_gui()
            
            if result:
                self.recording_status.config(text="✅ 錄音完成！", fg=self.colors['accent_green'])
                dialog.update()
                time.sleep(1)
                dialog.destroy()
                
                # 重新載入語音列表
                self.load_existing_voices()
                messagebox.showinfo("成功", "語音克隆完成！")
            else:
                self.recording_status.config(text="❌ 錄音失敗", fg=self.colors['accent_red'])
                
        except Exception as e:
            self.recording_status.config(text="❌ 錄音失敗", fg=self.colors['accent_red'])
            messagebox.showerror("錯誤", f"錄音失敗: {e}")
    
    def load_xtts_model(self):
        """載入 XTTS 模型"""
        try:
            self.update_status("🔄 載入 XTTS 模型...")
            self.load_model_btn.config(state='disabled', text="載入中...")
            
            # 在子線程中載入模型
            thread = threading.Thread(target=self._load_model_thread)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            self.update_status(f"❌ 載入模型失敗: {e}")
            self.load_model_btn.config(state='normal', text="🤖 載入模型", bg=self.colors['button_bg'])
    
    def _load_model_thread(self):
        """模型載入線程"""
        try:
            success = self.backend.load_xtts_model()
            
            # 更新 GUI（必須在主線程中）
            self.root.after(0, self._model_loaded_callback, success)
            
        except Exception as e:
            self.root.after(0, self._model_loaded_callback, False, str(e))
    
    def _model_loaded_callback(self, success, error=None):
        """模型載入完成回調"""
        if success:
            self.update_status("✅ XTTS 模型載入成功！")
            self.load_model_btn.config(state='normal', text="✅ 模型已載入", bg=self.colors['button_bg'])
        else:
            self.update_status(f"❌ 載入模型失敗: {error or '未知錯誤'}")
            self.load_model_btn.config(state='normal', text="🤖 載入模型", bg=self.colors['button_bg'])
    
    def toggle_translation(self):
        """切換翻譯狀態"""
        if not self.is_recording:
            self.start_translation()
        else:
            self.stop_translation()
    
    def start_translation(self):
        """開始即時翻譯"""
        # 檢查必要條件
        if not self.backend.gemini_api_key:
            messagebox.showerror("錯誤", "請先設置並測試 API Key")
            return
        
        if not self.backend.xtts_model:
            messagebox.showerror("錯誤", "請先載入 XTTS 模型")
            return
        
        if self.enable_voice_var.get() and not self.backend.is_voice_cloned:
            messagebox.showerror("錯誤", "語音輸出已啟用，請先錄製語音")
            return
        
        # 設置語言
        self.backend.source_language = self.source_lang_var.get()
        self.backend.target_language = self.target_lang_var.get()
        
        # 設置語音文件
        if self.enable_voice_var.get() and self.voice_var.get():
            self.backend.cloned_voice_path = os.path.join("cloned_voices", self.voice_var.get())
        
        # 清空逐字稿
        self.original_text.delete('1.0', tk.END)
        self.translated_text.delete('1.0', tk.END)
        
        # 開始錄音和翻譯
        self.is_recording = True
        self.start_btn.config(text="⏹️ 停止翻譯", bg=self.colors['button_bg'])
        self.start_recording_animation()
        self.update_status("🎤 正在監聽...")
        
        # 啟動後端翻譯系統
        self.backend.start_real_time_translation_gui(self)
    
    def stop_translation(self):
        """停止即時翻譯"""
        self.is_recording = False
        self.start_btn.config(text="🚀 開始即時翻譯", bg=self.colors['button_bg'])
        self.stop_recording_animation()
        self.update_status("⏹️ 翻譯已停止")
        
        # 停止後端翻譯系統
        self.backend.stop_real_time_translation()
    
    def start_recording_animation(self):
        """開始錄音動畫效果"""
        def animate():
            if self.is_recording:
                current_bg = self.start_btn.cget('bg')
                if current_bg == self.colors['button_bg']:
                    self.start_btn.config(bg=self.colors['button_hover'], relief='raised')
                else:
                    self.start_btn.config(bg=self.colors['button_bg'], relief='sunken')
                self.recording_animation_id = self.root.after(500, animate)
        animate()
    
    def stop_recording_animation(self):
        """停止錄音動畫效果"""
        if self.recording_animation_id:
            self.root.after_cancel(self.recording_animation_id)
            self.recording_animation_id = None
        self.start_btn.config(relief='raised')
    
    def update_status(self, message):
        """更新狀態顯示"""
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def add_original_text(self, text):
        """添加原文到逐字稿"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.original_text.insert(tk.END, f"[{timestamp}] {text}\n")
        self.original_text.see(tk.END)
    
    def add_translated_text(self, text):
        """添加翻譯到逐字稿"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.translated_text.insert(tk.END, f"[{timestamp}] {text}\n")
        self.translated_text.see(tk.END)
    
    def run(self):
        """運行 GUI"""
        # 設置關閉事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """關閉應用程序"""
        if self.is_recording:
            self.stop_translation()
        self.root.destroy()

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
        
        # GUI 引用
        self.gui = None
        
        # 初始化pygame
        pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=1024)
        
        print("🎤 即時語音克隆翻譯系統已啟動！")
    
    def clone_voice_step_gui(self):
        """GUI 版本的語音克隆"""
        try:
            if not os.path.exists("cloned_voices"):
                os.makedirs("cloned_voices")
            
            # 錄音
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
                        pass  # 繼續錄音
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
            
            # 保存語音克隆樣本
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
    
    def load_xtts_model(self):
        """載入XTTS模型"""
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
                if self.gui:
                    self.gui.root.after(0, lambda: self.gui.update_status("🎤 檢測到語音..."))
            self.last_speech_time = current_time
            return True
        else:
            if self.is_speech_detected:
                silence_duration = current_time - self.last_speech_time
                if silence_duration >= self.silence_duration:
                    self.is_speech_detected = False
                    if self.gui:
                        self.gui.root.after(0, lambda: self.gui.update_status("🔄 處理語音..."))
                    return False
            return self.is_speech_detected
    
    def start_real_time_translation_gui(self, gui):
        """GUI 版本的即時翻譯"""
        self.gui = gui
        
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
        translation_thread = threading.Thread(target=self.translation_worker_gui)
        translation_thread.daemon = True
        translation_thread.start()
        threads.append(translation_thread)
        
        # 音頻播放線程（如果啟用）
        if gui.enable_voice_var.get():
            playback_thread = threading.Thread(target=self.playback_worker)
            playback_thread.daemon = True
            playback_thread.start()
            threads.append(playback_thread)
        
        self.threads = threads
    
    def stop_real_time_translation(self):
        """停止即時翻譯"""
        self.should_stop = True
        self.is_real_time_active = False
        
        # 等待線程結束
        if hasattr(self, 'threads'):
            for thread in self.threads:
                thread.join(timeout=3)
    
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
    
    def translation_worker_gui(self):
        """GUI 版本的翻譯處理工作線程"""
        print("🔄 翻譯處理線程已啟動")
        
        while self.is_real_time_active or not self.audio_segments_queue.empty():
            try:
                audio_segment = self.audio_segments_queue.get(timeout=1)
                
                # 保存音頻段到臨時文件
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                scipy.io.wavfile.write(temp_file.name, self.rate, audio_segment)
                
                # 翻譯
                original_text, translated_text = self.transcribe_and_translate_gui(temp_file.name)
                
                if original_text and original_text.strip():
                    # 更新 GUI 逐字稿
                    if self.gui:
                        self.gui.root.after(0, lambda t=original_text: self.gui.add_original_text(t))
                
                if translated_text and translated_text.strip():
                    # 更新 GUI 翻譯
                    if self.gui:
                        self.gui.root.after(0, lambda t=translated_text: self.gui.add_translated_text(t))
                    
                    # 語音合成（如果啟用）
                    if self.gui and self.gui.enable_voice_var.get():
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
    
    def transcribe_and_translate_gui(self, audio_file_path):
        """GUI 版本的語音轉文字和翻譯"""
        try:
            audio_file = genai.upload_file(path=audio_file_path)

            source_lang_name = self.supported_languages[self.source_language]
            target_lang_name = self.supported_languages[self.target_language]
            
            if self.source_language == self.target_language:
                prompt = f"請將這段音頻中的{source_lang_name}語音內容轉換為文字。只回傳轉錄的文字內容，不要包含其他說明。"
                response = self.model.generate_content([audio_file, prompt])
                original_text = response.text.strip()
                translated_text = original_text
            else:
                transcribe_prompt = f"請將這段音頻中的{source_lang_name}語音內容轉換為文字。只回傳轉錄的文字內容，不要包含其他說明。"
                transcribe_response = self.model.generate_content([audio_file, transcribe_prompt])
                original_text = transcribe_response.text.strip()

                translate_prompt = f"請將以下{source_lang_name}文字翻譯為{target_lang_name}。只回傳翻譯後的文字內容，不要包含其他說明。\n\n{original_text}"
                translate_response = self.model.generate_content(translate_prompt)
                translated_text = translate_response.text.strip()
            
            genai.delete_file(audio_file.name)
            
            print(f"📝 原文: {original_text}")
            print(f"🌍 翻譯: {translated_text}")
            return original_text, translated_text
            
        except Exception as e:
            print(f"❌ 翻譯錯誤: {e}")
            return None, None
    
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
    # 檢查是否使用 GUI 模式
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        # 命令行模式
        system = RealTimeVoiceTranslationSystem()
        system.run()
    else:
        # GUI 模式（默認）
        app = VoiceTranslationGUI()
        app.run()