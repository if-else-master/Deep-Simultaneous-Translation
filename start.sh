#!/bin/bash

echo "🚀 啟動即時語音克隆翻譯系統..."

# 設置 MeCab 配置文件路徑
if [ -f "/opt/homebrew/etc/mecabrc" ]; then
    export MECABRC="/opt/homebrew/etc/mecabrc"
    echo "✅ 已設置 MeCab 配置路徑: $MECABRC"
elif [ -f "/usr/local/etc/mecabrc" ]; then
    export MECABRC="/usr/local/etc/mecabrc"
    echo "✅ 已設置 MeCab 配置路徑: $MECABRC"
else
    echo "⚠️ 未找到 MeCab 配置文件，某些語言可能無法正常合成"
fi

# 啟動 Python 程序
python main.py 