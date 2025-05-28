from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch

config = XttsConfig()
config.load_json("XTTS-v2/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="XTTS-v2/", eval=True)
# 确保使用GPU加速（如果可用）
if torch.cuda.is_available():
    model.cuda()

# 设置兼容性标志以解决generate方法缺失问题
# 降低transformers版本到v4.49.0可能是另一个解决方案
try:
    outputs = model.synthesize(
        "hello",
        config,
        speaker_wav="voice_output/output.wav",
        gpt_cond_len=3,
        language="en",
    )
    # 将音频保存到文件
    import scipy
    scipy.io.wavfile.write("output_speech.wav", rate=24000, data=outputs["wav"])
    print("语音合成成功，已保存到output_speech.wav")
except AttributeError as e:
    if "'GPT2InferenceModel' object has no attribute 'generate'" in str(e):
        print("错误：transformers库版本过高，请降级到4.49.0版本")
        print("请运行: pip install transformers==4.49.0")
    else:
        raise e
