from OpenVoice.checkpoints_v2.MeloTTS.melo.api import TTS
import os
import torch
from OpenVoice.openvoice import se_extractor
from OpenVoice.openvoice.api import ToneColorConverter
import nltk

nltk.download('averaged_perceptron_tagger_eng')



ckpt_converter = 'OpenVoice/checkpoints_v2/converter'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = 'outputs_v2'

tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

os.makedirs(output_dir, exist_ok=True)
reference_speaker = 'test.wav' # This is the voice you want to clone
target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=True)

texts = {
    #'EN_NEWEST': "Did you ever hear a folk tale about a giant turtle?",  # The newest English base speaker model
    #'ES': "El resplandor del sol acaricia las olas, pintando el cielo con una paleta deslumbrante.",
    #'FR': "La lueur dorée du soleil caresse les vagues, peignant le ciel d'une palette éblouissante.",
    #'ZH': "在这次vacation中，我们计划去Paris欣赏埃菲尔铁塔和卢浮宫的美景。",
    #'JP': "彼は毎朝ジョギングをして体を健康に保っています。",
    #'KR': "안녕하세요! 오늘은 날씨가 정말 좋네요.",
}


src_path = f'{output_dir}/tmp.wav'

# Speed is adjustable
speed = 1.0

for language, text in texts.items():
    model = TTS(language=language, device=device)
    speaker_ids = model.hps.data.spk2id
    
    for speaker_key in speaker_ids.keys():
        speaker_id = speaker_ids[speaker_key]
        speaker_key = speaker_key.lower().replace('_', '-')
        
        source_se = torch.load(f'OpenVoice/checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=device)
        if torch.backends.mps.is_available() and device == 'cpu':
            torch.backends.mps.is_available = lambda: False
        model.tts_to_file(text, speaker_id, src_path, speed=speed)
        save_path = f'{output_dir}/output_v2_{speaker_key}.wav'

        # Run the tone color converter
        encode_message = "@MyShell"
        tone_color_converter.convert(
            audio_src_path=src_path, 
            src_se=source_se, 
            tgt_se=target_se, 
            output_path=save_path,
            message=encode_message)