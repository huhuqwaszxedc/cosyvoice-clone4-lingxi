# -*- coding: utf-8 -*-


import argparse
import os, glob
import sys
sys.path.append('./')
import pdb
import soundfile as sf
import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import time
import librosa
import json
from jttts.config import GlobalConfig
GlobalConfigInst = GlobalConfig()



def cer(r: list, h: list):
    """
    Calculation of CER with Levenshtein distance.
    """
    # initialisation
    import numpy
    d = numpy.zeros((len(r) + 1) * (len(h) + 1), dtype=numpy.uint16)
    d = d.reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i
    # computation
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    return d[len(r)][len(h)] / float(len(r))




def get_dir_all_files(path, extension='.wav') :
    filenames = []
    for filename in glob.iglob(f'{path}/**/*{extension}', recursive=True):
        filenames += [filename]
    return filenames


def parse_and_config():
    """Parse arguments and set configuration parameters."""
    parser = argparse.ArgumentParser(
        description="Preprocess the format of rootdir as ljspeech "
    )
    parser.add_argument(
        '-i',"--input_dir",
        type=str,
        required=True,
        help="input_dir.",
    )
    parser.add_argument(
        '-o',"--output_dir",
        type=str,
        required=True,
        help="output_dir.",
    )
    args = parser.parse_args()
    return args






class ASR_Model():
    def __init__(self):

        asr_model = 'speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
        if os.path.exists(os.path.join(GlobalConfigInst.container_root_dir, './pretrained_models_cosv', asr_model)):
            self.asr_model_dir = os.path.join(GlobalConfigInst.container_root_dir, './pretrained_models_cosv', asr_model)
        else:
            self.asr_model_dir = os.path.join('/pretrained_models_cosv', asr_model)

        vad_model = 'speech_fsmn_vad_zh-cn-16k-common-pytorch'
        if os.path.exists(os.path.join(GlobalConfigInst.container_root_dir, './pretrained_models_cosv', vad_model)):
            self.vad_model_dir = os.path.join(GlobalConfigInst.container_root_dir, './pretrained_models_cosv', vad_model)
        else:
            self.vad_model_dir = os.path.join('/pretrained_models_cosv', vad_model)

        punc_model = 'punc_ct-transformer_zh-cn-common-vocab272727-pytorch'
        if os.path.exists(os.path.join(GlobalConfigInst.container_root_dir, './pretrained_models_cosv', punc_model)):
            self.punc_model_dir = os.path.join(GlobalConfigInst.container_root_dir, './pretrained_models_cosv', punc_model)
        else:
            self.punc_model_dir = os.path.join('/pretrained_models_cosv', punc_model)


        self.asr_model = pipeline(
            task=Tasks.auto_speech_recognition,
            model=self.asr_model_dir, model_revision="v2.0.4",
            vad_model=self.vad_model_dir, vad_model_revision="v2.0.4",
            punc_model=self.punc_model_dir, punc_model_revision="v2.0.4",
            # spk_model="iic/speech_campplus_sv_zh-cn_16k-common",
            # spk_model_revision="v2.0.2",
            disable_update=True
        )

    def forward(self, audio, sample_rate):
        if sample_rate != 16000:
            audio_new = librosa.resample(y=audio, orig_sr=sample_rate, target_sr=16000)
            sample_rate_new = 16000
        else:
            audio_new       = audio
            sample_rate_new = sample_rate
        text = self.asr_model(audio_new, fs=sample_rate_new, is_final=True)[0]["text"]
        return text
    def inference(self, audio, sample_rate):
        return self.forward(audio, sample_rate)

prompt_text_list = ['海洋中的每一个角落都有其独特的美丽和价值,它是地球上最宝贵的资源之一.',     # 0
                    '人生就像一场马拉松比赛,不是速度最快的人获胜,而是坚持到最后的人.',          # 1
                    '无论你遇到什么困难和挑战,都要保持信念和勇气,不断前行.',                    # 2
                    '在大自然的怀抱中,我们可以感受到生命的力量和美好,让我们的心灵得到升华.',     # 3
                    '大自然是一个神奇的世界,充满了各种美丽的景色.',                             # 4
                    '美国对港澳政策不会改变.',                                                  # 5 tts042
                ] 

def main():
    args = parse_and_config()
    input_dir       = args.input_dir
    output_dir      = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    wave_dir = os.path.join(output_dir, 'prompt_wavs')
    os.makedirs(wave_dir, exist_ok=True)
    ASR_Model_Inst = ASR_Model()

    all_wave_files = get_dir_all_files(input_dir)
    error_list = []
    for idx, subfile_path in enumerate(all_wave_files):
        
        utt_id = subfile_path.split('/')[-1][:-len('.wav')]
        audio_data, sample_rate = sf.read(subfile_path)
        audio_text = ASR_Model_Inst.inference(audio_data, sample_rate)

        selected_text = ''
        selected_cer = 1.0
        selected_idx = 0
        for idy, sub_promot_text in enumerate(prompt_text_list):
            sub_cer = cer(sub_promot_text, audio_text)
            if sub_cer < selected_cer:
                selected_cer = sub_cer
                selected_text = sub_promot_text
                selected_idx  = idy
        
        if selected_cer < 0.3:
            os.system(f'cp -r {subfile_path} {wave_dir}')

            json_file = os.path.join(output_dir, utt_id + ".json")
            spk_info={}
            spk_info['user_id']      =   utt_id
            spk_info['prompt_wav']   =   os.path.join("./prompt_wavs", utt_id + ".wav")
            spk_info['prompt_num']   =   selected_idx
            spk_info['prompt_text']   =  selected_text
            with open(json_file,"w", encoding="utf-8") as fjson:
                json.dump(spk_info, fjson, ensure_ascii=False,)

            print(f'{utt_id} : cer : {selected_cer}, {selected_text}')
        else:
            error_list.append(utt_id)
            print(f'[error] {utt_id} : cer : {selected_cer}, asr_text:{audio_text}, selected_text: {selected_text}')


    print(f'[error] process finish!! error_count = {len(error_list)}, error_list = {error_list}')


if __name__ == "__main__":
    main()




