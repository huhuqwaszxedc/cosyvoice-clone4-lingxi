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
import torchaudio
import torch


from pydub.silence import split_on_silence
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError


from jttts.config import GlobalConfigInst


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
        "--input_dir",
        type=str,
        required=True,
        help="input_dir.",
    )
    args = parser.parse_args()
    return args


class ASR_Model():
    def __init__(self):

        asr_model = 'speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
        self.asr_model_dir = os.path.join(GlobalConfigInst.pretrained_model_rootdir, asr_model)

        vad_model = 'speech_fsmn_vad_zh-cn-16k-common-pytorch'
        self.vad_model_dir = os.path.join(GlobalConfigInst.pretrained_model_rootdir, vad_model)

        punc_model = 'punc_ct-transformer_zh-cn-common-vocab272727-pytorch'
        self.punc_model_dir = os.path.join(GlobalConfigInst.pretrained_model_rootdir, punc_model)


        if os.path.exists(self.asr_model_dir) is False:
            return None
        if os.path.exists(self.vad_model_dir) is False:
            return None
        if os.path.exists(self.punc_model_dir) is False:
            return None

        self.asr_model = pipeline(
            task=Tasks.auto_speech_recognition,
            model=self.asr_model_dir, model_revision="v2.0.4",
            vad_model=self.vad_model_dir, vad_model_revision="v2.0.4",
            punc_model=self.punc_model_dir, punc_model_revision="v2.0.4",
            # spk_model="iic/speech_campplus_sv_zh-cn_16k-common",
            # spk_model_revision="v2.0.2",
            disable_update=True,
        )

    def forward(self, audio, sample_rate):
        try:
            if len(audio.shape) > 1:
                audio = np.mean(audio,axis=1)

            if sample_rate != 16000:
                audio = torch.Tensor(audio)
                audio = torchaudio.functional.resample(audio, orig_freq=sample_rate, new_freq=16000)
                audio_new = audio.cpu().numpy()
                sample_rate_new = 16000
            # if sample_rate != 16000:
            #     audio_new = librosa.resample(y=audio, orig_sr=sample_rate, target_sr=16000)
            #     sample_rate_new = 16000
            else:
                audio_new       = audio
                sample_rate_new = sample_rate

            text = self.asr_model(audio_new, fs=sample_rate_new, is_final=True)[0]["text"]
            return text
        except Exception as e:
            print(f'asr inference failed, please check')
            return 'null'


    def inference(self, audio, sample_rate):
        return self.forward(audio, sample_rate)



class Denoise_Model():
    def __init__(self):
        super().__init__()
        denoise_model = 'speech_dfsmn_ans_psm_48k_causal'
        self.denoise_model_dir = os.path.join(GlobalConfigInst.pretrained_model_rootdir, denoise_model)

        if os.path.exists(self.denoise_model_dir) is False:
            return None


        self.denoise_model = pipeline(
            Tasks.acoustic_noise_suppression,
            model= self.denoise_model_dir,
            disable_update= True,
        )
    def forward(self, audio, sample_rate):
        try:
            if len(audio.shape) > 1:
                audio = np.mean(audio,axis=1)

            if sample_rate != 48000:
                audio = torch.Tensor(audio)
                audio = torchaudio.functional.resample(audio, orig_freq=sample_rate, new_freq=48000)
                audio_new = audio.cpu().numpy()
                sample_rate_new = 48000
            else:
                audio_new       = audio
                sample_rate_new = sample_rate

            current_millis = int(round(time.time() * 1000))
            timestamp = time.strftime("%m-%d_%H-%M-%S", time.localtime())
            timestamp = timestamp + '-' + str(current_millis)[-3:]

            denoise_in_wavpath = './user_prompt_denoise_in_' + timestamp + '.wav'
            sf.write(denoise_in_wavpath, audio_new, sample_rate_new, "PCM_16",)
            denoise_out_wavpath = './user_prompt_denoise_out_' + timestamp + '.wav'

            result = self.denoise_model(denoise_in_wavpath,  output_path=denoise_out_wavpath)

            denoise_audio, denoise_sample_rate = sf.read(denoise_out_wavpath)

            if denoise_sample_rate != sample_rate:
                denoise_audio = torch.Tensor(denoise_audio)
                denoise_audio = torchaudio.functional.resample(denoise_audio, orig_freq=denoise_sample_rate, new_freq=sample_rate)
                denoise_audio_out = denoise_audio.cpu().numpy()
                denoise_sr_out = sample_rate
            else:
                denoise_audio_out       = denoise_audio
                denoise_sr_out = sample_rate

            os.system(f'rm -rf {denoise_in_wavpath}')
            os.system(f'rm -rf {denoise_out_wavpath}')

            return denoise_audio_out, denoise_sr_out

        except Exception as e:
            print(f'denoise inference failed, please check')
            return audio, sample_rate

    def inference(self, audio, sample_rate):
        return self.forward(audio, sample_rate)




class RemoveSil_Model():
    def __init__(self):
        super().__init__()


    def slice(self, audio_path):
        """_summary_

        Args:
            audio_path (_type_): audio path
        """
        try:
            audio = AudioSegment.from_file(audio_path)
        except:
            print(audio_path)
            return 0

        segments = split_on_silence(
            audio, min_silence_len=200, silence_thresh=-50, seek_step=100, keep_silence=100
        )


        combined = segments[0]
        for i in range(1, len(segments)):
            combined += segments[i]
        return combined


    def forward(self, audio, sample_rate):
        
        try:
            current_millis = int(round(time.time() * 1000))
            timestamp = time.strftime("%m-%d_%H-%M-%S", time.localtime())
            timestamp = timestamp + '-' + str(current_millis)[-3:]

            rmsil_in_wavpath = './user_prompt_rmsil_in_' + timestamp + '.wav'
            sf.write(rmsil_in_wavpath, audio, sample_rate, "PCM_16",)
            rmsil_out_wavpath = './user_prompt_rmsil_out_' + timestamp + '.wav'

            combined = self.slice(audio_path=rmsil_in_wavpath)
            name = rmsil_in_wavpath.split(sep="/")[-1] + '_rmsil'
            combined.export(rmsil_out_wavpath, format="wav")

            rmsil_audio, rmsil_sample_rate = sf.read(rmsil_out_wavpath)

            os.remove(rmsil_in_wavpath)
            os.remove(rmsil_out_wavpath)

            return rmsil_audio, rmsil_sample_rate
        except Exception as e:
            print(f'rmsil inference failed, please check')
            return audio, sample_rate

    def inference(self, audio, sample_rate):
        return self.forward(audio, sample_rate)





class SpeakerVerification_Model():
    def __init__(self):
        super().__init__()
        sv_model = 'speech_eres2net_sv_zh-cn_16k-common'
        self.sv_model_dir = os.path.join(GlobalConfigInst.pretrained_model_rootdir, sv_model)
        if os.path.exists(self.sv_model_dir) is False:
            return None

        self.sv_model = pipeline(
            task='speaker-verification',
            model=self.sv_model_dir,
            model_revision='v1.0.5',
            disable_update= True,
        )
        self.required_sr = 16000


    def forward(self, audio_a, sample_rate_a, audio_b, sample_rate_b):
        try:
            if len(audio_a.shape) > 1:
                audio_a = np.mean(audio_a,axis=1)

            if sample_rate_a != self.required_sr:
                audio_a = torch.Tensor(audio_a)
                audio_a = torchaudio.functional.resample(audio_a, orig_freq=sample_rate_a, new_freq=self.required_sr)
                audio_a = audio_a.cpu().numpy()
                sample_rate_a = self.required_sr

            if sample_rate_b != self.required_sr:
                audio_b = torch.Tensor(audio_b)
                audio_b = torchaudio.functional.resample(audio_b, orig_freq=sample_rate_b, new_freq=self.required_sr)
                audio_b = audio_b.cpu().numpy()
                sample_rate_b = self.required_sr

            audio_a = librosa.util.normalize(audio_a) * 0.95
            audio_b = librosa.util.normalize(audio_b) * 0.95

            result = self.sv_model([audio_a, audio_b])
            result = self.sv_model([audio_a, audio_b], output_emb=True)
            score  = result['outputs']['score']

            return score

        except Exception as e:
            print(f'speaker-verification inference failed, please check')
            return audio_a, sample_rate_a, audio_b, sample_rate_b

    def inference(self, audio_a, sample_rate_a, audio_b, sample_rate_b):
        return self.forward(audio_a, sample_rate_a, audio_b, sample_rate_b)










class VAD_Model():
    def __init__(self):
        super().__init__()
        self.vad_model = pipeline(
            task=Tasks.voice_activity_detection,
            model='iic/speech_fsmn_vad_zh-cn-16k-common-pytorch',
            model_revision="v2.0.4",
            disable_update=True,
        )
    def forward(self, audio, sample_rate):
        if len(audio.shape) > 1:
            audio = np.mean(audio,axis=1)
        if sample_rate != 16000:
            audio = torch.Tensor(audio)
            audio = torchaudio.functional.resample(audio, orig_freq=sample_rate, new_freq=16000)
            audio_new = audio.cpu().numpy()
            sample_rate_new = 16000
        # if sample_rate != 16000:
        #     audio_new = librosa.resample(y=audio, orig_sr=sample_rate, target_sr=16000)
        #     sample_rate_new = 16000
        else:
            audio_new       = audio
            sample_rate_new = sample_rate


        interval = self.vad_model(audio_new, fs=sample_rate_new, is_final=True)[0]['value']
        max_sil = 0.0
        p_end_ms = 0.0

        # if the audio is empty
        if len(interval) == 0: 
            max_sil = 200000

        for idy, sub_itvl in enumerate(interval):
            start_ms    = sub_itvl[0]
            end_ms      = sub_itvl[1]
            if idy == 0:
                max_sil = start_ms
            if (start_ms - p_end_ms) > max_sil and idy > 0:
                max_sil = start_ms - p_end_ms
            p_end_ms    = end_ms

        if (audio.shape[0] * 1000 / sample_rate - p_end_ms) > max_sil:
            max_sil = audio.shape[0] * 1000 / sample_rate - p_end_ms 

        return max_sil   # ms

    def inference(self, audio, sample_rate):
        return self.forward(audio, sample_rate)





def main():
    args = parse_and_config()
    input_dir       = args.input_dir
    ASR_Model_Inst = ASR_Model()
    VAD_Model_Inst = VAD_Model()


    all_wave_files = get_dir_all_files(input_dir)
    all_wave_files = all_wave_files[:100]
    asr_rtf_recip_sum = 0
    vad_rtf_recip_sum = 0
    for idx, subfile_path in enumerate(all_wave_files):
        
        utt_id = subfile_path.split('/')[-1][:-len('.wav')]
        audio_data, sample_rate = sf.read(subfile_path)
        start_time = time.time()
        text = ASR_Model_Inst.inference(audio_data, sample_rate)
        asr_end_time = time.time()
        asr_time_cost = asr_end_time - start_time

        start_time = time.time()
        max_sil = VAD_Model_Inst.inference(audio_data, sample_rate)
        vad_end_time = time.time()
        vad_time_cost = vad_end_time - start_time

        audio_len_s = audio_data.shape[0] / sample_rate

        asr_rtf_recip = audio_len_s / asr_time_cost
        vad_rtf_recip = audio_len_s / vad_time_cost

        asr_rtf_recip_sum   += asr_rtf_recip
        vad_rtf_recip_sum   += vad_rtf_recip

        print(f'{utt_id}  text = {text}, max_sil = {max_sil} ms, audio_len_ms = {audio_len_s},  asr_rtf_recip={asr_rtf_recip}, vad_rtf_recip={vad_rtf_recip}')



    asr_rtf_recip_avg = asr_rtf_recip_sum/len(all_wave_files)
    vad_rtf_recip_avg = vad_rtf_recip_sum/len(all_wave_files)

    print(f' asr_rtf_recip_avg = {asr_rtf_recip_avg}, vad_rtf_recip_avg = {vad_rtf_recip_avg}')


if __name__ == "__main__":
    main()




