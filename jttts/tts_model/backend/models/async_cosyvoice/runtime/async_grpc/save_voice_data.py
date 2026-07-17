# Copyright (c) 2025 JiuTian ChinaMobile. All rights reserved.
# Authors: Ma Yong <mayongyjy@chinamobile.com>.

import io
import os
import re
import sys
import time
import uuid
import base64
import logging
import argparse
from typing import Optional, Literal, Type

import torch
import torchaudio
from typing_extensions import Annotated
from pydantic import BaseModel, Field, ValidationError

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{ROOT_DIR}/../')
sys.path.append(f'{ROOT_DIR}/../third_party/Matcha-TTS')
from async_cosyvoice.frontend import CosyVoiceFrontEnd

from hyperpyyaml import load_hyperpyyaml

# cosyvoice: AsyncCosyVoice2 | None = None

def main(args):
    global cosyvoice

    model_dir = args.model_dir
    with open('{}/cosyvoice2.yaml'.format(model_dir), 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'model_path': model_dir})
    frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                        configs['feat_extractor'],
                                        '{}/campplus.onnx'.format(model_dir),
                                        '{}/speech_tokenizer_v2.onnx'.format(model_dir),
                                        '{}/spk2info.pt'.format(model_dir),
                                        configs['allowed_special'])

    # customName = "TTS_09"
    # user_id = "002"
    # voice_id = str(uuid.uuid4())[:8]
    # uri = f"002"
    # audio_file_path = '/root/work/filestorage/usr/mayong/app/server/tts/f5_tts/F5-TTS/ref_waves/tts_09/TTS_009_3.wav'
    # text = "我会等王老师走了之后再关门的哎呀真的是太棒了恭喜你. "

    customName = "TTS_09"
    user_id = "003"
    voice_id = str(uuid.uuid4())[:8]
    uri = f"003"
    audio_file_path = 'ref_wavs/009_prompt_v13.wav'
    text = "我觉得吧就是你如果把这个样本好放大到。"

    prompt_speech_original, sample_rate = torchaudio.load(audio_file_path)
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    prompt_speech_16k = resampler(prompt_speech_original)

    frontend.generate_spk_info(
        uri,
        text,
        prompt_speech_16k,
        24000,
        customName
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8022)
    parser.add_argument('--model_dir', type=str,
                        default='/root/work/filestorage/usr/mayong/app/server/tts/cosyvoice/speed/version1/version_wshge/VLLM-CosyVoice/pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    parser.add_argument('--load_jit', action='store_true', help='load jit model')
    parser.add_argument('--load_trt', action='store_true', help='load tensorrt model')
    parser.add_argument('--fp16', action='store_true', help='use fp16')
    args = parser.parse_args()
    main(args)

    '''
cd /root/work/filestorage/usr/mayong/app/server/tts/cosyvoice/speed/version1/version_wshge/VLLM-CosyVoice/async_cosyvoice/runtime/async_grpc
python save_voice_data.py --load_jit --load_trt --fp16

import torch
spkinfo = torch.load('/root/work/filestorage/usr/mayong/app/server/tts/cosyvoice/speed/version1/version_wshge/VLLM-CosyVoice/pretrained_models/CosyVoice2-0.5B/spk2info.pt')
torch.save(spkinfo, '/root/work/filestorage/usr/mayong/app/server/tts/cosyvoice/speed/version1/version_wshge/VLLM-CosyVoice/pretrained_models/CosyVoice2-0.5B/spk2info.pt')
    '''

