# Copyright (c) 2025 JiuTian ChinaMobile. All rights reserved.
# Authors: Ma Yong <mayongyjy@chinamobile.com>.
import os
import signal
import sys
import time
import asyncio
from concurrent import futures
import argparse
from typing import AsyncGenerator, Callable, Tuple, AsyncIterator, Union

import torch

import cosyvoice_pb2
import cosyvoice_pb2_grpc
import logging
import grpc
from grpc import aio
import torchaudio
from hyperpyyaml import load_hyperpyyaml

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{ROOT_DIR}/../../..')
sys.path.append(f'{ROOT_DIR}/../../../third_party/Matcha-TTS')
from async_cosyvoice.async_cosyvoice import AsyncCosyVoice2
from async_cosyvoice.runtime.async_grpc.utils import convert_audio_tensor_to_bytes, convert_audio_bytes_to_tensor

logging.basicConfig(level=logging.INFO, format='[ %(levelname)s %(asctime)s %(filename)s:%(lineno)d ] %(message)s')


parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=50000)
parser.add_argument('--max_conc', type=int, default=10)
parser.add_argument('--model_dir', type=str,
                    default='/root/work/filestorage/usr/mayong/app/server/tts/cosyvoice/speed/version1_new/CosyVoice/pretrained_models/CosyVoice2-0.5B',
                    help='local path or modelscope repo id')
parser.add_argument('--load_jit', action='store_true', help='load jit model')
parser.add_argument('--load_trt', action='store_true', help='load tensorrt model')
parser.add_argument('--fp16', action='store_true', help='use fp16')
args = parser.parse_args()

model_dir = args.model_dir
device = torch.device('cuda:0')
with open('{}/cosyvoice2.yaml'.format(model_dir), 'r') as f:
    configs = load_hyperpyyaml(f, overrides={'model_path': model_dir})
hift_model_file = '{}/hift.pt'.format(model_dir)
hift_model = configs['hift']
hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(hift_model_file, weights_only=True, map_location=device).items()}
hift_model.load_state_dict(hift_state_dict, strict=True)
hift_model.to(device).eval()
# torch.save(f'jtaudio/data/hift_inpu_{chunk_index}.pt', {"tts_mel":tts_mel.cpu(), "hift_cache_source": hift_cache_source.cpu()})

hift_input_dict_0 = torch.load('jtaudio/data/hift_inpu_0.pt', weights_only=True)
hift_input_dict_1 = torch.load('jtaudio/data/hift_inpu_1.pt', weights_only=True)
hift_input_dict_all = torch.load('jtaudio/data/hift_inpu_-1.pt', weights_only=True)

tts_mel = hift_input_dict_0['tts_mel'].to(device)
hift_cache_source = hift_input_dict_0['hift_cache_source'].to(device)


tts_speech, tts_source = hift_model.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
tts_source_part1 = hift_model.inference_part1(speech_feat=tts_mel, cache_source=hift_cache_source)
logging.info(f'tts_mel: {tts_mel.size()} hift_cache_source: {hift_cache_source.size()}')


tts_mel_batch = tts_mel.repeat(2, 1, 1)
hift_cache_source_batch = hift_cache_source.repeat(2, 1, 1)
tts_speech_batch, tts_source_batch = hift_model.inference(speech_feat=tts_mel_batch, cache_source=hift_cache_source_batch)
tts_source_part1_batch = hift_model.inference_part1(speech_feat=tts_mel_batch, cache_source=hift_cache_source_batch)
logging.info(f'tts_mel_batch: {tts_mel_batch.size()} hift_cache_source_batch: {hift_cache_source_batch.size()}')


# [16, 80, 40]
tts_mel_batch = tts_mel.repeat(16, 1, 1)
hift_cache_source_batch = hift_cache_source.repeat(16, 1, 1)
for i in range(10):
    time_start_one_step_start = time.time() * 1000
    tts_source_part1 = hift_model.inference_part1(speech_feat=tts_mel, cache_source=hift_cache_source)
    torch.cuda.synchronize()
    time_start_one_step_end = time.time() * 1000
    print(f'tts_mel: {tts_mel_batch.size()} inference_part1 time_start_one_step: {time_start_one_step_end - time_start_one_step_start} ms.')

s = tts_source_part1
for i in range(10):
    time_start_one_step_start = time.time() * 1000
    tts_source_part1 = hift_model.inference_part2(speech_feat=tts_mel, s=s)
    torch.cuda.synchronize()
    time_start_one_step_end = time.time() * 1000
    print(f'tts_mel: {tts_mel_batch.size()} inference_part2 time_start_one_step: {time_start_one_step_end - time_start_one_step_start} ms.')


print(f'*********************************')
# [16, 80, 48]
tts_mel = hift_input_dict_1['tts_mel'].to(device)
hift_cache_source = hift_input_dict_1['hift_cache_source'].to(device)
tts_mel_batch = tts_mel.repeat(16, 1, 1)
hift_cache_source_batch = hift_cache_source.repeat(16, 1, 1)
for i in range(10):
    time_start_one_step_start = time.time() * 1000
    tts_source_part1 = hift_model.inference_part1(speech_feat=tts_mel, cache_source=hift_cache_source)
    torch.cuda.synchronize()
    time_start_one_step_end = time.time() * 1000
    print(f'tts_mel: {tts_mel_batch.size()} time_start_one_step: {time_start_one_step_end - time_start_one_step_start} ms.')

import pdb;pdb.set_trace()
torch.allclose(tts_source, tts_source_part1, rtol=1e-02, atol=1e-03)
