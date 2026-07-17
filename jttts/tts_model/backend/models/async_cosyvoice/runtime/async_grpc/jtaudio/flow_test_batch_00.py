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

flow_model = configs['flow']
flow_model.fp16 = True
flow_model.half()
flow_model.encoder.static_chunk_size = 2 * flow_model.input_frame_rate
flow_model.decoder.estimator.static_chunk_size = 2 * flow_model.input_frame_rate * flow_model.token_mel_ratio

flow_model_file = '{}/flow.pt'.format(model_dir)
flow_model.load_state_dict(torch.load(flow_model_file, weights_only=True, map_location=device), strict=True)
flow_model.to(device).eval()

flow_input_data_0 = torch.load('jtaudio/data/flow_data_dict_1747120900291.5042.pt', weights_only=True)
flow_input_data_1 = torch.load('jtaudio/data/flow_data_dict_1747120987523.7107.pt', weights_only=True)

token_0 = flow_input_data_0['token'][:1,...]                         # [1, 51]          torch.int64
token_len_0 = flow_input_data_0['token_len'][:1,...]                 # [1]              torch.int32
prompt_token = flow_input_data_0['prompt_token'][:1,...]             # [1, 101]         torch.int32
prompt_token_len = flow_input_data_0['prompt_token_len'][:1,...]     # [1]              torch.int32
prompt_feat = flow_input_data_0['prompt_feat'][:1,...]               # [1, 202, 80]     torch.float32
prompt_feat_len = flow_input_data_0['prompt_feat_len'][:1,...]       # [1]              torch.int32
embedding = flow_input_data_0['embedding'][:1,...]                   # [1, 192]         torch.float32
finalizes = flow_input_data_0['finalizes'][:1,...]                   # [1]              torch.int32

token_1 = flow_input_data_1['token'][:1,...]                         # [1, 64]
token_len_1 = flow_input_data_1['token_len'][:1,...]                 # [1]

tts_mel, _  = flow_model.inference(
                token=token_0,
                token_len=token_len_0,
                prompt_token=prompt_token,
                prompt_token_len=prompt_token_len,
                prompt_feat=prompt_feat,
                prompt_feat_len=prompt_feat_len,
                embedding=embedding,
                finalizes=finalizes
            )
