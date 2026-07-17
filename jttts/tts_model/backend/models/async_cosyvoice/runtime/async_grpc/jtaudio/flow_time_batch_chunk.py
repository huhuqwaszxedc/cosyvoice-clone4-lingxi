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

flow_input_data_1 = torch.load('jtaudio/data/flow_data_dict_1747120987523.7107.pt', weights_only=True)

token_1 = flow_input_data_1['token'][:1,...]                         # [1, 64]
token_len_1 = flow_input_data_1['token_len'][:1,...]                 # [1]
token_len_0 = flow_input_data_1['token_len'][:1,...]                 # [1]              torch.int32
prompt_token = flow_input_data_1['prompt_token'][:1,...]             # [1, 101]         torch.int32
prompt_token_len = flow_input_data_1['prompt_token_len'][:1,...]     # [1]              torch.int32
prompt_feat = flow_input_data_1['prompt_feat'][:1,...]               # [1, 202, 80]     torch.float32
prompt_feat_len = flow_input_data_1['prompt_feat_len'][:1,...]       # [1]              torch.int32
embeddings = flow_input_data_1['embedding'][:1,...]                   # [1, 192]         torch.float32
finalizes = flow_input_data_1['finalizes'][:1,...]                   # [1]              torch.int32

token_1 = token_1.repeat(1, 4)  # [1, 256]

batch_size_list = [2, 4, 8, 16, 24]
token_list = [23, 43, 63, 83, 103, 123, 143, 163, 183, 203]

for batch_size in batch_size_list:
    for token_len_actual in token_list:
        token = token_1.repeat(batch_size, 1)[:, 0:token_len_actual]
        token_len = (token_len_1 / token_len_1 * token_len_actual).to(torch.int32).repeat(batch_size)
        embedding = embeddings.repeat(batch_size, 1)
        # logging.info(f'token: {token.size()} embedding: {embedding.size()}')
        for i in range(5):
            tts_mel, _ = flow_model.inference_chunck(token,
                            token_len,
                            None,
                            None,
                            None,
                            None,
                            embedding,
                            None
                        )

        time_record = 0
        for_loop = 5
        for i in range(for_loop):
            time_start_one_step_start = time.time() * 1000
            tts_mel, _ = flow_model.inference_chunck(token,
                            token_len,
                            None,
                            None,
                            None,
                            None,
                            embedding,
                            None
                        )
            torch.cuda.synchronize()
            time_start_one_step_end = time.time() * 1000
            time_record = time_record + time_start_one_step_end - time_start_one_step_start
        time_average = time_record / for_loop
        logging.info(f'[{batch_size}, {token_len_actual}] tiem: {time_average} ms.')

