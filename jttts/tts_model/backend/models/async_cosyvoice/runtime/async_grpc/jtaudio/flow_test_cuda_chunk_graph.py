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
import torch
from torch.cuda.amp import autocast

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

class CUDAGraphWrapper:
    def __init__(self, model, batch_size, token_max_len, warmup_iters=3):
        """
        初始化 CUDA Graph 封装器
        
        参数:
            model: 要加速的模型
            batch_size: 预期的批量大小
            token_max_len: 最大序列长度
            warmup_iters: 预热迭代次数
        """
        self.model = model
        self.batch_size = batch_size
        self.token_max_len = token_max_len
        self.warmup_iters = warmup_iters
        self.is_initialized = False
        self.graph = None
        self.static_inputs = {}
        self.static_outputs = {}
        
        # 创建静态输入张量
        self._create_static_tensors()
    
    def _create_static_tensors(self):
        """创建静态输入输出张量"""
        device = next(self.model.parameters()).device
        
        # 静态输入
        self.static_inputs = {
            'token': torch.randint(0, 100, (self.batch_size, self.token_max_len), dtype=torch.int32), 
            'token_len': torch.zeros((self.batch_size), dtype=torch.int32) + self.token_max_len,
            'embedding': torch.randn((self.batch_size, 192))
        }
        
        # 静态输出 (根据你的实际输出结构调整)
        self.static_output = torch.zeros((self.batch_size, 80, self.token_max_len * 2))
        
        # 移动到设备
        for k, v in self.static_inputs.items():
            self.static_inputs[k] = v.to(device)
        self.static_output = self.static_output.to(device)
    
    def _warmup(self):
        """预热模型以确定图结构"""
        print("Warming up for CUDA Graph initialization...")
        for _ in range(self.warmup_iters):
            with autocast():
                self.model.inference_chunck_cudagraph(**self.static_inputs)
        torch.cuda.synchronize()
    
    def _capture_graph(self):
        """捕获CUDA图"""
        print("Capturing CUDA Graph...")
        
        # 创建图
        self.graph = torch.cuda.CUDAGraph()
        
        # 预热
        self._warmup()
        
        # 开始捕获
        with torch.cuda.graph(self.graph):
            with autocast():
                out = self.model.inference_chunck_cudagraph(**self.static_inputs)
        
        # 保存静态输出
        self.static_output.copy_(out)
        torch.cuda.synchronize()
        
        self.is_initialized = True
        print("CUDA Graph initialized successfully.")
    
    def infer(self, **kwargs):
        """
        执行推理
        
        参数:
            kwargs: 与原始inference_batch_true相同的输入参数
        返回:
            模型输出
        """
        # 检查输入形状是否匹配
        for k, v in kwargs.items():
            if v.shape != self.static_inputs[k].shape:
                raise ValueError(f"Input {k} shape {v.shape} doesn't match static shape {self.static_inputs[k].shape}")
        
        # 如果未初始化，先捕获图
        if not self.is_initialized:
            self._capture_graph()
        
        # 将输入数据复制到静态输入
        for k, v in kwargs.items():
            self.static_inputs[k].copy_(v)
        
        # 重放图
        self.graph.replay()
        
        # 返回输出
        return self.static_output.clone()

for batch_size in batch_size_list:
    for token_len_actual in token_list:
        token = token_1.repeat(batch_size, 1)[:, 0:token_len_actual]
        token_len = (token_len_1 / token_len_1 * token_len_actual).to(torch.int32).repeat(batch_size)
        embedding = embeddings.repeat(batch_size, 1)
        inputs = {
            'token': token,
            'token_len': token_len,
            'embedding': embedding
        }
        tts_mel, _ = flow_model.inference_chunck_cudagraph(token,
                        token_len,
                        embedding,
                    )

        graph_wrapper = CUDAGraphWrapper(
                            model=flow_model,
                            batch_size=batch_size,
                            token_max_len=token_len_actual,
                            warmup_iters=3
                        )

        time_record = 0
        for_loop = 5
        for i in range(for_loop):
            output = graph_wrapper.infer(**inputs)
            torch.cuda.synchronize()
            time_start_one_step_end = time.time() * 1000
            time_record = time_record + time_start_one_step_end - time_start_one_step_start
        time_average = time_record / for_loop
        logging.info(f'[{batch_size}, {token_len_actual}] tiem: {time_average} ms.')
