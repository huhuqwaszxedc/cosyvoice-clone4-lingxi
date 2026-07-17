# Copyright (c) 2025 JT ChinaMobile. All rights reserved.
# Authors: Ma Yong <mayongyjy@chinamobile.com>.

import torch
import uuid
import yaml
import argparse
import copy
import tensorrt as trt
import sys
import os
import time
import torch.multiprocessing as mp

from jtservers.core.model import BaseModel_WorkerConfig, BaseModelWorker
from jtservers.core.request import Request_Single, Request_Batch
from jtservers.core.colorslog import ColorsLog
from hyperpyyaml import load_hyperpyyaml
from jttts.config import GlobalConfigInst

import logging as log
log.basicConfig(level=log.INFO,format='[ %(levelname)s %(asctime)s %(filename)s:%(lineno)d ] %(message)s')

torch.multiprocessing.set_sharing_strategy('file_system')

class CosyFlowMatchingWorker(BaseModelWorker):       
    def __init__(self, basemodel_workerconfig, config_file):
        super(CosyFlowMatchingWorker, self).__init__(basemodel_workerconfig, config_file)
        self.worker_name = basemodel_workerconfig.model_name
        cuda_idx_offset = 1 if self.worker_name != 'CosyFlowMatchingStream0' else 0
        # cuda_idx_offset = -1 if self.worker_name != 'CosyFlowMatchingStream0' else 0
        # cuda_idx_offset = 0
        if GlobalConfigInst.tts_gpu_num == 3:
            self.use_cuda_idx = 1 + cuda_idx_offset # CosyFlowMatchingStream0--1,  CosyFlowMatchingStream1--2, CosyFlowMatchingOffline--2
        elif GlobalConfigInst.tts_gpu_num == 2:
            cuda_idx_offset = 0
            self.use_cuda_idx = 1 + cuda_idx_offset # CosyFlowMatchingStream0--1,  CosyFlowMatchingStream1--1, CosyFlowMatchingOffline--1
        elif GlobalConfigInst.tts_gpu_num == 1:
            self.use_cuda_idx = 0 # CosyFlowMatchingStream0--0,  CosyFlowMatchingStream1--0, CosyFlowMatchingOffline--0
    
    def model_init(self, process_idx, model_dir):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.use_cuda_idx)
        log.info(f'[{self.model_name}:{self.use_cuda_idx}] model_init use_cuda_idx: {self.use_cuda_idx}')

        device = torch.device('cuda')
        torch._dynamo.config.cache_size_limit = 1000
        fp16 = True
        a800_plan = '{}/flow.decoder.estimator.fp16.a800.my.plan'.format(model_dir)
        h100_plan = '{}/flow.decoder.estimator.fp16.h100.jtaudio.plan'.format(model_dir)

        if os.path.exists(h100_plan):
            flow_decoder_estimator_model = h100_plan
        else:
            flow_decoder_estimator_model = a800_plan

        # flow_decoder_estimator_model = '{}/flow.decoder.estimator.fp16.a800.my.plan'.format(model_dir)

        # init model
        with open('{}/cosyvoice3.yaml'.format(model_dir), 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'model_path': model_dir})
        flow = configs['flow']
        # flow.load_state_dict(torch.load('{}/flow.pt'.format(model_dir), weights_only=True, map_location="cuda"), strict=True)
        
        flow_model = torch.load('{}/flow.pt'.format(model_dir), weights_only=True, map_location="cuda")
        flow_model.pop('epoch', None)
        flow_model.pop('step', None)
        flow.load_state_dict(flow_model, strict=True)
        flow.to("cuda").eval()        
        
        flow = flow.to("cuda")

        flow.fp16 = fp16
        if fp16 is True:
            flow.half()
        token_hop_len = 2 * flow.input_frame_rate
        # here we fix flow encoder/decoder decoding_chunk_size, in the future we will send it as arguments, or use cache
        flow.encoder.static_chunk_size = 2 * flow.input_frame_rate
        flow.decoder.estimator.static_chunk_size = 2 * flow.input_frame_rate * flow.token_mel_ratio

        # flow_encoder_model = '{}/flow.encoder.{}.zip'.format(model_dir, 'fp16' if fp16 is True else 'fp32')
        # flow_encoder = torch.jit.load(flow_encoder_model, map_location=device)
        # flow.encoder = flow_encoder

        assert torch.cuda.is_available(), 'tensorrt only supports gpu!'
        assert os.path.exists(flow_decoder_estimator_model), 'tensorrt model not existed!'
        if os.path.getsize(flow_decoder_estimator_model) == 0:
            raise ValueError('{} is empty file, delete it and export again!'.format(flow_decoder_estimator_model))
        del flow.decoder.estimator
        with torch.cuda.device("cuda"):
            with open(flow_decoder_estimator_model, 'rb') as f:
                flow.decoder.estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
            if flow.decoder.estimator_engine is None:
                raise ValueError('failed to load trt {}'.format(flow_decoder_estimator_model))
            flow.decoder.estimator = flow.decoder.estimator_engine.create_execution_context()

        hift_model_file = '{}/hift.pt'.format(model_dir)
        hift_model = configs['hift']
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(hift_model_file, weights_only=True, map_location=device).items()}
        hift_model.load_state_dict(hift_state_dict, strict=True)
        hift_model.to(device).eval()

        model_list = [flow, hift_model]
        return model_list

    def expand_tensor(self, tensor_input, target_dim):
        current_dim = tensor_input.size(0)

        if current_dim <= target_dim:
            num_repeats = (target_dim + current_dim - 1) // current_dim
            repeated_tensor = tensor_input.repeat(num_repeats, *([1] * (tensor_input.dim() - 1)))
            expanded_tensor = repeated_tensor[:target_dim]
        else:
            expanded_tensor = tensor_input[:target_dim]
        return expanded_tensor

    def forward(self, process_idx, model_list, request_batch, batch_input_tensor_list):
        device = torch.device("cuda")

        current_pid = os.getpid()
        flow_model, hift_model = model_list[0], model_list[1]

        token=batch_input_tensor_list[0]
        token=token.squeeze(1).to(device)
        token_len=batch_input_tensor_list[1]
        token_len=token_len.squeeze(1).to(device)
        embedding=batch_input_tensor_list[2]
        finalizes=batch_input_tensor_list[3]
        embedding, finalizes = embedding.squeeze(1).to(device), finalizes.squeeze(1).to(device)
        
        batch_size_current = token.size(0)
        if batch_size_current <= 4:
            target_batch_size = 4
        elif batch_size_current <= 16:
            target_batch_size = 16
        elif batch_size_current <= 24:
            target_batch_size = 24
        elif batch_size_current <= 32:
            target_batch_size = 32
        elif batch_size_current <= 48:
            target_batch_size = 48
        else:
            target_batch_size = batch_size_current
        token = self.expand_tensor(token, target_batch_size)
        token_len = self.expand_tensor(token_len, target_batch_size)
        prompt_token = None
        prompt_token_len = None
        prompt_feat = None
        prompt_feat_len = None
        embedding = self.expand_tensor(embedding, target_batch_size)
        finalizes = self.expand_tensor(finalizes, target_batch_size)

        time_start = time.time() * 1000
        if GlobalConfigInst.infer_speed_mode == 0:
            if finalizes[0].item() is True:
                tts_mel, mel_len  = flow_model.inference_batch_true(    # for tts
                                        token=token,
                                        token_len=token_len,
                                        prompt_token=prompt_token,
                                        prompt_token_len=prompt_token_len,
                                        prompt_feat=prompt_feat,
                                        prompt_feat_len=prompt_feat_len,
                                        embedding=embedding,
                                        finalizes=finalizes
                                    )
            else:
                tts_mel, _  = flow_model.inference_chunck(  # for tts
                                token=token,
                                token_len=token_len,
                                prompt_token=prompt_token,
                                prompt_token_len=prompt_token_len,
                                prompt_feat=prompt_feat,
                                prompt_feat_len=prompt_feat_len,
                                embedding=embedding,
                                finalizes=finalizes
                            )
        else:
            if finalizes[0].item() is True:
                tts_mel, mel_len  = flow_model.inference(             # only for clone 
                                        token=token,
                                        token_len=token_len,
                                        prompt_token=prompt_token,
                                        prompt_token_len=prompt_token_len,
                                        prompt_feat=prompt_feat,
                                        prompt_feat_len=prompt_feat_len,
                                        embedding=embedding,
                                        finalizes=finalizes
                                    )
            else:
                tts_mel, _  = flow_model.inference(   # only for clone 
                                token=token,
                                token_len=token_len,
                                prompt_token=prompt_token,
                                prompt_token_len=prompt_token_len,
                                prompt_feat=prompt_feat,
                                prompt_feat_len=prompt_feat_len,
                                embedding=embedding,
                                finalizes=finalizes
                            )





        time_end = time.time() * 1000
        # log.info(f'[{self.model_name}:{self.use_cuda_idx}] token: {token.size()} tts_mel: {tts_mel.size()} takes: {time_end - time_start} ms.')

        # import rpdb;rpdb.set_trace()

        if finalizes[0].item() is True:
            # token: torch.Size([4, 98]) tts_mel: torch.Size([4, 80, 196])
            tts_mel = tts_mel[0:batch_size_current, ...].cpu()
            mel_len = mel_len[0:batch_size_current, ...].cpu()
            # log.info(f'[{self.model_name}:{self.use_cuda_idx}] finalizes[0] is True.')
            # log.info(f'[{self.model_name}:{self.use_cuda_idx}] finalizes[0] is True. mel_len: {mel_len}')
            return [tts_mel, mel_len]
        elif tts_mel.size(2) == 40:
            # token: torch.Size([4, 23]) tts_mel: torch.Size([4, 80, 46])
            token_offset = 0
            tts_mel = tts_mel[:, :, token_offset * flow_model.token_mel_ratio:(token_len[0] - flow_model.pre_lookahead_len) * flow_model.token_mel_ratio]

            tts_speech, tts_source = hift_model.inference_part(speech_feat=tts_mel)
            tts_mel = tts_mel[0:batch_size_current, ...].cpu()
            tts_speech = tts_speech[0:batch_size_current, ...].cpu()
            tts_source = tts_source[0:batch_size_current, ...].cpu()

            # log.info(f'[{self.model_name}:{self.use_cuda_idx}] tts_mel: {tts_mel.size()} tts_speech: {tts_speech.size()} tts_source: {tts_source.size()}.')
            return [tts_mel, tts_speech, tts_source]
        # elif tts_mel.size(2) == 80:
        else:
            hift_cache_mel = batch_input_tensor_list[4].to(device).squeeze(1)
            hift_cache_source = batch_input_tensor_list[5].to(device).squeeze(1)
            token_offset_tensor = batch_input_tensor_list[6].to(device).squeeze(1)
            hift_cache_mel = self.expand_tensor(hift_cache_mel, target_batch_size)
            hift_cache_source = self.expand_tensor(hift_cache_source, target_batch_size)
            token_offset_tensor = self.expand_tensor(token_offset_tensor, target_batch_size)
            # log.info(f'token_offset_tensor: {token_offset_tensor}')
            # token: torch.Size([4, 43]) tts_mel: torch.Size([4, 80, 80])    20*2
            # token: torch.Size([4, 63]) tts_mel: torch.Size([4, 80, 120])   40*2
            # token: torch.Size([4, 83]) tts_mel: torch.Size([4, 80, 160])   60*2

            token_offset_tensor_new = token_offset_tensor.clone() * flow_model.token_mel_ratio
            batch_indices = torch.arange(target_batch_size).view(-1, 1, 1).to(device)
            middle_indices = torch.arange(80).view(1, -1, 1).to(device)
            offset = torch.arange(flow_model.peer_chunk_token_num * 2).view(1, 1, -1).to(device)
            col_indices = token_offset_tensor_new.view(-1, 1, 1) + offset
            col_indices = torch.clamp(col_indices, 0, tts_mel.shape[2] - 1)
            tts_mel = tts_mel[batch_indices, middle_indices, col_indices]
            # log.info(f'col_indices: {col_indices}')
            # log.info(f'tts_mel: {tts_mel.size()}')
            # tts_mel = tts_mel[:, :, token_offset_tensor * flow_model.token_mel_ratio: token_offset_tensor * flow_model.token_mel_ratio + flow_model.peer_chunk_token_num * 2]

            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
            hift_cache_source_new = hift_model.inference_part1(speech_feat=tts_mel)
            # log.info(f'[{self.model_name}:{self.use_cuda_idx}] tts_mel: {tts_mel.size()} hift_cache_source_new: {hift_cache_source_new.size()}.')

            tts_source = hift_cache_source_new.clone()
            tts_source[:, :, :hift_cache_source.shape[2]] = hift_cache_source
            tts_speech = hift_model.inference_part2(speech_feat=tts_mel, s = tts_source)

            tts_mel = tts_mel[0:batch_size_current, ...].cpu()
            tts_speech = tts_speech[0:batch_size_current, ...].cpu()
            tts_source = tts_source[0:batch_size_current, ...].cpu()
            # log.info(f'[{self.model_name}:{self.use_cuda_idx}] tts_mel: {tts_mel.size()} tts_speech: {tts_speech.size()} tts_source: {tts_source.size()}.')
            return [tts_mel, tts_speech, tts_source]
            

class Request_Single_CosyFlowMatching(Request_Single):
    def __init__(self, uuid):
        super().__init__(uuid)
        self.input_para = []

    def set_input_para(self, input_papra):
        self.input_para.append(input_papra)

    def __str__(self):
        parent_info = super().__str__()
        return f'--> [ {ColorsLog.YELLOW}name{ColorsLog.RESET}: {self.name} , {parent_info}, {ColorsLog.YELLOW}input_para{ColorsLog.RESET}: {self.input_para} ]'


def main():
    # 1. init model
    config_file = './jtaudio/server/config.yaml'
    f5_tts_config = BaseModel_WorkerConfig()
    f5_tts_config.model_name = 'F5TTSModel'
    f5_tts_config.model_num = 2
    f5_tts_config.combine_batch_wait_time = 50 # ms
    f5_tts_config.max_batch_size = 32

    f5_tts_model_worker = F5TTSModelWorker(f5_tts_config, config_file)
    f5_tts_model_worker.start()

    # 2. init generate info
    ref_audio_00 = '/root/work/filestorage/usr/mayong/app/train/tts/cosyvoice/v1/CosyVoice_optimized_v2_tensorrt/zero_shot_prompt.wav'
    ref_text_00 = '希望你以后能够做的比我还好呦。'
    gen_text_00 = '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。'

    ref_audio_01 = '/root/work/filestorage/usr/mayong/app/train/tts/cosyvoice/v1/CosyVoice_optimized_v2_tensorrt/zero_shot_prompt.wav'
    ref_text_01 = '希望你以后能够做的比我还好呦。'
    gen_text_01 = '中国人民万岁!'
    
    f5_tts_model_worker.stop()


if __name__ == "__main__":
    main()
