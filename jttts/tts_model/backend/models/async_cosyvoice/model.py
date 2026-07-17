# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import logging
import os
import queue
import threading
from typing import AsyncGenerator, Generator, List, Union
import torch
import numpy as np
import time
from torch.nn import functional as F
from contextlib import nullcontext
import uuid
from concurrent.futures import ThreadPoolExecutor

from cosyvoice.flow.flow import CausalMaskedDiffWithDiT
from cosyvoice.hifigan.generator import CausalHiFTGenerator
from cosyvoice.utils.common import fade_in_out
from cosyvoice.utils.file_utils import convert_onnx_to_trt
from cosyvoice.flow.flow_matching import EstimatorWrapper
from cosyvoice.utils.common import TrtContextWrapper


import pdb
# 启用vllm V1版本
os.environ["VLLM_USE_V1"] = '1'
from vllm import  AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams

from vllm import ModelRegistry
from async_cosyvoice.config import ENGINE_ARGS, SAMPLING_PARAMS, ESTIMATOR_COUNT

from async_cosyvoice.vllm_use_cosyvoice_model import CosyVoice3Model as CosyVoice3LLM
ModelRegistry.register_model("CosyVoice3Model", CosyVoice3LLM)

from jtservers.core.request import Request_Single
from jtservers.core.colorslog import ColorsLog
from async_cosyvoice.flowworker import Request_Single_CosyFlowMatching
from jttts.config import GlobalConfigInst

async def get_request(cosy_worker, request):
    loop = asyncio.get_running_loop()
    result_request = await loop.run_in_executor(None, cosy_worker.wait_complete, request)
    return result_request

class AsyncWrapper:
    """将一个同步生成器包装为异步生成器"""
    def __init__(self, obj):
        self.obj = obj

    async def __aiter__(self):
        for item in self.obj:
            yield item

def tensor_to_list(tensor: torch.tensor):
    return tensor.view(-1).cpu().numpy().tolist()


class CosyVoice3Model:

    def __init__(self,
         model_dir: str,
         flow: CausalMaskedDiffWithDiT | torch.nn.Module,
         hift: CausalHiFTGenerator | torch.nn.Module,
         fp16: bool,
         mix_ratio: List[int] = None,
         flow_worker_stream=None, 
         flow_worker_offline=None
    ):
        self.flow_worker_stream = flow_worker_stream
        self.flow_worker_offline = flow_worker_offline
        # pdb.set_trace()
        # vllm engine 的参数配置
        vllm_dir = os.path.join(model_dir, 'vllm')
        vllm_dir = os.path.abspath(vllm_dir)
        logging.info(f'vllm_dir = {vllm_dir}')
        engine_args = AsyncEngineArgs(
            model= vllm_dir,
            **ENGINE_ARGS,
        )
        self.llm_engine: AsyncLLMEngine = AsyncLLMEngine.from_engine_args(engine_args)
        # self.device = torch.device('cuda:0')

        # 这里只决定hifigan占用的卡
        if GlobalConfigInst.infer_speed_mode == 0:  # flow加速的情况
            if GlobalConfigInst.tts_gpu_num == 3:
                self.device = torch.device('cuda:2')
            elif GlobalConfigInst.tts_gpu_num == 2:
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cuda:0')
        else:               # flow 不加速
            if GlobalConfigInst.tts_gpu_num == 3:# flow不加速情况，设置3卡意义不大
                self.device = torch.device('cuda:1') #
            elif GlobalConfigInst.tts_gpu_num == 2:
                self.device = torch.device('cuda:1')
            else:
                self.device = torch.device('cuda:0')

        logging.info(f'tts_gpu_num={GlobalConfigInst.tts_gpu_num}, self.device = {self.device}')

        self.thread_count = 4
        self.thread_executor = ThreadPoolExecutor(max_workers=self.thread_count)
        self.stream_pool = queue.Queue()
        for _ in range(self.thread_count):
            cuda_stream = torch.cuda.Stream(self.device)
            self.stream_pool.put(cuda_stream)

        
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        self.flow.fp16 = fp16
        if self.fp16 is True:
            self.flow.half()
        self.token_hop_len = 2 * self.flow.input_frame_rate
        # here we fix flow encoder/decoder decoding_chunk_size, in the future we will send it as arguments, or use cache
        # self.flow.encoder.static_chunk_size = 2 * self.flow.input_frame_rate
        self.flow.decoder.estimator.static_chunk_size = 2 * self.flow.input_frame_rate * self.flow.token_mel_ratio
        # hift cache
        self.mel_cache_len = 8
        self.source_cache_len = int(self.mel_cache_len * 480)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)

        self.mix_ratio = mix_ratio or [5, 15]

        self.lock = asyncio.Lock()  # 改为异步锁

        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.hift_cache_dict = {}

        # 与vllm中的模型保持一致
        self.speech_token_size = 6561
        self.llm_token_size = 151936  

        self.extended_speech_vocab_size = self.speech_token_size + 200 
        self.text_token_offset = self.extended_speech_vocab_size
        # self.sos_eos_token_id = self.speech_token_size + 1
        # self.task_token_id = self.sos_eos_token_id + 1
        # self.zero_token_id = self.task_token_id + 1

        # 特殊 token 定义（全部在 extended 语音空间内）
        self.sos_token_id = self.speech_token_size + 0
        self.eos_token_id = self.speech_token_size + 1
        self.task_token_id = self.speech_token_size + 2
        self.fill_token_id = self.speech_token_size + 3


        self.stop_token_ids = list(range(self.speech_token_size, self.extended_speech_vocab_size))


        # vllm 的推理任务需要在一个固定的事件循环中，因此启动一个后台线程专用于推理任务
        self.background_loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.loop_thread.start()

    def _run_event_loop(self):
        asyncio.set_event_loop(self.background_loop)
        self.background_loop.run_forever()

    def load(self, flow_model, hift_model):
        # self.flow.load_state_dict(torch.load(flow_model, weights_only=True, map_location=self.device), strict=True)
        # 在flow加速情况下，此处load后只是占用显存，并不会去推理
        # 在flow不加速情况下，此处load后，会去做推理
        flow_model = torch.load(flow_model, weights_only=False, map_location=self.device)
        flow_model.pop('epoch', None)
        flow_model.pop('step', None)
        self.flow.load_state_dict(flow_model, strict=True)
        self.flow.to(self.device).eval()

        # self.flow.to(self.device).eval()
        # in case hift_model is a hifigan model
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(hift_model, weights_only=True, map_location=self.device).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

    def load_jit(self, flow_encoder_model):
        flow_encoder = torch.jit.load(flow_encoder_model, map_location=self.device)
        self.flow.encoder = flow_encoder



    def get_trt_kwargs(self):
        min_shape = [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4)]
        opt_shape = [(2, 80, 500), (2, 1, 500), (2, 80, 500), (2, 80, 500)]
        max_shape = [(2, 80, 3000), (2, 1, 3000), (2, 80, 3000), (2, 80, 3000)]
        input_names = ["x", "mask", "mu", "cond"]
        return {'min_shape': min_shape, 'opt_shape': opt_shape, 'max_shape': max_shape, 'input_names': input_names}

    def load_onnx_and_convert_trt(self, flow_decoder_estimator_model, flow_decoder_onnx_model, fp16):
        assert torch.cuda.is_available(), 'tensorrt only supports gpu!'
        if not os.path.exists(flow_decoder_estimator_model):
            convert_onnx_to_trt(flow_decoder_estimator_model, self.get_trt_kwargs(), flow_decoder_onnx_model, fp16)
        if os.path.getsize(flow_decoder_estimator_model) == 0:
            raise ValueError('{} is empty file, delete it and export again!'.format(flow_decoder_estimator_model))
        del self.flow.decoder.estimator
        import tensorrt as trt
        with open(flow_decoder_estimator_model, 'rb') as f:
            estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
        assert estimator_engine is not None, 'failed to load trt {}'.format(flow_decoder_estimator_model)
        trt_concurrent=ESTIMATOR_COUNT
        self.flow.decoder.estimator = TrtContextWrapper(estimator_engine, trt_concurrent=trt_concurrent, device=self.device)
        logging.info(f'load trt model success !')


    def load_trt(self, flow_decoder_estimator_model):
        assert torch.cuda.is_available(), 'tensorrt only supports gpu!'
        if not os.path.exists(flow_decoder_estimator_model):
            logging.error(f'flow estimator model is not exists !, {flow_decoder_estimator_model}')
            exit()
        if os.path.getsize(flow_decoder_estimator_model) == 0:
            raise ValueError('{} is empty file, delete it and export again!'.format(flow_decoder_estimator_model))
        del self.flow.decoder.estimator
        import tensorrt as trt

        with open(flow_decoder_estimator_model, 'rb') as f:
            estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
        assert estimator_engine is not None, 'failed to load trt {}'.format(flow_decoder_estimator_model)
        trt_concurrent = ESTIMATOR_COUNT
        self.flow.decoder.estimator = TrtContextWrapper(estimator_engine, trt_concurrent=trt_concurrent, device=self.device)
        logging.info(f'load trt model success !')



    async def background_llm_inference(self, out_queue, prompt_token_ids, request_id, stop_token_ids, max_tokens):
        sampling_params = SamplingParams(**SAMPLING_PARAMS)
        sampling_params.stop_token_ids = stop_token_ids # or [6561, 6563]
        if max_tokens:
            sampling_params.max_tokens = max_tokens
        async for output in self.llm_engine.generate(
                {
                    "prompt_token_ids": prompt_token_ids,
                },
                sampling_params=sampling_params,
                request_id=request_id or f"{time.time()}",
        ):
            out_queue.put((output.outputs[0], output.finished))

    # async def llm_inference(self, prompt_token_ids: List[int], request_id: str=None, stop_token_ids=None, max_tokens=None):
    #     out_queue = queue.Queue()
    #     asyncio.run_coroutine_threadsafe(
    #         self.background_llm_inference(out_queue, prompt_token_ids, request_id, stop_token_ids, max_tokens), self.background_loop
    #     )
    #     # 接收 out_queue 返回的结果
    #     finished = False
    #     while not finished:
    #         if not out_queue.empty():
    #             (output, finished) = out_queue.get_nowait()
    #             yield output
    #         else:
    #             # 主动等待5ms
    #             await asyncio.sleep(0.005)

    async def llm_inference(self, prompt_token_ids: List[int], request_id: str=None, stop_token_ids=None, max_tokens=None):
        # empty_token_list = [4299, 4218, 3732, 4056, 6486] 
        empty_token_list = []
        consecutive_empty_count = 0 
        filter_subsequent = False
        empty_token_len = 4

        out_queue = queue.Queue()
        asyncio.run_coroutine_threadsafe(
            self.background_llm_inference(out_queue, prompt_token_ids, request_id, stop_token_ids, max_tokens), self.background_loop
        )
        # 接收 out_queue 返回的结果
        finished = False
        while not finished:
            if not out_queue.empty():
                (output, finished) = out_queue.get_nowait()
                # logging.info(f'********* output.token_ids: {output.token_ids}')

                if not (hasattr(output, 'token_ids') and output.token_ids):
                    yield output
                    continue

                filtered_tokens = []
                current_tokens = output.token_ids

                for token in current_tokens:
                    if filter_subsequent and token in empty_token_list:
                        continue
                    if token not in empty_token_list:
                        consecutive_empty_count = 0
                        filter_subsequent = False
                        filtered_tokens.append(token)
                    else:
                        consecutive_empty_count += 1
                        if consecutive_empty_count <= empty_token_len:
                            filtered_tokens.append(token)
                        else:
                            filter_subsequent = True
                output.token_ids = filtered_tokens
                if len(filtered_tokens) != 0: # 单请求的情况下len(filtered_tokens) =1
                    yield output
            else:
                # 主动等待5ms
                await asyncio.sleep(0.005)

    async def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
        prompt_text = tensor_to_list(prompt_text + torch.tensor(self.text_token_offset))
        llm_prompt_speech_token = tensor_to_list(llm_prompt_speech_token)

        start_time = time.time()
        if isinstance(text, Union[Generator,AsyncGenerator]):
            if isinstance(text, Generator):
                text = AsyncWrapper(text)
            last_tokens = []
            prompt_token_ids = [self.sos_token_id]
            text_tokens_cache = prompt_text
            async for this_text in text:
                this_text = tensor_to_list(this_text + torch.tensor(self.text_token_offset))
                # text need tokens
                text_tokens_cache += this_text
                while len(llm_prompt_speech_token) != 0:
                    if len(text_tokens_cache) >= self.mix_ratio[0]:
                        text_input_token = text_tokens_cache[:self.mix_ratio[0]]
                        speech_input_token = llm_prompt_speech_token[:self.mix_ratio[1]]
                        prompt_token_ids += text_input_token + speech_input_token
                        # reset the last cache
                        text_tokens_cache = text_tokens_cache[self.mix_ratio[0]:]
                        llm_prompt_speech_token = llm_prompt_speech_token[self.mix_ratio[1]:]
                    else:
                        break
                if len(llm_prompt_speech_token) == 0:
                    if (len(last_tokens) > 0 and last_tokens[-1] == self.eos_token_id) or len(prompt_token_ids) == 1:
                        if len(text_tokens_cache) >= self.mix_ratio[0]:
                            text_tokens_temp = text_tokens_cache[:self.mix_ratio[0]]
                            prompt_token_ids += text_tokens_temp
                            text_tokens_cache = text_tokens_cache[self.mix_ratio[0]:]
                        else:
                            continue
                    async for output in self.llm_inference(prompt_token_ids, request_id=uuid, stop_token_ids=self.stop_token_ids):
                        last_tokens = output.token_ids
                        if last_tokens[-1] >= self.speech_token_size:
                            need_add_tokens = last_tokens[:-1]
                        else:
                            need_add_tokens = last_tokens
                        self.tts_speech_token_dict[uuid].extend(need_add_tokens)
                        prompt_token_ids.extend(need_add_tokens)

            prompt_token_ids += text_tokens_cache + [self.task_token_id]
            async for output in self.llm_inference(prompt_token_ids, request_id=uuid, stop_token_ids=self.stop_token_ids):
                if output.token_ids[-1] >= self.speech_token_size:
                    need_add_tokens = last_tokens[:-1]
                else:
                    need_add_tokens = output.token_ids
                self.tts_speech_token_dict[uuid].extend(need_add_tokens)
        else:
            text = tensor_to_list(text + torch.tensor(self.text_token_offset))
            prompt_token_ids = [self.sos_token_id] + prompt_text + text + \
                               [self.task_token_id] + llm_prompt_speech_token
            max_tokens = len(text) * 20
            async for output in self.llm_inference(
                    prompt_token_ids,
                    request_id=uuid,
                    stop_token_ids=self.stop_token_ids,
                    max_tokens=max_tokens,
            ):
                if output.token_ids[-1] >= self.eos_token_id: # 单请求下len(output.token_ids)=1
                    need_add_tokens = output.token_ids[:-1]
                else:
                    need_add_tokens = output.token_ids
                self.tts_speech_token_dict[uuid].extend(need_add_tokens)

        # self.tts_speech_token_dict[uuid].extend([4299]) #  用于解决 6s 的token 即150token左右时，尾巴容易出爆破音的问题。
        self.tts_speech_token_dict[uuid]
        self.llm_end_dict[uuid] = True
        # logging.info(f'llm job done, generated {len(self.tts_speech_token_dict[uuid]):>4} tokens, time cost: {time.time() - start_time:.3f}s')
        # logging.debug(f'speech_tokens: len: {len(self.tts_speech_token_dict[uuid])}  data: {self.tts_speech_token_dict[uuid]}')
        # 记录 prompt_token_ids self.tts_speech_token_dict[uuid] 数据用于后续的训练，与flow推理测试


    def token2wav(self, token, prompt_token, prompt_feat, embedding, uuid, token_offset, stream=False, finalize=False, speed=1.0, token_len=-1, chunk_index = -1):
        torch.cuda.current_stream().synchronize() # 将当前流进行同步了再处理后续逻辑
        cuda_stream = self.stream_pool.get()
        with torch.cuda.stream(cuda_stream):
            tts_mel, _ = self.flow.inference(token=token.to(self.device),
                                             token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                             prompt_token=prompt_token.to(self.device),
                                             prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                             prompt_feat=prompt_feat.to(self.device),
                                             prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                             embedding=embedding.to(self.device),
                                             streaming=stream,
                                             finalize=finalize)

            tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]
            # append mel cache
            if (uuid in self.hift_cache_dict ) and (self.hift_cache_dict[uuid] is not None):
                hift_cache_mel = self.hift_cache_dict[uuid]['mel']
                tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
                self.hift_cache_dict[uuid]['mel'] = tts_mel
            else:
                self.hift_cache_dict[uuid] = {'mel': tts_mel, 'speech_offset': 0}




            # if speed != 1.0:
            #     assert token_offset == 0 and finalize is True, 'speed change only support non-stream inference mode'
            #     tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')


            tts_speech, _ = self.hift.inference(speech_feat=tts_mel, finalize=finalize)
            tts_speech = tts_speech[:, self.hift_cache_dict[uuid]['speech_offset']:]
            self.hift_cache_dict[uuid]['speech_offset'] += tts_speech.shape[1]

            torch.cuda.synchronize(torch.cuda.current_stream())
            self.stream_pool.put(cuda_stream)
            return tts_speech


    # def token2wav_flow_speed(self, token, prompt_token, prompt_feat, embedding, uuid, token_offset, stream=False, finalize=False, speed=1.0, token_len=-1, chunk_index = -1):
        # # 1. flow worker
        # request_flow_worker = Request_Single_CosyFlowMatching("test_" + str(uuid))
        # token, token_len, prompt_token, prompt_feat, embedding = token.cpu(), token_len.cpu(), prompt_token.cpu(), prompt_feat.cpu(), embedding.cpu()
        # token.share_memory_()
        # token_len.share_memory_()
        # prompt_token.share_memory_()
        # prompt_feat.share_memory_()
        # embedding.share_memory_()
        # finalize = torch.as_tensor([finalize], dtype=torch.bool).cpu()
        # finalize.share_memory_()
        # if finalize.item() is True:
        #     input_list = [token, token_len, embedding, finalize]
        #     request_flow_worker.set_input_para(['token', 'token_len', 'embedding', 'finalizes'])
        # elif chunk_index == 0:
        #     input_list = [token, token_len, embedding, finalize]
        #     request_flow_worker.set_input_para(['token', 'token_len', 'embedding', 'finalizes'])
        # else:
        #     # logging.info(f'token_len: {token_len}')
        #     token_offset_tensor = torch.tensor([token_offset], dtype = torch.int32).cpu()
        #     hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
        #     hift_cache_mel.cpu()
        #     hift_cache_source.cpu()
        #     hift_cache_mel.share_memory_()
        #     hift_cache_source.share_memory_()
        #     token_offset_tensor.share_memory_()
        #     input_list = [token, token_len, embedding, finalize, hift_cache_mel, hift_cache_source, token_offset_tensor]
        #     request_flow_worker.set_input_para(['token', 'token_len', 'embedding', 'finalizes', 'hift_cache_mel', 'hift_cache_source', 'token_offset_tensor'])

        # request_flow_worker.set_input_tensor_list(input_list)
        # request_flow_worker.set_duration(token_len.item())

        # if finalize.item() is True:
        #     self.flow_worker_offline.add_request(request_flow_worker)
        #     request_flow_worker = await get_request(self.flow_worker_offline, request_flow_worker)
        #     tts_mel = request_flow_worker.get_batch_result()[0]
        #     mel_len = request_flow_worker.get_batch_result()[1]
        #     tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:mel_len - self.flow.pre_lookahead_len]

        #     if self.hift_cache_dict[uuid] is not None:
        #         hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
        #         tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2).to(self.device)
        #         hift_cache_source = hift_cache_source.to(self.device)
        #     else:
        #         tts_mel = tts_mel.to(self.device)
        #         hift_cache_source = torch.zeros(1, 1, 0).to(self.device)

        #     if speed != 1.0:
        #         assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
        #         tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')

        #     tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
        #     if self.hift_cache_dict[uuid] is not None:
        #         tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        # else:
        #     if chunk_index == 0 or chunk_index == 1:
        #         flow_worker_stream = self.flow_worker_stream[0]
        #     else:
        #         flow_worker_stream = self.flow_worker_stream[-1]
        #     flow_worker_stream.add_request(request_flow_worker)
        #     request_flow_worker = await get_request(flow_worker_stream, request_flow_worker)
        #     tts_mel = request_flow_worker.get_batch_result()[0]
            
        #     tts_speech = request_flow_worker.get_batch_result()[1]
        #     tts_source = request_flow_worker.get_batch_result()[2]

        #     if self.hift_cache_dict[uuid] is not None:
        #         tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        #     self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
        #                                     'source': tts_source[:, :, -self.source_cache_len:],
        #                                     'speech': tts_speech[:, -self.source_cache_len:]}
        #     tts_speech = tts_speech[:, :-self.source_cache_len]
        
        # return tts_speech

    # def token2wav(self, token, prompt_token, prompt_feat, embedding, uuid, token_offset, stream=False, finalize=False, speed=1.0, token_len=-1, chunk_index = -1):
        # if GlobalConfigInst.infer_speed_mode == 0:
        #     return self.token2wav_flow_speed(token=token, 
        #                                            prompt_token=prompt_token, 
        #                                            prompt_feat=prompt_feat, 
        #                                            embedding=embedding, 
        #                                            uuid=uuid, 
        #                                            token_offset=token_offset, 
        #                                            stream=stream,
        #                                            finalize=finalize, 
        #                                            speed=speed, 
        #                                            token_len=token_len, 
        #                                            chunk_index = chunk_index)

        # else:


        # return self.token2wav_origin(token=token, 
        #                                         prompt_token=prompt_token, 
        #                                         prompt_feat=prompt_feat, 
        #                                         embedding=embedding, 
        #                                         uuid=uuid, 
        #                                         token_offset=token_offset, 
        #                                         stream=stream,
        #                                         finalize=finalize, 
        #                                         speed=speed, 
        #                                         token_len=token_len, 
        #                                         chunk_index = chunk_index)
    





    async def async_tts(self, text, flow_embedding, llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80), stream=False, speed=1.0, **kwargs):
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        loop_index = 0 
        loop_gen = True
        chunk_index = 0
        while loop_gen and loop_index <= 3:
            # logging.info(f'start loop_gen: {loop_gen}')
            async with self.lock:
                self.tts_speech_token_dict[this_uuid] = []
                self.llm_end_dict[this_uuid] = False
                self.hift_cache_dict[this_uuid] = None
            # queue: asyncio.Queue[int|None] = asyncio.Queue()
            llm_task = asyncio.create_task(self.llm_job(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
            if stream is True:
                token_offset = 0
                # peer_chunk_token_num = 20     # 设置初始的每个chunk处理语音token的数量
                peer_chunk_token_num = self.flow.peer_chunk_token_num
                await asyncio.sleep(0.05)
                loop = asyncio.get_event_loop()
                start_time = time.time()
                chunk_index = 0
                while True:
                    if (pending_num:= len(self.tts_speech_token_dict[this_uuid]) - token_offset) >= (peer_chunk_token_num + self.flow.pre_lookahead_len):
                        this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_offset + peer_chunk_token_num + self.flow.pre_lookahead_len]).unsqueeze(dim=0)
                        this_tts_speech_token_len = torch.tensor([this_tts_speech_token.size(1)], dtype=torch.int32)

                        this_tts_speech = await loop.run_in_executor(self.thread_executor,
                                self.token2wav,
                                this_tts_speech_token,
                                flow_prompt_speech_token,
                                prompt_speech_feat,
                                flow_embedding,
                                this_uuid,
                                token_offset,
                                stream,
                                False,
                                this_tts_speech_token_len,
                                chunk_index
                        )


                        # this_tts_speech = await self.token2wav(
                        #         token        = this_tts_speech_token,
                        #         prompt_token = flow_prompt_speech_token,
                        #         prompt_feat  = prompt_speech_feat,
                        #         embedding    = flow_embedding,
                        #         uuid         = this_uuid,
                        #         token_offset = token_offset,
                        #         stream       = stream,
                        #         finalize     = False,
                        #         token_len    = this_tts_speech_token_len,
                        #         chunk_index  = chunk_index
                        # )

                        token_offset += peer_chunk_token_num
                        yield {'tts_speech': this_tts_speech.cpu(), 'tts_status': 0, 'chunk_index': chunk_index, 'speech_token': this_tts_speech_token.cpu()}
                        chunk_index += 1
                        cost_time = time.time() - start_time
                        
                        if chunk_index == 10: # 先改10，应该可以解决部分不能实时的问题
                            break

                    else:
                        if self.llm_end_dict[this_uuid] is True:
                            break
                        else:
                            await asyncio.sleep(0.02)
                await llm_task
                chunk_index = -1
                if len(self.tts_speech_token_dict[this_uuid]) <= 15:
                    # logging.info(f'-- len(self.tts_speech_token_dict[this_uuid]) <= 20')
                    loop_index = loop_index + 1
                    if loop_index == 4:
                        yield {'tts_speech': '', 'tts_status': -2, 'chunk_index': chunk_index, 'speech_token': ''}
                else:
                    # logging.info(f'-- loop_gen = False')
                    loop_gen = False
                    # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
                    this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
                    this_tts_speech_token_len = torch.tensor([this_tts_speech_token.size(1)], dtype=torch.int32)

                    this_tts_speech = await loop.run_in_executor(self.thread_executor,
                        self.token2wav,
                            this_tts_speech_token,
                            flow_prompt_speech_token,
                            prompt_speech_feat,
                            flow_embedding,
                            this_uuid,
                            token_offset,
                            stream,
                            True,
                            this_tts_speech_token_len,
                            chunk_index
                    )


                    # this_tts_speech = await self.token2wav(
                    #                     token=this_tts_speech_token,
                    #                     prompt_token=flow_prompt_speech_token,
                    #                     prompt_feat=prompt_speech_feat,
                    #                     embedding=flow_embedding,
                    #                     uuid=this_uuid,
                    #                     token_offset=token_offset,
                    #                     stream=stream,
                    #                     finalize=True,
                    #                     token_len=this_tts_speech_token_len,
                    #                     chunk_index=chunk_index
                    #                 )
                    if this_tts_speech.shape[1] == 0:
                        #logging.debug(f'no tts_speech_token shape: {this_tts_speech.shape}, data: {this_tts_speech}')
                        yield {'tts_speech': '', 'tts_status': -2, 'chunk_index': chunk_index, 'speech_token': ''}
                        # return
                    else:
                        yield {'tts_speech': this_tts_speech.cpu(), 'tts_status': 0, 'chunk_index': chunk_index, 'speech_token': this_tts_speech_token.cpu()}
            else: # stream=False
                # deal with all tokens
                await llm_task
                if len(self.tts_speech_token_dict[this_uuid]) <= 15:
                    loop_index = loop_index + 1
                    if loop_index == 4:
                        yield {'tts_speech': '', 'tts_status': -2, 'chunk_index': chunk_index, 'speech_token': ''}
                        break
                else:
                    chunk_index = -1
                    this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
                    this_tts_speech_token_len = torch.tensor([this_tts_speech_token.size(1)], dtype=torch.int32)
                    loop = asyncio.get_event_loop()

                    this_tts_speech = await loop.run_in_executor(self.thread_executor,
                                                                self.token2wav,
                                                                this_tts_speech_token,
                                                                flow_prompt_speech_token,
                                                                prompt_speech_feat,
                                                                flow_embedding,
                                                                this_uuid,
                                                                0,
                                                                False,
                                                                True,
                                                                speed,
                                                                this_tts_speech_token_len,
                                                                chunk_index
                                                                )

                    # this_tts_speech = await self.token2wav(
                    #                     token=this_tts_speech_token,
                    #                     prompt_token=flow_prompt_speech_token,
                    #                     prompt_feat=prompt_speech_feat,
                    #                     embedding=flow_embedding,
                    #                     uuid=this_uuid,
                    #                     token_offset=0,
                    #                     stream=False,
                    #                     finalize=True,
                    #                     speed=speed,
                    #                     token_len=this_tts_speech_token_len,
                    #                     chunk_index=chunk_index
                    #                 )
                    yield {'tts_speech': this_tts_speech.cpu(), 'tts_status': 0, 'chunk_index': chunk_index, 'speech_token': this_tts_speech_token.cpu()}
                    break
            async with self.lock:
                self.tts_speech_token_dict.pop(this_uuid)
                self.llm_end_dict.pop(this_uuid)
                self.hift_cache_dict.pop(this_uuid)





    def opensource_origin_tts(self, text=torch.zeros(1, 0, dtype=torch.int32), flow_embedding=torch.zeros(0, 192), llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80), source_speech_token=torch.zeros(1, 0, dtype=torch.int32), stream=False, speed=1.0, **kwargs):
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
            self.hift_cache_dict[this_uuid] = None
        if source_speech_token.shape[1] == 0:
            p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        else:
            p = threading.Thread(target=self.vc_job, args=(source_speech_token, this_uuid))
        p.start()
        if stream is True:
            token_offset = 0
            prompt_token_pad = int(np.ceil(flow_prompt_speech_token.shape[1] / self.token_hop_len) * self.token_hop_len - flow_prompt_speech_token.shape[1])
            while True:
                time.sleep(0.1)
                this_token_hop_len = self.token_hop_len + prompt_token_pad if token_offset == 0 else self.token_hop_len
                if len(self.tts_speech_token_dict[this_uuid]) - token_offset >= this_token_hop_len + self.flow.pre_lookahead_len:
                    this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_offset + this_token_hop_len + self.flow.pre_lookahead_len]).unsqueeze(dim=0)
                    this_tts_speech = self.opensource_origin_token2wav(token=this_tts_speech_token,
                                                     prompt_token=flow_prompt_speech_token,
                                                     prompt_feat=prompt_speech_feat,
                                                     embedding=flow_embedding,
                                                     token_offset=token_offset,
                                                     uuid=this_uuid,
                                                     stream=stream,
                                                     finalize=False)
                    token_offset += this_token_hop_len
                    yield {'tts_speech': this_tts_speech.cpu()}
                if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) - token_offset < this_token_hop_len + self.flow.pre_lookahead_len:
                    break
            p.join()
            # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.opensource_origin_token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             token_offset=token_offset,
                                             uuid=this_uuid,
                                             finalize=True)
            yield {'tts_speech': this_tts_speech.cpu()}
        else:
            # deal with all tokens
            p.join()
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             token_offset=0,
                                             uuid=this_uuid,
                                             finalize=True,
                                             speed=speed)
            yield {'tts_speech': this_tts_speech.cpu()}
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.current_stream().synchronize()


    def opensource_origin_token2wav(self, token, prompt_token, prompt_feat, embedding, token_offset, uuid, stream=False, finalize=False, speed=1.0):
        with torch.cuda.amp.autocast(self.fp16):
            tts_mel, _ = self.flow.inference(token=token.to(self.device, dtype=torch.int32),
                                             token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                             prompt_token=prompt_token.to(self.device),
                                             prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                             prompt_feat=prompt_feat.to(self.device),
                                             prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                             embedding=embedding.to(self.device),
                                             streaming=stream,
                                             finalize=finalize)
            tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]
            # append mel cache
            if self.hift_cache_dict[uuid] is not None:
                hift_cache_mel = self.hift_cache_dict[uuid]['mel']
                tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
                self.hift_cache_dict[uuid]['mel'] = tts_mel
            else:
                self.hift_cache_dict[uuid] = {'mel': tts_mel, 'speech_offset': 0}
            if speed != 1.0:
                assert token_offset == 0 and finalize is True, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, _ = self.hift.inference(speech_feat=tts_mel, finalize=finalize)
            tts_speech = tts_speech[:, self.hift_cache_dict[uuid]['speech_offset']:]
            self.hift_cache_dict[uuid]['speech_offset'] += tts_speech.shape[1]
        return tts_speech


