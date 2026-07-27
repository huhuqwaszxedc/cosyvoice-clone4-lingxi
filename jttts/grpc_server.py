# Copyright (c) 2024 jiutian Inc 
#

import os
import signal
import sys, io, pdb
import asyncio
from concurrent import futures
import argparse
from typing import AsyncGenerator, Callable, Tuple, AsyncIterator, Union

import torch
import tts_interface_pb2
import tts_interface_pb2_grpc
import logging
import grpc
from grpc import aio
from jttts.config import GlobalConfigInst
from jttts.tts_common.logger import logger, logger_debug
from jttts.tts_common.request_processor import _perform_asr_quality_check, run_denoise_and_register_in_background_async, _validate_and_normalize_upload_prompt,  _prepare_voice_clone_request, truncate_large_values, convert_to_single_channel_audiobytes

import soundfile as sf
import json
import numpy as np
import time
from datetime import datetime
from functools import partial  # 用于处理带参数的函数
from collections import OrderedDict

from async_cosyvoice.runtime.async_grpc.utils import convert_audio_tensor_to_bytes, convert_audio_bytes_to_tensor
from jttts.tts_model.tts_utils import  get_aigc_metadata, generate_unique_id, audio_post_process


logging.basicConfig(level=logging.INFO, format='[ %(levelname)s %(asctime)s %(filename)s:%(lineno)d ] %(message)s')

flow_worker_stream  = None
flow_worker_offline = None
flow_worker_stream0 = None

class GrpcServiceImpl(tts_interface_pb2_grpc.cloneServicer):
    def __init__(self, inference_semaphore=None):
        self.tts_model = None
        if inference_semaphore is None:
            print(f'inference_semaphore is required')
            exit()
        self._inference_lock = inference_semaphore

    async def RegisterSpk(self, request: tts_interface_pb2.RegisterSpeakerRequest, context: aio.ServicerContext) -> tts_interface_pb2.Response:
        try:
            # 将 gRPC message 转为 dict（假设字段名一致）
            raw_dict = {
                "request_id":       getattr(request, "request_id", "null"),
                "user_id":          getattr(request, "user_id", None),
                "speaker_id":       getattr(request, "speaker_id", None),
                "prompt_num":       getattr(request, "prompt_num", None),
                "upload_prompt_text": getattr(request, "upload_prompt_text", "null"),
                "prompt_wav":       getattr(request, "prompt_wav", None),  
                "bypass_mode":      getattr(request, "bypass_mode", 0),
            }

            safe_log_dict = truncate_large_values(raw_dict, max_length=100, show_prefix=10)
            dict_str = json.dumps(safe_log_dict, ensure_ascii=False)
            logger.info(f"RegisterSpeaker [grpc] Start, {dict_str} ")

            # 注意：gRPC 的 prompt_wav 是 bytes，而 HTTP 是 base64 str
            is_valid, result = await asyncio.to_thread(
                _validate_and_normalize_upload_prompt, raw_dict
            )

            if not is_valid:
                error_msg = result
                logger.info(f"[grpc] RegisterSpk validation failed: {error_msg}")
                return tts_interface_pb2.Response(ResultStatus=400001, Msg=error_msg)

            request_id = result["request_id"]
            user_id = result["user_id"]
            prompt_text = result["prompt_text"]
            prompt_wav_bytes = result["prompt_wav"]  # 转回 bytes


            origin_wav_path = os.path.join(GlobalConfigInst.speaker_info_dir, "prompt_wavs", f"{user_id}.wav")
            os.makedirs(os.path.dirname(origin_wav_path), exist_ok=True)
            with open(origin_wav_path, "wb") as f:
                f.write(prompt_wav_bytes)

            prompt_wav_bytes, origin_sample_rate = convert_to_single_channel_audiobytes(origin_wav_path)

            # if origin_sample_rate <16000:
            #     msg = f'The register audio sample_rate is {origin_sample_rate}, and The sample_rate must >= 16k,'
            #     tts_interface_pb2.Response(ResultStatus=400007, Msg=msg)

            # === 统一 ASR 质检（在后台线程执行）===
            passed, msg, final_prompt_text = await asyncio.to_thread(
                _perform_asr_quality_check,
                prompt_wav_bytes=prompt_wav_bytes,
                prompt_text=prompt_text,
                request_id=request_id,
                user_id=user_id,
                bypass_mode=0  # gRPC 暂不支持 bypass_mode
            )

            if not passed:
                logger.info(f"RegisterSpk failed (ASR), request_id: {request_id}, user_id: {user_id}, reason: {msg}")
                return tts_interface_pb2.Response(ResultStatus=400007, Msg=msg)

            # 启动异步后台任务
            asyncio.create_task(
                run_denoise_and_register_in_background_async(
                    user_id=user_id,
                    prompt_wav_bytes=prompt_wav_bytes,
                    prompt_text=final_prompt_text,
                    prompt_num=10000,
                    bypass_mode=0,
                    request_id=request_id,
                    cosyvoice_frontend=self.tts_model.frontend if self.tts_model else None,
                    sample_rate_model=self.tts_model.sample_rate
                )
            )

            return tts_interface_pb2.Response(ResultStatus=0, Msg="RegisterSpk success")

        except Exception as e:
            logging.error(f"RegisterSpk error: {e}", exc_info=True)
            logger.info(f"RegisterSpk error: {e}",)
            return tts_interface_pb2.Response(ResultStatus=400001, Msg="RegisterSpk failed")




    async def Inference(self, request: tts_interface_pb2.TtsServiceRequest, context: aio.ServicerContext) -> AsyncIterator[
        tts_interface_pb2.Response]:
        """统一异步流式处理入口"""
        try:
            # 获取处理器和预处理后的参数
            req, processor, processor_args = await self._prepare_processor(request, request.text)

            # 通过通用处理器生成响应
            async for response in self._handle_generic(req, processor, processor_args):
                yield response

        except Exception as e:
            logging.error(f"Request processing failed: {str(e)}", exc_info=True)
            await context.abort(
                code=grpc.StatusCode.INTERNAL,
                details=f"Processing error: {str(e)}"
            )

    async def StreamInference(self, request_iterator, context: aio.ServicerContext) -> AsyncIterator[
        tts_interface_pb2.Response]:
        """异步双工流式处理入口，请不要在第一个 request 中包含 tts_text"""
        try:
            async def text_generator(request_iterator):
                async for request in request_iterator:
                    yield request.text

            try:
                # 使用第一个 request 中的参数，构建处理器参数
                first_request = await request_iterator.__anext__()
            except Exception as e:
                return
            # 从后续的请求中 构建 text_gen
            text_gen = text_generator(request_iterator)
            processor, processor_args = await self._prepare_processor(first_request, text_gen)

            # 通过通用处理器生成响应
            async for response in self._handle_generic(first_request, processor, processor_args):
                yield response
        except Exception as e:
            logging.error(f"Request processing failed: {str(e)}", exc_info=True)
            await context.abort(
                code=grpc.StatusCode.INTERNAL,
                details=f"Processing error: {str(e)}"
            )

    async def _check_request(self, request: tts_interface_pb2.TtsServiceRequest):
        
        if request.speed_ratio == 0.0:
            request.speed_ratio = GlobalConfigInst.default_speed
        
        if request.volume_ratio == 0.0:
            request.volume_ratio = 1.0

        if request.sample_rate == 0:
            request.sample_rate = GlobalConfigInst.default_sample_rate

        if request.encoding == '':
            request.encoding = GlobalConfigInst.default_encoding


        return request

    async def _prepare_processor(self, request: tts_interface_pb2.TtsServiceRequest, text: Union[str, AsyncGenerator]) -> Tuple[dict,Callable, list]:
        # log_str1 = f'[TTS_Service] request_id:{request.request_id}, text: {text}, user_id: {request.user_id}, speaker_id:{request.speaker_id}, '
        # log_str2 = f'if_ver={request.if_ver}, speed_ratio={request.speed_ratio}, volume_ratio={request.volume_ratio}, '
        # log_str3 = f'sample_rate={request.sample_rate}, streaming_mode={request.streaming_mode}, encoding={request.encoding}'
        # log_str = log_str1 + log_str2 + log_str3
        # logger.info(log_str)

        raw_dict_log = {
            "request_id":       getattr(request, "request_id", ""),
            "text":             getattr(request, "text", ""),
            "text_pinyins":     getattr(request, "text_pinyins", ""),
            "user_id":          getattr(request, "user_id", ""),
            "speaker_id":       getattr(request, "speaker_id", ""),
            "speed_ratio":      getattr(request, "speed_ratio", ""),
            "volume_ratio":     getattr(request, "volume_ratio", ""),
            "sample_rate":      getattr(request, "sample_rate", ""),
            "streaming_mode":   getattr(request, "streaming_mode", ""),
            "encoding":         getattr(request, "encoding", ""),
            "filter_bracket_content": getattr(request, "filter_bracket_content", False),
        }
        saved_id = ''
        if GlobalConfigInst.save_middle_result == 1:
            saved_id = generate_unique_id()
            raw_dict_log['saved_id'] = saved_id

        logger.info(f"TTSInfer [grpc] req:{raw_dict_log}", is_encrypt=True)

        # request_id = request.request_id
        request = await self._check_request(request)

        # 将 gRPC message 转为 dict（假设字段名一致）
        raw_dict = {
            "request_id":       getattr(request, "request_id", "null"),
            "text":             getattr(request, "text", ''),
            "text_pinyins":     getattr(request, "text_pinyins", ''),
            "user_id":          getattr(request, "user_id", ''),
            "speaker_id":       getattr(request, "speaker_id", ''),
            "speed_ratio":      getattr(request, "speed_ratio", GlobalConfigInst.default_speed),
            "volume_ratio":     getattr(request, "volume_ratio", 1.0),
            "sample_rate":      getattr(request, "sample_rate", GlobalConfigInst.default_sample_rate),
            "streaming_mode":   getattr(request, "streaming_mode", False),
            "Streaming_mode":   getattr(request, "Streaming_mode", False), 
            "encoding":         getattr(request, "encoding", GlobalConfigInst.default_encoding),
            "language":         getattr(request, "language", "auto"),
            "debug_mode":       getattr(request, "debug_mode", 0),
            "filter_bracket_content": getattr(request, "filter_bracket_content", False),
        }
        if len(saved_id)>0:
            raw_dict['saved_id'] = saved_id
        # 调用统一预处理函数（在后台线程）
        is_valid, result = await asyncio.to_thread(_prepare_voice_clone_request, raw_dict)

        if not is_valid:
            req = raw_dict
            error_msg = result
            logger.info(f"Synthesize validation failed: {error_msg}")
            return req, self.tts_model.inference_zero_shot_by_spk_id,  [f'{error_msg}']
        else:
            req = result


        text_frontend = True
        return req, self.tts_model.inference_zero_shot_by_spk_id, [
            req['text'],
            req['user_id'],
            req['streaming_mode'],
            req['speed_ratio'],
            text_frontend,
        ]

    async def _handle_generic(
            self,
            req: dict,
            processor: Callable,
            processor_args: list
    ) -> AsyncGenerator[tts_interface_pb2.Response, None]:
        """通用流式处理管道"""
        # logging.debug(f"Processing with {processor.__name__}")
        if len(processor_args) == 1:
            logger.info(f'{req["request_id"]}, {processor_args[0]}')
            yield tts_interface_pb2.Response(ResultStatus=400001, Msg=processor_args[0])
        else:
            if req['streaming_mode']: # 仅有TTS支持流式，Clone不支持流式
                async with self._inference_lock:
                    speech_token = []
                    if 'saved_id' in req:
                        is_save_speech_token = True
                    else:
                        is_save_speech_token = False

                    async for model_chunk in processor(*processor_args):
                        # import pdb;pdb.set_trace()
                        tts_status = model_chunk['tts_status']

                        if tts_status == -2:
                            yield tts_interface_pb2.Response(ResultStatus=420001, Msg='tts failed,tts_status=-2')
                        else:
                            sequence = model_chunk['chunk_index']
                            audio_data = model_chunk['tts_speech'] # audio_data.shape = [1, 15360]
                            audio_data = audio_data.squeeze()
                            audio_data = audio_data.numpy().astype(np.float32)
                            audio_data, sr = audio_post_process(audio_data, 24000,  req) # 输入audio_data是float，输出是int
                            audio_data = (audio_data / 32768).astype(np.float32)
                            audio_data = torch.Tensor(audio_data).to(model_chunk['tts_speech'].device)

                            aigc_metadata = get_aigc_metadata(protocol='grpc',)
                            aigc_metadata_str = json.dumps(aigc_metadata, ensure_ascii=False)

                            audio_bytes = await asyncio.to_thread(
                                convert_audio_tensor_to_bytes,
                                audio_data, 
                                req['encoding'],
                                req['sample_rate'],
                                aigc_metadata
                            )
                            
                            if is_save_speech_token:
                                speech_token_tmp = model_chunk['speech_token']
                                speech_token_tmp = speech_token_tmp.squeeze()
                                speech_token_tmp = speech_token_tmp.tolist()
                                speech_token.extend(speech_token_tmp)

                                if sequence == -1:
                                    json_data_dict = OrderedDict()
                                    json_data_dict['saved_id']      = req['saved_id']
                                    json_data_dict['speech_token']  = speech_token
                                    dump_line = json.dumps(json_data_dict, ensure_ascii=False)
                                    logger_debug.info(f"{dump_line}")                                    


                            yield tts_interface_pb2.Response(audio_data=audio_bytes, encoding=req['encoding'], ResultStatus=0, Msg='tts success', sample_rate=sr, sequence=sequence, AIGC=aigc_metadata_str)
            else:
                async with self._inference_lock:
                    # 服务端合并音频数据后，再编码返回一个完整的音频文件
                    audio_bytes: torch.Tensor = None
                    audio_data = None

                    speech_token = []
                    if 'saved_id' in req:
                        is_save_speech_token = True
                    else:
                        is_save_speech_token = False

                    model_chunk_device = None
                    async for model_chunk in processor(*processor_args):
                        tts_status = model_chunk['tts_status']
                        if tts_status == -2:
                            continue
                        else:
                            audio_data_tmp = model_chunk['tts_speech']
                            audio_data_tmp = audio_data_tmp.squeeze()
                            audio_data_tmp = audio_data_tmp.numpy().astype(np.float32)
                            model_chunk_device = model_chunk['tts_speech'].device
                            if audio_data is None:
                                audio_data = audio_data_tmp
                            else:
                                audio_data = np.concatenate([audio_data, audio_data_tmp], axis=0)

                            if is_save_speech_token:
                                speech_token_tmp = model_chunk['speech_token']
                                speech_token_tmp = speech_token_tmp.squeeze()
                                speech_token_tmp = speech_token_tmp.tolist()
                                speech_token.extend(speech_token_tmp)



                    if audio_data is None:
                        yield tts_interface_pb2.Response(ResultStatus=420001, Msg='tts failed,audio_data is None')

                    audio_data, sr = audio_post_process(audio_data, 24000,  req) # 输入audio_data是float，输出是int
                    audio_data = (audio_data / 32768).astype(np.float32)

                    aigc_metadata = get_aigc_metadata(protocol='grpc',)
                    aigc_metadata_str = json.dumps(aigc_metadata, ensure_ascii=False)
                    #audio_data = torch.Tensor(audio_data).to(model_chunk_device)
                    audio_data = torch.Tensor(audio_data)
                    audio_bytes = await asyncio.to_thread(
                        convert_audio_tensor_to_bytes,
                        audio_data, 
                        req['encoding'],
                        req['sample_rate'],
                        aigc_metadata 
                    )

                    if is_save_speech_token:
                        json_data_dict = OrderedDict()
                        json_data_dict['saved_id']      = req['saved_id']
                        json_data_dict['speech_token']  = speech_token
                        dump_line = json.dumps(json_data_dict, ensure_ascii=False)
                        logger_debug.info(f"{dump_line}")     

                    yield tts_interface_pb2.Response(audio_data=audio_bytes, encoding=req['encoding'], ResultStatus=0, Msg='tts success', sample_rate=24000, sequence=-1, AIGC=aigc_metadata_str)


if __name__ == '__main__':
    pass
