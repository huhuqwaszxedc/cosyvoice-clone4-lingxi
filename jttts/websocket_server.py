# websocket_server.py

import os
import sys
import pdb
import threading
import time
import uuid
import re
import io
import json
import asyncio
from datetime import datetime
from functools import partial
from collections import OrderedDict
from threading import Lock
from fastapi.middleware.cors import CORSMiddleware
import base64
from io import BytesIO
import struct
import torch
import librosa
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from jttts.tts_common.logger import logger, logger_debug


from jttts.tts_common.request_processor import (
    _perform_asr_quality_check,
    run_denoise_and_register_in_background,
    run_denoise_and_register_in_background_async,
    _validate_and_normalize_upload_prompt,
    _prepare_voice_clone_request,
    truncate_large_values,
    convert_to_single_channel_audiobytes
)
from jttts.config import GlobalConfigInst
from jttts.tts_model.tts_utils import pack_audio, audio_post_process, get_aigc_metadata, generate_unique_id

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ******************************************************************************************************************
@app.websocket("/jttts/Upload_Prompt")
@app.websocket("/jttts/RegisterSpeaker")
async def register_speaker_websocket(ws: WebSocket):
    await ws.accept()

    # 存储等待音频的请求上下文
    pending_requests = {}

    try:
        while True:

            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                # logger.info("Client disconnected during registration.")
                break
            request_id = None

            if msg["type"] == "websocket.receive":
                if "text" in msg:
                    payload = msg["text"]
                    try:
                        raw_dict = json.loads(payload)
                        safe_log_dict = truncate_large_values(raw_dict, max_length=100, show_prefix=10)
                        dict_str = json.dumps(safe_log_dict, ensure_ascii=False)
                        request_id = raw_dict.get("request_id", "null")

                        logger.info(f"RegisterSpeaker [ws] Received control: {dict_str}")

                        raw_dict['prompt_wav'] =  b'\x00'
                        is_valid, result = _validate_and_normalize_upload_prompt(raw_dict)
                        if not is_valid:
                            error_msg = result
                            logger.info(f"[ws] RegisterSpeaker validation failed: {error_msg}, request_id: {request_id}")
                            await ws.send_text(json.dumps({
                                "request_id": request_id,
                                "ResultStatus": 400001,
                                "Msg": error_msg
                            }, ensure_ascii=False))
                            continue

                        user_id         = result["user_id"]
                        prompt_num      = result["prompt_num"]
                        prompt_text     = result["prompt_text"]
                        bypass_mode     = result["bypass_mode"]

                        # 缓存文本信息，等待音频
                        pending_requests[request_id] = {
                            "user_id": user_id,
                            "prompt_num": prompt_num,
                            "prompt_text": prompt_text,
                            "bypass_mode": bypass_mode,
                            "received_audio_finish": False,
                            "start_time": time.time(),
                            "prompt_wav": b''
                        }

                        # 回复等待音频
                        await ws.send_text(json.dumps({
                            "request_id": request_id,
                            "ResultStatus": 0,
                            "Msg": "Waiting for audio data..."
                        }, ensure_ascii=False))

                    except json.JSONDecodeError:
                        content = {"ResultStatus": 410001, "Msg": "invalid JSON"}
                        await ws.send_text(json.dumps(content))
                        continue

                elif "bytes" in msg:
                    payload_bytes = msg["bytes"]
                    if not payload_bytes:
                        await ws.send_text(json.dumps({
                            "request_id": "unknown",
                            "ResultStatus": 400001,
                            "Msg": "Empty audio bytes received"
                        }))
                        continue

                    total_len = payload_bytes[0]
                    if len(payload_bytes) < total_len:
                        print("Incomplete header, skip.")
                        continue

                    header_type = payload_bytes[1]
                    header_version = payload_bytes[2]
                    sequence = struct.unpack('>h', payload_bytes[3:5])[0]  # big-endian int16
                    req_id_len = payload_bytes[5]
                    req_id_start = 6
                    req_id_end = req_id_start + req_id_len

                    request_id = payload_bytes[req_id_start:req_id_end].decode('utf-8')

                    if request_id not in pending_requests.keys():
                        await ws.send_text(json.dumps({
                            "request_id": request_id,
                            "ResultStatus": 400001,
                            "Msg": "Please send text first."
                        }))
                        continue                        

                    audio_data_start = total_len

                    # === 提取音频体 ===
                    audio_body = payload_bytes[audio_data_start:] if audio_data_start < len(payload_bytes) else b''

                    previous_audio = pending_requests[request_id]['prompt_wav']
                    pending_requests[request_id]['prompt_wav'] = previous_audio + audio_body

                    if sequence == -1 :
                        pending_requests[request_id]['received_audio_finish'] = True
                    else:
                        continue

                    ctx = pending_requests[request_id]
                    del pending_requests[request_id]  # 防重放

                    # === 开始处理完整请求 ===
                    user_id = ctx["user_id"]
                    prompt_text = ctx["prompt_text"]
                    bypass_mode = ctx["bypass_mode"]
                    prompt_wav_bytes = ctx["prompt_wav"]

                    os.makedirs(GlobalConfigInst.speaker_info_dir, exist_ok=True)
                    os.makedirs(os.path.join(GlobalConfigInst.speaker_info_dir, "prompt_wavs"), exist_ok=True)

                    origin_wav_path = os.path.join(GlobalConfigInst.speaker_info_dir, "prompt_wavs", f"{user_id}.wav")

                    # 直接写入二进制音频（无需 base64 解码）
                    with open(origin_wav_path, "wb") as f:
                        f.write(prompt_wav_bytes)

                    # 转为单声道（与原逻辑一致）
                    prompt_wav_bytes, origin_sample_rate = convert_to_single_channel_audiobytes(origin_wav_path)

                    # === ASR 质检 ===
                    passed, msg, final_prompt_text = _perform_asr_quality_check(
                        prompt_wav_bytes=prompt_wav_bytes,
                        prompt_text=prompt_text,
                        request_id=request_id,
                        user_id=user_id,
                        bypass_mode=bypass_mode
                    )
                    if not passed:
                        logger.info(f"[ws] RegisterSpeaker failed (ASR), request_id: {request_id}, user_id: {user_id}, reason: {msg}")
                        await ws.send_text(json.dumps({
                            "request_id": request_id,
                            "ResultStatus": 400007,
                            "Msg": msg
                        }, ensure_ascii=False))
                        continue

                    # === 启动后台去噪 & 注册 ===
                    tts_engine = getattr(ws.app.state, 'TTS_MODEL', None)
                    sample_rate = getattr(tts_engine, 'sample_rate', 16000) if tts_engine else 16000
                    cosyvoice_frontend = getattr(tts_engine, 'frontend', None) if tts_engine else None

                    asyncio.create_task(
                        run_denoise_and_register_in_background_async(
                            user_id=user_id,
                            prompt_wav_bytes=prompt_wav_bytes,
                            prompt_text=final_prompt_text,
                            prompt_num=10000,
                            bypass_mode=bypass_mode,
                            request_id=request_id,
                            cosyvoice_frontend=cosyvoice_frontend,
                            sample_rate_model=sample_rate
                        )
                    )

                    logger.info(f"[ws] RegisterSpeaker success, request_id: {request_id}, user_id: {user_id}")
                    await ws.send_text(json.dumps({
                        "request_id": request_id,
                        "ResultStatus": 0,
                        "Msg": "RegisterSpeaker Success"
                    }, ensure_ascii=False))

                else:
                    continue

    except WebSocketDisconnect:
        logger.info("Client disconnected during registration.")


    except Exception as e:
        logger.error(f"RegisterSpeaker WebSocket error: {e}")
        try:
            await ws.send_text(json.dumps({
                "request_id": request_id or "unknown",
                "ResultStatus": 410001,
                "Msg": "Internal server error",
                "error": str(e)
            }, ensure_ascii=False))
        except:
            pass


def construct_header(
    request_id: str,
    sequence: int,
    header_type: int = 0,
    header_version: int = 0,
    sample_rate: int = 24000,
    encoding: str = 0 
) -> bytes:
    """
    构造二进制头部，格式：
    
    Args:
        request_id: 请求ID（UTF-8字符串，建议长度 < 40
        sequence: 块序号（范围 -32768 ~ 32767，-1 通常表示结束帧）
        header_type: 头部类型（默认 0 = 音频数据帧）
        header_version: 协议版本（默认 1）
    
    Returns:
        bytes: 完整的二进制头部
    """
    
    req_id_len = len(request_id)
    req_id_bytes = request_id.encode('utf-8')

    if not (-32768 <= sequence <= 32767):
        sequence = 0
    if not (0 <= header_type <= 255):
        header_type = 0
    if not (0 <= header_version <= 255):
        header_version = 0

    sample_rate_map = {
        24000 : 0,
        22500 : 1,
        16000 : 2,
        8000  : 3,
        48000 : 4,
    }

    encoding_map = {
        'raw': 0,
        'wav': 1,
        'pcm': 2
    }

    sample_rate_type    = sample_rate_map.get(sample_rate, 0)
    encoding_type       = encoding_map.get(encoding, 0)

    header_without_total_len = (
        struct.pack('B', header_type) +        # 1
        struct.pack('B', header_version) +     # 1
        struct.pack('>h', sequence) +          # 2
        struct.pack('B', req_id_len) +         # 1
        req_id_bytes +                         # N
        struct.pack('B', sample_rate_type) +   # 1
        struct.pack('B', encoding_type)        # 1
    ) 

    # 4. total_len = 1 (自身) + len(header_without_total_len)
    total_len = 1 + len(header_without_total_len)
    if total_len > 255:
        raise RuntimeError(f"Header total length ({total_len}) exceeds uint8 limit (255)")

    return struct.pack('B', total_len) + header_without_total_len




async def process_single_tts_request(ws: WebSocket, raw_payload: str):
    try:
        try:
            raw_dict = json.loads(raw_payload)
            saved_id = ''
            if GlobalConfigInst.save_middle_result == 1:
                saved_id = generate_unique_id()
                raw_dict['saved_id'] = saved_id

            logger.info(f"TTSInfer [ws] req: {raw_dict}")
        except json.JSONDecodeError:
            content={
                "status_code":400,
                "ResultStatus": 410001,  
                "Msg": "invalid JSON"
            }
            logger.info(f"TTSInfer JSON Load failed: invalid JSON ")
            await ws.send_text(json.dumps(content))
            return
        is_valid, result = _prepare_voice_clone_request(raw_dict)
        request_id = raw_dict.get("request_id", "null")
        if not is_valid:
            error_msg = result  # str
            logger.info(f"[ws] TTSInfer validation failed: {error_msg}, request_id: {request_id}")
            content={
                "status_code":400,
                "request_id": request_id,
                "ResultStatus": 410001,  
                "Msg": error_msg
            }
            await ws.send_text(json.dumps(content))
            return
        else:
            req = result

        request_id = req["request_id"]
        
        if len(request_id) >= 64:
            await ws.send_text(json.dumps({
                "ResultStatus": 410001, 
                "request_id": request_id,
                "Msg": "request_id too long (>=64 bytes)"
            }))
            return


        tts_model_local = ws.app.state.TTS_MODEL
        if tts_model_local is None:
            content={
                "status_code":400,
                "request_id": request_id,
                "ResultStatus": 410001,  
                "Msg": "TTS model not initialized"
            }            
            await ws.send_text(json.dumps(content))
            return

        audio_data = None
        metadata_sent = False
        postprocessed_sr = 0
        sequence = 0
        async with ws.app.state.inference_semaphore:
            speech_token = []
            if 'saved_id' in req:
                is_save_speech_token = True
            else:
                is_save_speech_token = False

            try:
                async for model_chunk in tts_model_local.inference_zero_shot_by_spk_id(
                    req["text"],
                    req['user_id'],
                    req['streaming_mode'],
                    req['speed'],
                    text_frontend=True
                ):
                    tts_status  = model_chunk.get('tts_status', 0)
                    chunk_index = model_chunk.get('chunk_index', 0)
                    if tts_status == -2:
                        continue
                    if chunk_index ==  -1:
                        sequence = -1

                    audio_chunk = model_chunk.get('tts_speech')
                    if audio_chunk is None:
                        continue

                    audio_chunk = audio_chunk.squeeze().cpu().numpy().astype(np.float32)
                    aigc_metadata = get_aigc_metadata(protocol='websocket', req=req)

                    if req['streaming_mode']:
                        audio_chunk = np.clip(audio_chunk, -1.0, 1.0)
                        if audio_chunk.dtype != np.float32:
                            audio_chunk = audio_chunk.astype(np.float32)


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


                        audio_chunk, postprocessed_sr = audio_post_process(audio_chunk, tts_model_local.sample_rate, req) # 输入audio_data是float，输出是int
                        audio_bytes = pack_audio(BytesIO(), audio_chunk, postprocessed_sr, req['encoding'], aigc_metadata).getvalue()

                        header_bytes = construct_header(request_id=request_id, sequence = sequence)
                        await ws.send_bytes(header_bytes + audio_bytes)
                        sequence += 1
                        # print(f'stream chunk_index = {chunk_index}')

                    else:
                        if audio_data is None:
                            audio_data = audio_chunk
                        else:
                            audio_data = np.concatenate([audio_data, audio_chunk], axis=0)

                        if is_save_speech_token:
                            speech_token_tmp = model_chunk['speech_token']
                            speech_token_tmp = speech_token_tmp.squeeze()
                            speech_token_tmp = speech_token_tmp.tolist()
                            speech_token.extend(speech_token_tmp)



            except Exception as e:
                logger.error(f"[ws] TTSInfer error: {e}")
                content={
                    "status_code":400,
                    "ResultStatus": 410001,  
                    "Msg": "TTS inference failed"
                }
                await ws.send_text(json.dumps(content))
                return

            if req['streaming_mode'] is False:  
                if audio_data is None or len(audio_data) == 0:
                    content={
                        "status_code":400,
                        "request_id": request_id,
                        "ResultStatus": 410001,  
                        "Msg": "tts failed, No audio generated."
                    }            
                    await ws.send_text(json.dumps(content))
                    return

                audio_data = np.clip(audio_data, -1.0, 1.0)
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                audio_data, postprocessed_sr = audio_post_process(audio_data, tts_model_local.sample_rate, req) # 输入audio_data是float，输出是int
                aigc_metadata = get_aigc_metadata(protocol='websocket', req=req)
                audio_bytes = pack_audio(BytesIO(), audio_data, postprocessed_sr, req['encoding'], aigc_metadata).getvalue()

                if is_save_speech_token:
                    json_data_dict = OrderedDict()
                    json_data_dict['saved_id']      = req['saved_id']
                    json_data_dict['speech_token']  = speech_token
                    dump_line = json.dumps(json_data_dict, ensure_ascii=False)
                    logger_debug.info(f"{dump_line}")     

                chunk_size = 32768
                for idx in range(0, len(audio_bytes), chunk_size):
                    if idx + chunk_size >= len(audio_bytes) :
                        sequence = -1
                    else:
                        sequence = idx // chunk_size
                    chunk = audio_bytes[idx:idx + chunk_size]
                    header_bytes = construct_header(request_id=request_id, sequence = sequence)
                    await ws.send_bytes(header_bytes + chunk)
                    # print(f'nonstream chunk_index = {int(i/chunk_size)}')


    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"TTSInfer WebSocket error: {e}")


@app.websocket("/jttts/TTSInfer")
@app.websocket("/jttts/Voice_Clone")
@app.websocket("/jttts/TTS_Service")
async def tts_process_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            message = await ws.receive()  
            if message["type"] == "websocket.disconnect":
                # logger.info("Client disconnected during TTSInfer.")
                break
                
            elif message["type"] == "websocket.receive":
                if "text" in message:
                    payload = message["text"]
                    asyncio.create_task(process_single_tts_request(ws, payload))

    except Exception as e:
        logger.error(f"WebSocket error: {e}")