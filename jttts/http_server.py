
import os
import sys, pdb
from jttts.tts_common.request_processor import _perform_asr_quality_check, run_denoise_and_register_in_background, _validate_and_normalize_upload_prompt, _prepare_voice_clone_request, truncate_large_values, convert_to_single_channel_audiobytes

import threading, time
from jttts.tts_common.logger import logger, logger_debug
from jttts.config import GlobalConfigInst
from datetime import datetime

import traceback
import asyncio
import base64
import re
import torchaudio
from pathlib import Path
import argparse
import signal
import numpy as np
import subprocess
import soundfile as sf
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import FastAPI, UploadFile, File
from fastapi import BackgroundTasks

import uvicorn
import torch, librosa, json
from io import BytesIO
from pydantic import BaseModel
import re, io
from threading import Lock, Thread
Lock = Lock()
import uuid
from functools import partial  
from collections import OrderedDict
import logging
logging.basicConfig(level=logging.INFO, format='[ %(levelname)s %(asctime)s %(filename)s:%(lineno)d ] %(message)s')


from jttts.tts_model.tts_utils import pack_audio, audio_post_process, get_aigc_metadata, generate_unique_id

class LaunchFailed(Exception):
    pass


def handle_control(command:str):
    if command == "restart":
        os.execl(sys.executable, sys.executable, *argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)


async def Upload_Prompt_http_Warmup(APP: FastAPI):
    input_text_file = './build_in_speaker/warm_up/warm_up.txt'
    with open(os.path.join(input_text_file),encoding="utf-8",) as ttf:
        lines = ttf.readlines()
        for idx in range(len(lines)):
            utt_id = lines[idx].strip().split()[0]
            prompt_text = lines[idx].strip()[len(utt_id):].strip()
            File_path   = os.path.join('./build_in_speaker/warm_up/', utt_id + '.wav')

            # prompt_audio, prompt_sr = torchaudio.load(File_path)

            # if prompt_sr != 16000:
            #     prompt_audio = torchaudio.functional.resample(prompt_audio, prompt_sr, 16000)        
            # APP.TTS.frontend.generate_spk_info(utt_id, prompt_text, prompt_audio, APP.TTS.sample_rate)
            APP.TTS.frontend.generate_spk_info_by_audiopath(utt_id, prompt_text, File_path, output_dir = GlobalConfigInst.speaker_info_dir)
            logger.info(f"[warm_up] RegisterSpk {utt_id} Success!!")
            logging.info(f"[warm_up] RegisterSpk {utt_id} Success!!")


async def Voice_Clone_http_Warmup(APP: FastAPI):
    Text_sets = [
                '九天大模型将为十亿用户带来新体验、心服务，为千航百业带来新变革、新发展',
                '今天很开心，很高兴认识你',
                '由于高空的温度较低，水汽达到饱和状态造成了凝结',
                '昨天，北京低空湿度条件不利，但动力条件真好，最后是大力出奇迹，就像拧毛巾，最后拧出来了。',
                '低层空气其实非常干燥，考虑到在飘落过程中会被蒸发掉，我们本来预计平原地区是不会形成降雪的。',
                '从雷达图上看，从西北边有一条很小的回波往东南方向推进，但已经呈现出减弱趋势，所以飘雪基本上就集中在城区和东部地区，分布非常不均匀，而且移出的速度非常快，预计一两个小时就将结束。',
                '记者了解到，虽然大家视觉上看到飘落的雪花不小，但由于持续时间很短，加上量非常小，所以基本上地面不会形成积雪。',
                '从经济角度来看，提前还贷的前提是要对比其他投资机会，如果发现提前还贷的性价比更高，才会选择这种“反向操作”，但当前环境下，这种操作可能并不是最优选择。',
                '一般而言，个人住房贷款具有期限较长、利率较低、申请方便、还款灵活等特点。',
                '随着降低房贷利率、降低首付比例等政策的相继推出，提前还贷情况有所缓解。不过，对于贷款人而言，是否提前还贷还要考虑诸多因素。'
                ]

    for idx, sub_text in enumerate(Text_sets):
        tts_text_split = [sub_text]
        for req_idx , sub_text in enumerate(tts_text_split):
            text_frontend = True
            async for model_chunk in APP.TTS.inference_zero_shot_by_spk_id(
                        sub_text,
                        'warm_up_exp1',
                        False,
                        1.0,
                        text_frontend,
            ):
                audio_data_tmp = model_chunk['tts_speech']
        logger.info(f"[warm_up] TTS_Service {idx} Success!!")
        logging.info(f"[warm_up] TTS_Service {idx} Success!!")


@asynccontextmanager
async def lifespan(APP: FastAPI):
    yield

APP = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)


@APP.middleware("http")
async def add_x_frame_options_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response


@APP.get("/control")
async def control(command: str = None):
    if command is None:
        return JSONResponse(status_code=400, content={"ResultStatus": 0, "Msg": "command is required"})
    handle_control(command)

# *********************************************************************************************
async def tts_handle(req: dict, http_request:Request):
    request_id = 'null'
    try:
        request_id = req.get("request_id", 'null')
        streaming_mode = req.get("streaming_mode", False)
        

        async with http_request.app.state.inference_semaphore:
            if streaming_mode:
                pass
            else:
                audio_data = None
                speech_token = []

                text_frontend = True
                if 'saved_id' in req:
                    is_save_speech_token = True
                else:
                    is_save_speech_token = False
                
                tts_model_local = http_request.app.TTS
                async for model_chunk in tts_model_local.inference_zero_shot_by_spk_id(
                            req["text"],
                            req['user_id'],
                            req['streaming_mode'],
                            req['speed_ratio'],
                            text_frontend,
                ):
                    tts_status = model_chunk['tts_status']
                    if tts_status == -2:
                        continue
                    else:
                        audio_data_tmp = model_chunk['tts_speech']
                        audio_data_tmp = audio_data_tmp.squeeze()
                        audio_data_tmp = audio_data_tmp.numpy().astype(np.float32)

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
                    return JSONResponse(status_code=400, content={"request_id":request_id, "ResultStatus": 410000, "Msg": f"The input text is too long!",})

                if is_save_speech_token:
                    json_data_dict = OrderedDict()
                    json_data_dict['saved_id']      = req['saved_id']
                    json_data_dict['speech_token']  = speech_token
                    dump_line = json.dumps(json_data_dict, ensure_ascii=False)
                    logger_debug.info(f"{dump_line}")



                audio_data, postprocessed_sr = audio_post_process(audio_data, tts_model_local.sample_rate, req) # 输入audio_data是float，输出是int
                aigc_metadata = get_aigc_metadata(protocol='http', req=req)

                sequence = -1
                if GlobalConfigInst.interface_type == 2:
                    audio_data_pack_raw = pack_audio(BytesIO(), audio_data, postprocessed_sr, req['encoding'], aigc_metadata).getvalue()
                    audio_data_retrun = base64.b64encode(audio_data_pack_raw).decode()
                    return JSONResponse(status_code=200, content={"request_id":request_id, "ResultStatus": 0, "Msg": "tts success", "sample_rate":postprocessed_sr, "version": str(2.0), "audio_data": audio_data_retrun, "sequence": sequence, "AIGC":aigc_metadata}) 
                elif GlobalConfigInst.interface_type == 1:
                    audio_data = pack_audio(BytesIO(), audio_data, postprocessed_sr, "wav", aigc_metadata).getvalue()
                    audio_base64 = str(base64.b64encode(audio_data), encoding="utf-8")
                    return JSONResponse(status_code=200, content={"ResultStatus": 0, "Msg": "tts success", "version": str(1.0), "clone_wave": audio_base64, "sequence": sequence, "AIGC":aigc_metadata})
                else:
                    return JSONResponse(status_code=400, content={"request_id":request_id, "ResultStatus": 410010, "Msg": f"tts failed, interface_type failed "})
    except Exception as e:
        logger.info(f"[error][http] TTSInfer failed {str(e)}", is_encrypt=True)
        return JSONResponse(status_code=400, content={"request_id":request_id, "ResultStatus": 410000, "Msg": f"tts failed", "Exception": str(e)})


# *************************************************************************************************
class Voice_Clone_Payload(BaseModel):
    request_id: str = 'null'
    text: str = ''
    text_pinyins: str = ''
    user_id: str = ''
    speaker_id: str = ''
    speed_ratio: float   = GlobalConfigInst.default_speed
    volume_ratio: float  = float(1.0)
    sample_rate: int   = int(GlobalConfigInst.default_sample_rate)
    streaming_mode: bool = False
    Streaming_mode: bool = False
    encoding: str = GlobalConfigInst.default_encoding
    language: str = 'auto'
    debug_mode: int = 0

@APP.post("/jttts/TTSInfer")
@APP.post("/jttts/Voice_Clone")
@APP.post("/jttts/TTS_Service")
async def tts_post_endpoint(payload: Voice_Clone_Payload, http_request:Request):
    request_id = 'null'
    try:
        raw_dict = payload.dict()
        if GlobalConfigInst.save_middle_result == 1:
            raw_dict['saved_id'] = generate_unique_id()

        logger.info(f"TTSInfer [http] req:{raw_dict}", is_encrypt=True)

        is_valid, result = _prepare_voice_clone_request(raw_dict)

        if not is_valid:
            request_id = raw_dict.get("request_id", "null")
            error_msg = result  
            logger.info(f"[http] TTSInfer validation failed: {error_msg}, request_id: {request_id}")
            return JSONResponse(
                status_code=400,
                content={
                    "request_id": request_id,
                    "ResultStatus": 410001,  # 或统一用 410001
                    "Msg": error_msg
                }
            )

        req = result
        request_id = req["request_id"]

        streaming_mode = req.get("streaming_mode", False)
        if streaming_mode is True:
            error_msg = '[http] The http donot support stream response !'  
            logger.info(f"TTSInfer validation failed: {error_msg}, request_id: {request_id}")
            return JSONResponse(
                status_code=400,
                content={
                    "request_id": request_id,
                    "ResultStatus": 410001,  # 或统一用 410001
                    "Msg": error_msg
                }
            )            


        return await  tts_handle(req, http_request)
    except Exception as e:
        return JSONResponse(status_code=400, content={"request_id":request_id, "ResultStatus": 410000, "Msg": f"tts failed", "Exception": str(e)})

    finally:
        pass

# *************************************************************************************************
class Test_Payload(Voice_Clone_Payload):
    llm_tokens: str=''

@APP.post("/jttts/test")
async def tts_test_endpoint(payload: Test_Payload, http_request:Request):
    request_id = 'null'
    try:
        req = payload.dict()
        audio_data = None

        if  req["speaker_id"]:
            if len(req["speaker_id"].strip()) > 0:
                req["user_id"] = req["speaker_id"]

        req["volume"] = req["volume_ratio"]
        req["speed"] = req["speed_ratio"]

        tts_model_local = http_request.app.TTS
        llm_tokens_str = req["llm_tokens"]
        llm_tokens = llm_tokens_str.split()
        llm_tokens = [int(token) for token in llm_tokens]
        model_chunk = await tts_model_local.inference_llm_token_by_spk_id(llm_tokens=llm_tokens, spk_id=req['user_id'], req= req)

        tts_status = model_chunk['tts_status']
        if tts_status == -2:
            audio_data = None
        else:
            audio_data_tmp = model_chunk['tts_speech']
            audio_data_tmp = audio_data_tmp.squeeze()
            audio_data_tmp = audio_data_tmp.numpy().astype(np.float32)

            if audio_data is None:
                audio_data = audio_data_tmp
            else:
                audio_data = np.concatenate([audio_data, audio_data_tmp], axis=0)

        if audio_data is None:
            return JSONResponse(status_code=400, content={"request_id":request_id, "ResultStatus": 410000, "Msg": f"The input text is too long!",})

        audio_data, postprocessed_sr = audio_post_process(audio_data, tts_model_local.sample_rate, req) # 输入audio_data是float，输出是int
        aigc_metadata = get_aigc_metadata(protocol='http', req=req)

        sequence = -1

        audio_data_pack_raw = pack_audio(BytesIO(), audio_data, postprocessed_sr, req['encoding'], aigc_metadata).getvalue()
        audio_data_retrun = base64.b64encode(audio_data_pack_raw).decode()
        return JSONResponse(status_code=200, content={"request_id":request_id, "ResultStatus": 0, "Msg": "tts success", "sample_rate":postprocessed_sr, "version": str(2.0), "audio_data": audio_data_retrun, "sequence": sequence, "AIGC":aigc_metadata}) 



    except Exception as e:
        return JSONResponse(status_code=400, content={"request_id":request_id, "ResultStatus": 410000, "Msg": f"tts failed", "Exception": str(e)})

    finally:
        pass


# *************************************************************************************************
class Upload_Prompt_Payload(BaseModel):
    request_id: str = 'null'
    user_id: str = ''
    speaker_id: str = ''
    prompt_num: int = None
    upload_prompt_text: str = 'null'
    prompt_wav: str = None
    bypass_mode: int = 0 # 0--no bypass

@APP.post("/jttts/Upload_Prompt")
@APP.post("/jttts/RegisterSpeaker")
async def tts_Upload_Prompt(payload: Upload_Prompt_Payload, background_tasks: BackgroundTasks, http_request:Request):
    request_id = 'null'
    try:
        raw_dict = payload.dict()
        safe_log_dict = truncate_large_values(raw_dict, max_length=100, show_prefix=10)
        dict_str = json.dumps(safe_log_dict, ensure_ascii=False)
        request_id = raw_dict.get("request_id", "null")
        logger.info(f"RegisterSpeaker [http] Start, {dict_str} ")

        is_valid, result = _validate_and_normalize_upload_prompt(raw_dict)

        if not is_valid:
            request_id = raw_dict.get("request_id", "null")
            error_msg = result  # str
            logger.info(f"[http] RegisterSpeaker validation failed, request_id: {request_id}, Msg:{error_msg}, ")
            return JSONResponse(
                status_code=400,
                content={
                    "request_id": request_id,
                    "ResultStatus": 400001, 
                    "Msg": error_msg
                }
            )

        request_id = result["request_id"]
        user_id = result["user_id"]
        prompt_num = result["prompt_num"]
        prompt_text = result["prompt_text"]
        wave_data_b64 = result["prompt_wav"]
        bypass_mode = result["bypass_mode"]


        os.makedirs(GlobalConfigInst.speaker_info_dir, exist_ok=True)
        os.makedirs(os.path.join(GlobalConfigInst.speaker_info_dir, "prompt_wavs"), exist_ok=True)
        origin_wav_path = os.path.join(GlobalConfigInst.speaker_info_dir, "prompt_wavs", user_id + ".wav")
        prompt_wav_bytes = base64.b64decode(wave_data_b64)
        with open(origin_wav_path, "wb") as refer_wav_file_fid:
            refer_wav_file_fid.write(prompt_wav_bytes)

        prompt_use_wav_path = os.path.join("./prompt_wavs", user_id + ".wav")
    
        prompt_wav_bytes, origin_sample_rate = convert_to_single_channel_audiobytes(origin_wav_path)

        # if origin_sample_rate <16000:
        #     msg = f'The register audio sample_rate is {origin_sample_rate}, and The sample_rate must >= 16k,'
        #     return JSONResponse(status_code=400, content={
        #         "request_id": request_id,
        #         "ResultStatus": 400007,
        #         "Msg": msg
        #     })


        # === 统一 ASR 质检 ===
        passed, msg, final_prompt_text = _perform_asr_quality_check(
            prompt_wav_bytes=prompt_wav_bytes,
            prompt_text=prompt_text,
            request_id=request_id,
            user_id=user_id,
            bypass_mode=bypass_mode
        )
        if not passed:
            logger.info(f"[http] RegisterSpeaker failed (ASR), request_id: {request_id}, user_id: {user_id}, reason: {msg}")
            return JSONResponse(status_code=400, content={
                "request_id": request_id,
                "ResultStatus": 400007,
                "Msg": msg
            })

        # === 启动后台任务 ===
        tts_engine  = getattr(http_request.app, 'TTS', None)
        sample_rate = getattr(tts_engine, 'sample_rate', 16000) if tts_engine else 16000

        background_tasks.add_task(
            run_denoise_and_register_in_background,
            user_id=user_id,
            prompt_wav_bytes=prompt_wav_bytes,
            prompt_text=final_prompt_text,
            prompt_num=prompt_num,
            bypass_mode=bypass_mode,
            request_id=request_id,
            cosyvoice_frontend=getattr(tts_engine, 'frontend', None) if tts_engine else None,
            sample_rate_model=sample_rate
        )
        logger.info(f"[http] RegisterSpeaker success (ASR), request_id: {request_id}, user_id: {user_id}, reason: {msg}")
        return JSONResponse(status_code=200, content={
            "request_id": request_id,
            "ResultStatus": 0,
            "Msg": "RegisterSpeaker Success"
        })

    except Exception as e:
        logger.info(f"[http] RegisterSpeaker error, request_id: {request_id}, error={str(e)}") 
        return JSONResponse(status_code=400, content={"request_id":request_id, "ResultStatus": 400008, "Msg": "RegisterSpeaker failed", "error": e})
    


class ClearSpkinfo_Payload(BaseModel):
    request_id: str = 'null'
    user_id: str = None
    speaker_id: str = None

@APP.post("/jttts/ClearSpkInfo")
async def tts_clear_spkinfo(payload: ClearSpkinfo_Payload, background_tasks: BackgroundTasks, ):
    request_id = 'null'
    try:
        json_post_raw = payload.dict()
        request_id = json_post_raw['request_id']
        if json_post_raw['speaker_id']:
            json_post_raw['user_id'] = json_post_raw['speaker_id']

        IsClearFlag = APP.TTS.frontend.clear_spk_info(json_post_raw['user_id'])
        if IsClearFlag :
            return JSONResponse(status_code=200, content={"request_id":request_id, "ResultStatus": 0, "Msg": "Clear Spkinfo success"})
        else:
            return JSONResponse(status_code=400, content={"request_id":request_id, "ResultStatus": 400008, "Msg": "Clear Spkinfo failed, not in spk2info"})
    except Exception as e:
        logger.info(f"Clear Spkinfo error, payload: {payload}") 
        return JSONResponse(status_code=400, content={"request_id":request_id, "ResultStatus": 400008, "Msg": "Clear Spkinfo failed", "error": e})
    





# @APP.get("/jttts/check_service_health")
# async def check_service_health():
#     text_frontend = True
#     sub_text = '一般而言，个人住房贷款具有期限较长、利率较低、申请方便、还款灵活等特点。'
#     async for model_chunk in APP.TTS.inference_zero_shot_by_spk_id(
#                 sub_text,
#                 'warm_up_exp1',
#                 False,
#                 1.0,
#                 text_frontend,
#     ):
#         audio_data_tmp = model_chunk['tts_speech']



class GetVersion_Payload(BaseModel):
    GetConfig: str = False
    GetCodeVersion: str = False


@APP.post("/jttts/GetVersion")
async def tts_GetVersion(payload: GetVersion_Payload):
    try:
        req        = payload.dict()
        GetConfig  = req.get("GetConfig", False)
        GetCodeVersion  = req.get("GetCodeVersion", False)

        TTS_Run_Info = {}
        GlobalConfigInst_dict = vars(GlobalConfigInst)

        if GetConfig:
            TTS_Run_Info['GlobalConfig'] = GlobalConfigInst_dict
            # TTS_Run_Info['MainPyFile']   = py_file
            # TTS_Run_Info['LicenseFile']  = license_file_path

        if GetCodeVersion:
            code_info_file = './code_version.json'
            if os.path.exists(code_info_file) :
                with open(code_info_file, 'r', encoding="utf-8") as fjson:
                    code_info = json.load(fjson)

                TTS_Run_Info['code_info']  = code_info

        return JSONResponse(status_code=200, content={"Msg": "GetVersion Success",'info': json.dumps(TTS_Run_Info)})
    except Exception as e:
        return JSONResponse(status_code=400, content={"Msg": "GetVersion Failed", "error": e})

class GetStatus_Payload(BaseModel):
    GetStatus: str = False

@APP.post("/jttts/GetStatus")
async def tts_GetVersion(payload: GetStatus_Payload):
    try:
        req        = payload.dict()
        TTS_Status_Info = {}
        TTS_Status_Info['status'] = 'Hello VoiceClone ! Service is ok !'

        return JSONResponse(status_code=200, content={'Msg': 'Hello VoiceClone ! Service is ok !'})
    except Exception as e:
        return JSONResponse(status_code=400, content={"Msg": "GetStatus Failed", "error": e})


@APP.get("/jttts/healthz")
async def health_check():    
    return {"status": "ok", "message": "Hello TTS Service is ready."}


if __name__ == "__main__":
    try:
        # t1=threading.Thread(target=checklicense)
        # t1.start()

        if GlobalConfigInst.prompt_quality_check or GlobalConfigInst.prompt_denoise:
            def run_python_script(script_name):
                try:
                    subprocess.run(['python', script_name], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error running {script_name}: {e}")
            if os.path.exists('./register_service.pyc'):
                t3=threading.Thread(target=run_python_script, args=('register_service.pyc',))
                t3.start()
            else:
                t3=threading.Thread(target=run_python_script, args=('register_service.py',))
                t3.start()
            time.sleep(30)

        uvicorn.run(app=f'{Path(__file__).stem}:APP', host="0.0.0.0", \
                    port=GlobalConfigInst.http_port, workers=GlobalConfigInst.workers)
    except Exception as e:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
