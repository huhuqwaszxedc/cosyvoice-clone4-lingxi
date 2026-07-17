
import os
import sys, pdb
import threading, time
from jttts.tts_common.logger import logger as log
from jttts.config import GlobalConfigInst


ftp_proxy_value = os.getenv('ftp_proxy', None)
https_proxy_value = os.getenv('https_proxy', None)
http_proxy_value = os.getenv('http_proxy', None)

if ftp_proxy_value:
    del os.environ['ftp_proxy']
if https_proxy_value:
    del os.environ['https_proxy']
if http_proxy_value:
    del os.environ['http_proxy']

# ********************************************************************************************
current_file_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_file_dir, './jttts/tts_engine/jtaudio/'))


import traceback
import base64
import re
from pathlib import Path
import signal
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import FastAPI, UploadFile, File
import uvicorn
import torch, librosa, json
from io import BytesIO
from pydantic import BaseModel
import re, io
from threading import Lock, Thread
Lock = Lock()
import uuid
from jttts.tts_model.tts_utils import pack_audio


class LaunchFailed(Exception):
    pass

move_asrmodel_to_cpu    = None
move_asrmodel_to_gpu    = None

move_denoisemodel_to_cpu    = None
move_denoisemodel_to_gpu    = None

def handle_control(command:str):
    if command == "restart":
        os.execl(sys.executable, sys.executable, *argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)

ASR_Model_Inst = None

# @asynccontextmanager
def lifespan(APP: FastAPI):
    # init inst
    global ASR_Model_Inst
    model_dir = GlobalConfigInst.backend_acoustic_model
    if os.path.exists(model_dir):
        log.info(f"MODEL_DIR is {model_dir}")

        APP.cpu_device = torch.device('cpu')
        APP.gpu_device = torch.device('cuda')
        if GlobalConfigInst.prompt_quality_check is False:
            ASR_Model_Inst = None
        else:
            try:
                from jttts.tts_engine.external_models.model import ASR_Model
            except:
                ASR_Model = None

            if ASR_Model is not None:
                try:
                    ASR_Model_Inst = ASR_Model()
                    refer_wav_path = './build_in_speaker/prompt_wavs/tts042_030386.wav'
                    prompt_audio, prompt_sr = sf.read(refer_wav_path)
                    text_asr = ASR_Model_Inst.inference(prompt_audio, prompt_sr)

                    ASR_Model_Inst.asr_model.model.model.model.to(APP.cpu_device)
                    ASR_Model_Inst.asr_model.model.model.vad_model.to(APP.cpu_device)
                    ASR_Model_Inst.asr_model.model.model.punc_model.to(APP.cpu_device)


                    # move_asrmodel_to_cpu(ASR_Model_Inst)
                except:
                    ASR_Model_Inst = None
            else:
                ASR_Model_Inst = None

            if ASR_Model_Inst :
                log.info(f"ASR model init success!") 
            else:
                log.info(f"ASR model init failed!") 
                print(f"ASR model init failed!")
                exit()

        
        if GlobalConfigInst.prompt_denoise is False:
            Denoise_Model_Inst = None
        else:
            try:
                from jttts.tts_engine.external_models.model import Denoise_Model
            except:
                Denoise_Model = None

            if Denoise_Model is not None:
                try:
                    Denoise_Model_Inst = Denoise_Model()
                    refer_wav_path = './build_in_speaker/prompt_wavs/tts042_030386.wav'
                    prompt_audio, prompt_sr = sf.read(refer_wav_path)
                    denoise_audio_, denoise_sr_ = Denoise_Model_Inst.inference(prompt_audio, prompt_sr)
                    Denoise_Model_Inst.denoise_model.model.to(APP.cpu_device)
                    Denoise_Model_Inst.denoise_model.model.to(APP.gpu_device)
                    Denoise_Model_Inst.denoise_model.model.to(APP.cpu_device)
                except:
                    Denoise_Model_Inst = None
            else:
                Denoise_Model_Inst = None
            
            if Denoise_Model_Inst:
                log.info(f"Denoise model init success!") 
            else:
                log.info(f"Denoise model init failed!") 
                print(f"Denoise model init failed!")
                exit()


        if GlobalConfigInst.prompt_remove_sil is False:
            RemoveSil_Model_Inst = None
        else:
            try:
                from jttts.tts_engine.external_models.model import RemoveSil_Model
            except:
                RemoveSil_Model = None

            if RemoveSil_Model is not None:
                try:
                    RemoveSil_Model_Inst = RemoveSil_Model()
                    refer_wav_path = './build_in_speaker/prompt_wavs/tts042_030386.wav'
                    prompt_audio, prompt_sr = sf.read(refer_wav_path)
                    rmsil_audio_, rmsil_sr_ = RemoveSil_Model_Inst.inference(prompt_audio, prompt_sr)
                except:
                    RemoveSil_Model_Inst = None
            else:
                RemoveSil_Model_Inst = None
            
            if RemoveSil_Model_Inst:
                log.info(f"RemoceSil model init success!") 
            else:
                log.info(f"RemoceSil model init failed!") 

        APP.ASR_Model_Inst      = ASR_Model_Inst
        APP.Denoise_Model_Inst  = Denoise_Model_Inst
        APP.RemoveSil_Model_Inst  = RemoveSil_Model_Inst

        log.info("Init ThirdParty Model successful !!")

    else:
        raise LaunchFailed("MODEL_DIR environment must set")
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
def control(command: str = None):
    if command is None:
        return JSONResponse(status_code=400, content={"ResultStatus": 0, "Msg": "command is required"})
    handle_control(command)


SV_Model_Inst = None
from jttts.tts_engine.external_models.model_utils import move_svmodel_to_cpu, move_svmodel_to_gpu

class GetSVScore_Request(BaseModel):
    audio_a: str = None
    audio_b: str = None
    audio_b_user_id: str = None

@APP.post("/jttts/GetSVScore")
def tts_GetSVScore(request: GetSVScore_Request):
    try:
        req        = request.dict()

        data_a = req['audio_a']
        audio_bytes_a = base64.b64decode(data_a)
        audio_data_a, sample_rate_a = sf.read(io.BytesIO(audio_bytes_a))
        if req['audio_b_user_id']:
            user_id = req['audio_b_user_id']
            json_file = os.path.join(GlobalConfigInst.speaker_info_dir, user_id + ".json")
            speaker_info_dir = GlobalConfigInst.speaker_info_dir
            if os.path.exists(json_file) is False:
                json_file = os.path.join(GlobalConfigInst.build_in_speaker_info_dir, user_id + ".json")
                speaker_info_dir = GlobalConfigInst.build_in_speaker_info_dir

            if os.path.exists(json_file) is False:
                return JSONResponse(status_code=400, content={"ResultStatus": 410008, "Msg": "The Prompt Audio is not upload!"}) 

            with open(json_file, 'r', encoding="utf-8") as fjson:
                spk_info = json.load(fjson)

            if './speaker_info/' in spk_info['prompt_wav']:
                spk_info['prompt_wav'] = spk_info['prompt_wav'].replace('speaker_info/', '')

            refer_wav_path = os.path.join(speaker_info_dir, spk_info['prompt_wav']) 
            audio_data_b, sample_rate_b = sf.read(refer_wav_path)

        else:
            data_b = req['audio_b']
            audio_bytes_b = base64.b64decode(data_b)
            audio_data_b, sample_rate_b = sf.read(io.BytesIO(audio_bytes_b))

        global SV_Model_Inst
        if SV_Model_Inst is None:
            try:
                from jttts.tts_engine.external_models.model import SpeakerVerification_Model
            except:
                SpeakerVerification_Model = None

            if SpeakerVerification_Model is not None:
                try:
                    SV_Model_Inst = SpeakerVerification_Model()
                    SV_Model_Inst.sv_model.model.to(APP.cpu_device)
                except:
                    SV_Model_Inst = None
            else:
                SV_Model_Inst = None

        if SV_Model_Inst:
            pass
        else:
            return JSONResponse(status_code=400, content={'Msg': 'SV_Model_Inst init failed !!'})

        SV_Model_Inst.sv_model.model.to(APP.gpu_device)
        score = SV_Model_Inst.inference(audio_data_a, sample_rate_a, audio_data_b, sample_rate_b)
        SV_result = {}
        SV_result['score'] = round(score, 2)
        SV_Model_Inst.sv_model.model.to(APP.cpu_device)
        return JSONResponse(status_code=200, content={'Msg': 'GetSVScore success', 'sv_score': json.dumps(SV_result)})
    except Exception as e:
        return JSONResponse(status_code=400, content={"Msg": "GetSVScore Failed", "error": e})

class GetAsrResult_Request(BaseModel):
    audio: str = None

@APP.post("/jttts/GetAsrResult")
def tts_GetAsrResult(request: GetAsrResult_Request):
    try:
        req        = request.dict()

        data = req['audio']
        audio_bytes = base64.b64decode(data)
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        if APP.ASR_Model_Inst:
            APP.ASR_Model_Inst.asr_model.model.model.model.to(APP.gpu_device)
            APP.ASR_Model_Inst.asr_model.model.model.vad_model.to(APP.gpu_device)
            APP.ASR_Model_Inst.asr_model.model.model.punc_model.to(APP.gpu_device)

            text_asr = APP.ASR_Model_Inst.inference(audio_data, sample_rate)

            APP.ASR_Model_Inst.asr_model.model.model.model.to(APP.cpu_device)
            APP.ASR_Model_Inst.asr_model.model.model.vad_model.to(APP.cpu_device)
            APP.ASR_Model_Inst.asr_model.model.model.punc_model.to(APP.cpu_device)

            asr_result = {}
            asr_result['text'] = text_asr
            return JSONResponse(status_code=200, content={'Msg': 'GetAsrResult success', 'asr_result': json.dumps(asr_result)})
        else:
            return JSONResponse(status_code=400, content={'Msg': 'GetAsrResult Failed, ASR Model is not init !'})
    except Exception as e:
        return JSONResponse(status_code=400, content={"Msg": "GetAsrResult Failed", "error": e})



@APP.post("/jttts/GetDenoiseResult")
def tts_GetDenoiseResult(request: GetAsrResult_Request):
    try:
        req        = request.dict()
        request_id = req.get('request_id', 'null')
        data = req['audio']
        audio_bytes = base64.b64decode(data)
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        if APP.Denoise_Model_Inst:
            APP.Denoise_Model_Inst.denoise_model.model.to(APP.gpu_device)
            denoise_audio_, denoise_sr_ = APP.Denoise_Model_Inst.inference(audio_data, sample_rate)
            APP.Denoise_Model_Inst.denoise_model.model.to(APP.cpu_device)
            sequence = -1

            audio_data_pack_raw = pack_audio(BytesIO(), denoise_audio_, denoise_sr_, 'wav').getvalue()
            audio_data_retrun = base64.b64encode(audio_data_pack_raw).decode()
            return JSONResponse(status_code=200, content={"request_id":request_id, "ResultStatus": 0, "Msg": "GetDenoiseResult success", "sample_rate":denoise_sr_, "version": str(2.0), "audio_data": audio_data_retrun, "sequence": sequence}) 
        else:
            return JSONResponse(status_code=400, content={'Msg': 'GetDenoiseResult Failed, Denoise Model is not init !'})

    except Exception as e:
        return JSONResponse(status_code=400, content={"Msg": "GetDenoiseResult Failed", "error": e})




if __name__ == "__main__":
    try:
        uvicorn.run(app=f'{Path(__file__).stem}:APP', host="0.0.0.0", port=GlobalConfigInst.check_port, workers=1)
    except Exception as e:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
