#coding:utf-8


import os, pdb
import subprocess
import json, time, base64
from jttts.tts_common.logger import logger as log

from jttts.config import GlobalConfigInst
import requests
import soundfile as sf
import io
code_version_file_latesttime = None

ftp_proxy_value = os.getenv('ftp_proxy', None)
https_proxy_value = os.getenv('https_proxy', None)
http_proxy_value = os.getenv('http_proxy', None)


if ftp_proxy_value:
    del os.environ['ftp_proxy']

if https_proxy_value:
    del os.environ['https_proxy']

if http_proxy_value:
    del os.environ['http_proxy']



def TTS_FileRepair():
    port = GlobalConfigInst.http_port
    port = str(port)
    base_url = f'http://localhost:{port}/' 

    RepairDic={}
    RepairDic['request_id']     = 'sefd'
    RepairDic['type']           = 'all'  # wordseg   chinese_pron  front_repair  all

    url =  base_url + "jttts/TTS_FileRepair"

    payload = json.dumps(RepairDic)

    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(url, headers=headers, data=payload)

    if response.status_code == 200:
        json_data       = response.json()
        # print(json_data['Msg'])

        if 'info' in json_data:
            pass
            # print(json_data['info'])

        return True
    else:
        json_data       = response.json()

        log.info(f"auto update params ! Failed , {json_data['Msg']}")
        return False


def update_params_once():
    global code_version_file_latesttime

    code_version_file            = os.path.join(GlobalConfigInst.front_init_dir, 'code_version.json')

    if os.path.exists(code_version_file):
        now_time = os.path.getmtime(code_version_file)

        if code_version_file_latesttime is None:
            code_version_file_latesttime = now_time
        else:
            if now_time > code_version_file_latesttime:
                code_version_file_latesttime = now_time
                is_success = TTS_FileRepair()
                if is_success:
                    log.info(f"auto update params ! success")
                else:
                    log.info(f"[error] auto update params ! Failed")



def update_params_period():
    is_first_check = 0
    while True:
        if is_first_check == 0:
            time.sleep(60*20)
            is_first_check = 1
        
        update_params_once()
        time.sleep(60*1)



# **************************************************************************


def Localhost_GetAsrResult(RequestDic: dict={}):
    port = GlobalConfigInst.check_port
    port = str(port)
    base_url = f'http://localhost:{port}/' 

    url =  base_url + "jttts/GetAsrResult"

    payload = json.dumps(RequestDic)

    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(url, headers=headers, data=payload)

    if response.status_code == 200:
        json_data       = response.json()
        asr_result      = json.loads(json_data['asr_result'])['text']
        return asr_result
    else:
        return None


def Localhost_GetDenoiseResult(RequestDic: dict={}):
    port = GlobalConfigInst.check_port
    port = str(port)
    base_url = f'http://localhost:{port}/' 

    url =  base_url + "jttts/GetDenoiseResult"

    payload = json.dumps(RequestDic)

    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(url, headers=headers, data=payload)

    if response.status_code == 200:
        json_data       = response.json()

        sample_rate     = json_data['sample_rate']
        audio_data      = json_data['audio_data']
        audio_bytes      = base64.b64decode(audio_data)
        sequence        = json_data['sequence']

        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))


        return audio_data, sample_rate
    else:
        return None, None



if __name__ == "__main__":

    pass