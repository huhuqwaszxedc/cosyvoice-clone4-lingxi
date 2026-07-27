import argparse
import json
import base64
import os
import requests
import time, wave
import json
import pdb
import soundfile as sf
import numpy as np
import io
import threading
import concurrent.futures
import random
import subprocess
import uuid
# try:
#     from kl_tts_Authorization import get_kc_act, check_token_expiration
# except:
#     exit()
#     from kl_access_token_origin.kl_tts_Authorization import get_kc_act, check_token_expiration

ftp_proxy_value = os.getenv('ftp_proxy', None)
https_proxy_value = os.getenv('https_proxy', None)
http_proxy_value = os.getenv('http_proxy', None)


if ftp_proxy_value:
    del os.environ['ftp_proxy']

if https_proxy_value:
    del os.environ['https_proxy']

if http_proxy_value:
    del os.environ['http_proxy']



import logging as log
log.basicConfig(level=log.INFO,format='[ %(levelname)s %(asctime)s %(filename)s:%(lineno)d ] %(message)s')

# global setting
access_token = ''

base_url = 'http://localhost:7820/'

interface_version = 2
env_name = ''
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=int, required=False, default=1, help='0-upload_prompt, 1-voice_clone, 2-multithread voice clone')
    parser.add_argument('-ifv', '--interface_version', type=int, required=False, default=2, help='1-v1.0, 2-v2.0')
    # parser.add_argument('-o', '--output_file', type=str,required=False, default="./test.wav")
    args = parser.parse_args()

    return args

# ************************************************************Upload_Prompt****************************************************************************************************
def Upload_Prompt(base_url, access_token, manual_user_id = None, TextWaveDict = {}):
    url =  base_url + "jttts/Upload_Prompt" 

    RequestDic={}
    if 1:
        RequestDic['user_id']      = 'test_speaker'
        RequestDic['prompt_num']   = 10000
        RequestDic['request_id']   = 'test_speaker_request'

        RequestDic['upload_prompt_text']   = '希望你以后能够做的比我还好呦。'
        File_path           = './prompt_wav/test_speaker.wav'
    else:

        if manual_user_id is None:
            RequestDic['user_id']      = 'CM1'
            RequestDic['prompt_num']   = 10000
            RequestDic['request_id']   = 'spk_test_request'

            RequestDic['upload_prompt_text']   = '快去#1帮他吧#3，我们#1还要#1把#1粮食#2抢回来呢#4。'
            RequestDic['upload_prompt_text']   = RequestDic['upload_prompt_text'].replace('#1', '').replace('#2', '').replace('#3', '').replace('#4', '')
            File_path           = '/mnt/g/海天数据集-2024.12/中文男童合成库-影视配音/data/wave/01sentence/003122.wav'

        else:
            RequestDic['user_id']      = manual_user_id
            RequestDic['prompt_num']   = 10000
            RequestDic['request_id']   = 'spk_test_request'

            prompt_uut_id = manual_user_id

            RequestDic['upload_prompt_text']   = TextWaveDict[prompt_uut_id][0]
            File_path                          = TextWaveDict[prompt_uut_id][1]



    with open(File_path,'rb') as fileObj:
        audio_data = fileObj.read()
        base64_data = base64.b64encode(audio_data).decode()

    RequestDic['prompt_wav']=base64_data

    payload = json.dumps(RequestDic)
    if 'jiutian.10086.cn' in base_url :
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + access_token
        }
    else:
        headers = {
            'Content-Type': 'application/json'
        }
    
    response = requests.request("POST", url, headers=headers, data=payload)

    print(f'{env_name} Upload_Prompt status_code = {response.status_code}')

    json_data       = response.json()
    print(f'Upload_Prompt status_code = {response.status_code}, Msg : {json_data["Msg"]}')

    # pdb.set_trace()

# ************************************************************Voice_Clone****************************************************************************************************

def Voice_Clone(Requst_Text = None, base_url = None, access_token = None, enable_sv=False, manual_user_id = None, manual_out_uttid = None):
    
    # default para
    default_streaming_mode  = False
    default_encoding        = 'raw'

    if interface_version == 2:
        default_sample_rate = 24000
    else:
        default_sample_rate = 32000

    url =  base_url + "jttts/Voice_Clone"
    RequestDic={}
    RequestDic['text']           = '孙悟空的笑话：土地公公，没有轻重音.'
    RequestDic['user_id']      =  'test_speaker' # 'TTS_006'  # F1 F2 F3 M1 dM2  liujia_L_origin_denoise_S0000_H0005_1preH  jinya_test

    RequestDic['request_id']   =   'request_001' + '_' + str(uuid.uuid4())
    if interface_version == 2:
        RequestDic['speed_ratio']        =   1.0 #
    else:
        RequestDic['speed']        =   5 
    # RequestDic['volume_ratio']       =   1.0
    RequestDic['sample_rate']  =   24000
    RequestDic['streaming_mode']  =   False
    RequestDic['encoding']    =   'wav'  # 'raw' or 'wav'
    RequestDic['filter_bracket_content'] = False  # True 时不合成括号及括号内文字

    # 配置超时时间（秒）
    CONNECT_TIMEOUT = 3  # 连接超时：建立TCP连接的最长时间
    READ_TIMEOUT = 5     # 读取超时：接收响应的最长时间
    TIMEOUT = (CONNECT_TIMEOUT, READ_TIMEOUT)


    payload = json.dumps(RequestDic)
    if 'jiutian.10086.cn' in base_url :
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + access_token
        }
    else:
        headers = {
            'Content-Type': 'application/json'
        }
    start_time = time.time()
    try:
        response = requests.post(url, headers=headers, data=payload, stream=RequestDic['streaming_mode'],timeout=TIMEOUT)
        # 检查状态码是否为200（根据接口实际情况调整，例如201、204等）
        if response.status_code == 200:
            # 可选：验证响应内容（根据业务需求定制）
            response_data = response.json()
            if response_data.get("status") == "healthy":
                return True
            else:
                print(f"健康检查失败：响应内容异常 - {response_data}")
        else:
            print(f"健康检查失败：状态码为 {response.status_code}")
    
    # 捕获超时异常
    except requests.exceptions.ConnectTimeout:
        print(f"健康检查失败：连接超时（超过 {CONNECT_TIMEOUT} 秒）")
    except requests.exceptions.ReadTimeout:
        print(f"健康检查失败：读取响应超时（超过 {READ_TIMEOUT} 秒）")
    
    # 捕获其他请求异常
    except requests.exceptions.RequestException as e:
        print(f"健康检查失败：请求异常 - {str(e)}")
    
    # 捕获JSON解析异常（若响应不是JSON格式）
    except ValueError:
        print("健康检查失败：响应内容不是有效的JSON格式")
    
    return False

if __name__ == "__main__":
    # 退出码：0表示服务正常，1表示异常
    exit(0 if Voice_Clone() else 1)
