#coding:utf-8
#判断License是否过期
from jttts.tts_common.Aes     import Aescrypt
from Crypto.Cipher      import AES
import os
import subprocess
import json, time
from jttts.tts_common.logger import logger as log

import traceback
import signal

passwd          = "asdfwetyhjuytrfd"
iv              = "gfdertfghjkuyrtg"
aescryptor      = Aescrypt(passwd,AES.MODE_CBC,iv) # CBC模廾O


def check_license_process(license_file):
    if os.path.exists(license_file) is False:
        cmd = "ps -aux|grep python|awk '{print $2}'|xargs -i kill -9 {}"
        val = os.system(cmd)
        return False

    with open(license_file,'rb') as f:
        read_json_aes = f.read()
    read_json_aes = aescryptor.aesdecrypt(read_json_aes).decode('utf-8')
    license_info_dict = json.loads(read_json_aes)

    license_last_time   = license_info_dict["time_last"]
    license_end_time    = license_info_dict["time_end"]
    log.info(f"The license start time is {license_last_time}, end time is {license_end_time}")
    
    License_GPU_UUID_LIST = []
    License_GPU_UUID_LIST = license_info_dict['GPU_UUID'].keys()
    time_now = time.strftime("%Y%m%d%H%M%S", time.localtime())
    if len(License_GPU_UUID_LIST) > 1:
        nvidia_info = subprocess.Popen(['nvidia-smi --query-gpu=uuid --format=csv,noheader'],stdout=subprocess.PIPE,shell=True).communicate()
        nvidia_info = nvidia_info[0].decode('utf-8')
        gpu_list    = nvidia_info.split('\n')[0:-1]
        if len(gpu_list) == 0:
            print("Donot find GPU, please check")
            log.info("Donot find GPU, please check")
            cmd = "ps -aux|grep python|awk '{print $2}'|xargs -i kill -9 {}"
            val = os.system(cmd)
            return False

        log.info("Find the GPU LIST: {}".format(gpu_list))
        
        check_pass_uuid = []
        for gpuuuid in gpu_list:
            if gpuuuid not in License_GPU_UUID_LIST:
                print(f'GPU {gpuuuid} is not in license')
                log.error(f'GPU {gpuuuid} is not in license')
                cmd = "ps -aux|grep python|awk '{print $2}'|xargs -i kill -9 {}"
                val = os.system(cmd)
                return False
            else:
                check_pass_uuid.append(gpuuuid)

        for gpuuuid in check_pass_uuid:
            [gpu_time_last, gpu_time_end] = license_info_dict['GPU_UUID'][gpuuuid]
            if time_now > gpu_time_end or time_now < gpu_time_last:
                print(f"The {gpuuuid} license time end is {gpu_time_end}, please check!")
                log.info(f"The {gpuuuid} license time end is {gpu_time_end}, please check!")
                cmd = "ps -aux|grep python|awk '{print $2}'|xargs -i kill -9 {}"
                val = os.system(cmd)
                return False
            else:
                license_info_dict['GPU_UUID'][gpuuuid] = [time_now, gpu_time_end]
    else:
        if time_now > license_end_time or time_now < license_last_time:
            print(f"The license time end is {license_end_time}, please check!")
            log.info(f"The license time end is {license_end_time}, please check!")
            cmd = "ps -aux|grep python|awk '{print $2}'|xargs -i kill -9 {}"
            val = os.system(cmd)
            return False
        else:
            license_info_dict["time_last"] = time_now
            license_info_dict["time_end"]  = license_end_time
    
    # rewrite license file
    write_json = json.dumps(license_info_dict)
    write_json_aes  = aescryptor.aesencrypt(write_json)

    with open(license_file, 'wb') as f:
        f.write(write_json_aes)

    return True

def checklicense(license_file):
    while True:
        flag = check_license_process(license_file)
        if flag:
            time.sleep(60*60*2)
        else:
            cmd = "ps -aux|grep python|awk '{print $2}'|xargs -i kill -9 {}"
            val = os.system(cmd)
            break


if __name__ == "__main__":

    # para setting
    external_license_file = '/tts_license/tts_licensev2' # 宿主机映射到本机的license
    internal_file_path = './tts_licensev2' # 本地的license

    license_file_path = ''
    if os.path.exists(external_license_file):
        license_file_path = external_license_file
    elif os.path.exists(internal_file_path):
        license_file_path = internal_file_path
    else:
        print(f'[error] {license_file_path} is not exists !, please check')
        log.info("license_file is not exists!")
        cmd = "ps -aux|grep python|awk '{print $2}'|xargs -i kill -9 {}"
        val = os.system(cmd)
        exit()

    try:
        checklicense(license_file_path)
    except Exception as e:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
