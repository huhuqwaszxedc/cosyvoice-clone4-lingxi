#coding:utf-8
import torch
import os, glob, pdb
from jttts.tts_common.logger import logger as log

def move_asrmodel_to_cpu(model):
    if model is None :
        return
    cpu_device = torch.device('cpu')
    model.asr_model.model.model.model.to(cpu_device)
    model.asr_model.model.model.vad_model.to(cpu_device)
    model.asr_model.model.model.punc_model.to(cpu_device)
    log.info("Moved ASR models to CPU to save GPU memory.")

def move_asrmodel_to_gpu(model):
    if model is None :
        return
    gpu_device = torch.device('cuda')
    model.asr_model.model.model.model.to(gpu_device)
    model.asr_model.model.model.vad_model.to(gpu_device)
    model.asr_model.model.model.punc_model.to(gpu_device)
    log.info("Moved ASR models to GPU memory.")

def move_to_cpu(tts):
    cpu_device = torch.device('cpu')
    tts.set_device(cpu_device, False)
    tts.enable_half_precision(False, False)
    print("Moved TTS models to CPU to save GPU memory.")



def move_denoisemodel_to_cpu(model):
    if model is None :
        return
    if model.denoise_model is None:
        return
    cpu_device = torch.device('cpu')
    model.denoise_model.model.to(cpu_device)
    log.info("Moved Denoise models to CPU to save GPU memory.")

def move_denoisemodel_to_gpu(model):
    if model is None:
        return
    if model.denoise_model is None:
        return
    gpu_device = torch.device('cuda')
    model.denoise_model.model.to(gpu_device)
    log.info("Moved Denoise models to GPU memory.")


def move_svmodel_to_cpu(model):
    if model is None:
        return
    cpu_device = torch.device('cpu')
    model.sv_model.model.to(cpu_device)
    log.info("Moved sv models to CPU to save GPU memory.")

def move_svmodel_to_gpu(model):
    if model is None:
        return
    gpu_device = torch.device('cuda')
    model.sv_model.model.to(gpu_device)
    log.info("Moved sv models to GPU memory.")


# def move_to_original(tts: TTS, tts_config: TTS_Config):
#     tts.set_device(tts_config.device, False)
#     tts.enable_half_precision(tts_config.is_half, False)
#     print("Moved TTS models back to original device for performance.")


def find_folders_with_prefix(directory, prefix="CosyVoice-300M"):
    folder_list = []
    # 遍历指定目录及其子目录
    for root_dir, dirs, files in os.walk(directory):
        # 检查当前目录的名字是否以指定的前缀开头
        if root_dir.endswith(prefix):
            folder_list.append(root_dir)
        else:
            # 检查当前目录下的子目录名字是否以指定的前缀开头
            for dir_name in dirs:
                if dir_name.startswith(prefix):
                    folder_list.append(os.path.join(root_dir, dir_name))
    return folder_list


def scan_latest_modelcheckpoint(cp_dir, prefix="CosyVoice-300M"):
    all_folders = find_folders_with_prefix(cp_dir, prefix)
    
    model_version_list = []
    for sub_folder in all_folders: # CosyVoice-300M_v3.1_20240919_10DS
        sub_folder = sub_folder.rstrip('/')
        sub_folder = sub_folder.split('/')[-1]

        if len(sub_folder) <= len(prefix): # CosyVoice-300M
            model_version = '0.0'
        else:
            model_version = sub_folder.split('_')[1][1:]
            if '.' in model_version:
                xiaoshu = model_version.split('.')[-1]
                if len(xiaoshu) == 1: # 3.1
                    model_version = model_version[:-2] + '0' + model_version[-1]

            else: # CosyVoice-300M_libri_yjydata_fintune_8000framemax/
                model_version = '1.0'
        model_version_list.append(model_version)
    
    max_idx = 0
    max_version = 0.0
    for idx, sub_version in enumerate(model_version_list):
        if float(sub_version) > max_version:
            max_version = float(sub_version)
            max_idx = idx
    
    final_folder = all_folders[max_idx]
    log.info(f"The latest model is {final_folder} .")
    return final_folder
