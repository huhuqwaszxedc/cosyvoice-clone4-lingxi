#coding:utf-8

# jttts/request_processor.py

import os
import time
import json
import base64
from io import BytesIO
import soundfile as sf
import torchaudio
import numpy as np
from jttts.tts_common.logger import logger 
from jttts.config import GlobalConfigInst
from jttts.tts_common.localhost_request import Localhost_GetAsrResult, Localhost_GetDenoiseResult
from jttts.tts_common.common_utils import filter_punc, cer
from tempfile import NamedTemporaryFile
from jttts.tts_engine.external_models.model import RemoveSil_Model
from typing import Tuple, Union, Dict, Any
from jttts.tts_common.common_utils import check_contain_valid_str


def convert_to_single_channel_audiobytes(src_audio_path = None):
    audio_np, sample_rate = sf.read(src_audio_path)
    if len(audio_np.shape) == 2:
        audio_np = np.mean(audio_np,axis=1)

    buffer = BytesIO()
    sf.write(buffer, audio_np, sample_rate, format='WAV', subtype='PCM_16')
    audio_bytes = buffer.getvalue()

    return audio_bytes, sample_rate


def truncate_large_values(data_dict, max_length=100, show_prefix=10):
    """
    对字典中每个值进行检查：
      - 如果 str(value) 长度 > max_length，则替换为前 show_prefix 个字符 + '...'
      - 否则保留原值（不转字符串，保持类型）
    """
    safe_dict = {}
    for key, value in data_dict.items():
        str_val = str(value)
        if len(str_val) > max_length:
            safe_dict[key] = str_val[:show_prefix] + "..."
        else:
            safe_dict[key] = value  # 保持原始类型，便于后续 json.dumps 正确显示
    return safe_dict

# ==============================================================================
# 1. 注册音色的请求参数检查（同步，协议无关）
# ==============================================================================

prompt_text_list = ['海洋中的每一个角落都有其独特的美丽和价值,它是地球上最宝贵的资源之一.',     # 0
                    '人生就像一场马拉松比赛,不是速度最快的人获胜,而是坚持到最后的人.',          # 1
                    '无论你遇到什么困难和挑战,都要保持信念和勇气,不断前行.',                    # 2
                    '在大自然的怀抱中,我们可以感受到生命的力量和美好,让我们的心灵得到升华.',     # 3
                    '大自然是一个神奇的世界,充满了各种美丽的景色.',                             # 4
                    '美国对港澳政策不会改变.',                                                  # 5 tts042
                ] 

def _validate_and_normalize_upload_prompt(
    raw_payload: Dict[str, Any]
) -> Tuple[bool, Union[Dict[str, Any], str]]:
    """
    验证并标准化上传提示音请求参数。
    
    Args:
        raw_payload: 原始请求字典，需包含以下字段（可来自 Pydantic 或 gRPC）:
            - request_id (str, optional)
            - user_id (str, optional)
            - speaker_id (str, optional)
            - prompt_num (int, optional)
            - upload_prompt_text (str, optional)
            - prompt_wav (str, base64, optional)
            - bypass_mode (int, optional)

    Returns:
        (True, normalized_dict) if valid
        (False, error_message) if invalid
    """
    # 安全获取字段，避免 KeyError
    request_id          = raw_payload.get("request_id", "null") or "null"
    user_id             = raw_payload.get("user_id")
    speaker_id          = raw_payload.get("speaker_id")
    prompt_num          = raw_payload.get("prompt_num")
    upload_prompt_text  = raw_payload.get("upload_prompt_text", "null") or "null"
    wave_data           = raw_payload.get("prompt_wav")
    bypass_mode         = raw_payload.get("bypass_mode", 0)

    if speaker_id:
        if len(speaker_id.strip()) > 0:
            user_id = speaker_id

    # 特殊映射（F/M → F2/M2）
    if user_id == "F":
        user_id = "F2"
    elif user_id == "M":
        user_id = "M2"

    # 校验 user_id
    if not user_id or user_id.strip() == "":
        return False, "user_id is required"
    if " " in user_id:
        return False, "The blank space in user_id is not required"

    # === 2. prompt_num 与 prompt_text 逻辑 ===
    if prompt_num is None:
        return False, "prompt_num is required"

    if prompt_num == 10000:
        if upload_prompt_text == "null" or not upload_prompt_text.strip():
            return False, f"upload_prompt_text is required, user_id={user_id}"
        prompt_text = upload_prompt_text
    elif prompt_num == 20000:
        prompt_text = "asr_translation"
    else:
        if not isinstance(prompt_num, int):
            return False, f"prompt_num must be integer,user_id={user_id}"
        if prompt_num < 0 or prompt_num >= len(prompt_text_list):
            return False, f"prompt_num is not supported,user_id={user_id}"
        prompt_text = prompt_text_list[prompt_num]

    # === 3. 音频校验 ===
    if not wave_data :
        return False, f"prompt_wav is required, user_id={user_id}"

    # === 4. 构建标准化输出 ===
    normalized = {
        "request_id": request_id,
        "user_id": user_id,
        "prompt_num": prompt_num,
        "prompt_text": prompt_text,
        "prompt_wav": wave_data,  # base64 string
        "bypass_mode": bypass_mode,
    }

    return True, normalized

# ==============================================================================
# 2. 语音合成的请求参数检查（同步，协议无关）
# ==============================================================================


def _prepare_voice_clone_request(
    raw_payload: Dict[str, Any]
) -> Tuple[bool, Union[Dict[str, Any], str]]:
    """
    标准化并验证 Voice Clone 请求参数，加载声纹信息。
    
    Args:
        raw_payload: 原始请求字典，需包含以下字段（可来自 Pydantic 或 gRPC）:
            - request_id (str)
            - text (str, optional)
            - text_pinyins (str, optional)
            - user_id (str, optional)
            - speaker_id (str, optional)
            - speed_ratio (float)
            - volume_ratio (float)
            - sample_rate (int)
            - streaming_mode / Streaming_mode (bool)
            - encoding (str)
            - language (str)
            - debug_mode (int)

    Returns:
        (True, enriched_req_dict) if valid
        (False, error_message) if invalid
    """
    # 安全提取字段（提供默认值）
    req = {
        "request_id":   raw_payload.get("request_id", "null") or "null",
        "text":         raw_payload.get("text"),
        "text_pinyins": raw_payload.get("text_pinyins"),
        "user_id":      raw_payload.get("user_id"),
        "speaker_id":   raw_payload.get("speaker_id"),
        "speed_ratio":  raw_payload.get("speed_ratio", GlobalConfigInst.default_speed),
        "volume_ratio": raw_payload.get("volume_ratio", 1.0),
        "sample_rate":  raw_payload.get("sample_rate", GlobalConfigInst.default_sample_rate),
        "streaming_mode":   raw_payload.get("streaming_mode", False) or raw_payload.get("Streaming_mode", False),
        "encoding":         raw_payload.get("encoding", GlobalConfigInst.default_encoding),
        "language":         raw_payload.get("language", "auto"),
        "debug_mode":       raw_payload.get("debug_mode", 0),
    }

    for key in raw_payload:
        if key not in req:
            req[key] = raw_payload[key]


    request_id = req["request_id"]

    # === 1. user_id 处理：优先 user_id，fallback 到 speaker_id ===
    if  req["speaker_id"]:
        if len(req["speaker_id"].strip()) > 0:
            req["user_id"] = req["speaker_id"]

    # 特殊映射
    user_id_map = {
            "F": "F2", 
            "M": "M2", 
            "006": "jinya"
            }
    if req["user_id"] in user_id_map:
        req["user_id"] = user_id_map[req["user_id"]]

    user_id = req["user_id"]

    # 校验 user_id
    if not user_id or user_id.strip() == "":
        return False, "user_id is required"
    if " " in user_id:
        return False, "The blank space in user_id is not required"

    # === 2. speed 校验 ===
    speed = req["speed_ratio"]
    try:
        speed_float = float(speed) if speed is not None else 1.0
        if speed_float < 0.25 or speed_float > 4.0:
            return False, "speed must be between 0.25 and 4.0"
        req["speed"] = speed_float
    except (TypeError, ValueError):
        req["speed"] = 1.0

    # === 3. volume 校验（宽松处理）===
    try:
        req["volume"] = float(req["volume_ratio"])
    except (TypeError, ValueError):
        req["volume"] = 1.0

    # === 4. 文本校验 ===
    text = req["text"]
    # text_pinyins = req["text_pinyins"]
    if text in [None, ""]:
        return False, "text is required"
    if not check_contain_valid_str(text):
        return False, f"text is not contain valid symbol, text={text}"

    # === 5. encoding 校验 ===
    if req["encoding"] not in ["pcm","wav", "raw"]:
        return False, f"input encoding ({req['encoding']})error, only support : raw \ wav \ pcm"

    # === 6. language 校验 ===
    # supported_langs = {"zh", "en", "jp", "yue", "ko", "zh-en", "auto"}
    # if req["language"] not in supported_langs:
    #     return False, "input language error, please check!"

    # === 7. 加载声纹信息（spk_info.json）===
    json_file = os.path.join(GlobalConfigInst.speaker_info_dir, f"{user_id}.json")
    speaker_info_dir = GlobalConfigInst.speaker_info_dir

    if not os.path.exists(json_file):
        json_file = os.path.join(GlobalConfigInst.build_in_speaker_info_dir, f"{user_id}.json")
        speaker_info_dir = GlobalConfigInst.build_in_speaker_info_dir

    if not os.path.exists(json_file):
        return False, f"The Prompt Audio is not upload! user_id={user_id}"

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            spk_info = json.load(f)
    except Exception as e:
        return False, f"Failed to load speaker info: {e}, user_id={user_id}"

    # 修复 prompt_wav 路径（兼容旧格式）
    prompt_wav = spk_info.get("prompt_wav", "")
    if "./speaker_info/" in prompt_wav:
        prompt_wav = prompt_wav.replace("./speaker_info/", "").replace("speaker_info/", "")

    refer_wav_path = os.path.join(speaker_info_dir, prompt_wav)
    if not os.path.exists(refer_wav_path):
        return False, f"The Prompt Audio file is missing! user_id={user_id}"

    # 注入关键字段
    req["ref_audio_path"] = refer_wav_path
    req["prompt_text"] = spk_info.get("prompt_text", "")

    return True, req





# ==============================================================================
# 1. 统一 ASR 质检函数（同步，协议无关）
# ==============================================================================
def _perform_asr_quality_check(
    prompt_wav_bytes: bytes,
    prompt_text: str,
    request_id: str = "null",
    user_id: str = "unknown",
    bypass_mode: int = 0,
) -> tuple[bool, str, str]:
    """
    执行 ASR 质检，返回 (is_passed: bool, message: str, final_prompt_text: str)

    - 若 bypass_mode != 0 或质检关闭，直接通过
    - 支持 'asr_translation' 模式：用 ASR 结果作为最终文本
    - 所有逻辑同步执行，可在任意线程调用
    """
    if not GlobalConfigInst.prompt_quality_check:
        return True, "Quality check disabled", prompt_text

    if bypass_mode != 0:
        return True, "Bypass mode enabled", prompt_text

    # 调用 ASR 服务
    base64_data = base64.b64encode(prompt_wav_bytes).decode()
    text_asr = Localhost_GetAsrResult({"audio": base64_data})
    text_asr = text_asr.strip()

    if not text_asr.strip():
        return False, "ASR result is empty", prompt_text

    # 处理 'asr_translation' 模式
    final_prompt_text = prompt_text
    if prompt_text == "asr_translation":
        final_prompt_text = text_asr

    # 文本清洗
    prompt_clean = filter_punc(final_prompt_text)
    asr_clean = filter_punc(text_asr)


    # 长度差异检查（容忍 ±2 字符）
    diff_len = abs(len(prompt_clean) - len(asr_clean))
    cer_value = cer(prompt_clean, asr_clean)
    logger.info(f"asr success, request_id: {request_id}, user_id: {user_id}, prompt_text : {prompt_text},Text_asr: {text_asr}, Prompt_cer: {cer_value}， diff_len = {diff_len}")

    if diff_len > 2:
        return False, f"Text length mismatch (diff={diff_len})", prompt_text

    # CER 检查    
    if cer_value > GlobalConfigInst.prompt_cer_threshold:
        return False, f"request_id: {request_id}, User_id: {user_id}, CER too high, prompt_text : {prompt_text},Text_asr: {text_asr}, Prompt_cer: {cer_value}", prompt_text

    return True, "Passed", final_prompt_text


# ==============================================================================
# 2. 统一后台处理核心（同步，协议无关）
# ==============================================================================
def _core_post_process(
    user_id: str,
    prompt_wav_bytes: bytes,
    prompt_text: str,
    prompt_num: int,
    bypass_mode: int = 0,
    request_id: str = "null",
    cosyvoice_frontend=None,
    sample_rate_model: int = 16000,
):
    """
    核心同步处理逻辑：降噪 → 去静音 → 保存 spk_info.json → 注册声纹
    此函数不包含任何 async/await，也不依赖 asyncio。
    """
    try:
        current_audio_bytes = prompt_wav_bytes
        prompt_use_wav_path = f"./prompt_wavs/{user_id}.wav"


        # 清理之前存在的wav文件
        for user_id_wave_name in [user_id + "-denoise.wav", user_id + "-denoise-rmsil.wav", user_id + "-rmsil.wav"]:
            remove_file_path = os.path.join(GlobalConfigInst.speaker_info_dir, "./prompt_wavs", user_id_wave_name)
            if os.path.exists(remove_file_path):
                os.remove(remove_file_path)

        # === 1. 【可选】降噪 ===
        if GlobalConfigInst.prompt_denoise and bypass_mode == 0:
            base64_data = base64.b64encode(current_audio_bytes).decode()
            denoise_audio, denoise_sr = Localhost_GetDenoiseResult({"audio": base64_data})
            denoise_output_file = os.path.join(
                GlobalConfigInst.speaker_info_dir, "prompt_wavs", f"{user_id}-denoise.wav"
            )
            sf.write(denoise_output_file, denoise_audio, denoise_sr, "PCM_16")
            with open(denoise_output_file,'rb') as fileObj:
                current_audio_bytes = fileObj.read()
            prompt_use_wav_path = f"./prompt_wavs/{user_id}-denoise.wav"

        # === 2. 【可选】去除静音 ===
        if GlobalConfigInst.prompt_remove_sil and bypass_mode == 0:
            with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(current_audio_bytes)
                tmp_path = tmp.name
            try:
                audio_np, sr = sf.read(tmp_path)
            finally:
                os.unlink(tmp_path)  # 手动清理


            rmsil_model = RemoveSil_Model()
            rmsil_audio, rmsil_sr = rmsil_model.inference(audio_np, sr)

            # 构建输出路径
            if "-denoise" in prompt_use_wav_path:
                output_name = f"{user_id}-denoise-rmsil.wav"
            else:
                output_name = f"{user_id}-rmsil.wav"
            rmsil_output_file = os.path.join(GlobalConfigInst.speaker_info_dir, "prompt_wavs", output_name)
            prompt_use_wav_path = f"./prompt_wavs/{output_name}"

            sf.write(rmsil_output_file, rmsil_audio, rmsil_sr, "PCM_16")

        # === 3. 保存最终 spk_info.json ===
        json_file = os.path.join(GlobalConfigInst.speaker_info_dir, f"{user_id}.json")
        spk_info = {
            "user_id": user_id,
            "prompt_wav": prompt_use_wav_path,
            "prompt_num": prompt_num,
            "prompt_text": prompt_text,
            "date": time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()),
        }
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(spk_info, f, ensure_ascii=False, indent=4)

        # === 4. 【可选】注册到 CosyVoice frontend ===
        if cosyvoice_frontend is not None:
            final_wav_full_path = os.path.join(GlobalConfigInst.speaker_info_dir, prompt_use_wav_path)
            cosyvoice_frontend.generate_spk_info_by_audiopath(user_id, prompt_text, final_wav_full_path, output_dir = GlobalConfigInst.speaker_info_dir)

        logger.info(f"Post-processing success, request_id: {request_id}, user_id: {user_id}")

    except Exception as e:
        logger.info(
            f"Post-processing failed, request_id: {request_id}, user_id: {user_id}, error: {e}")


# ==============================================================================
# 3. 协议适配层：同步包装（供 FastAPI background_tasks 使用）
# ==============================================================================
def run_denoise_and_register_in_background(
    user_id: str,
    prompt_wav_bytes: bytes,
    prompt_text: str,
    prompt_num: int,
    bypass_mode: int = 0,
    request_id: str = "null",
    cosyvoice_frontend=None,
    sample_rate_model: int = 16000,
):
    """FastAPI background_tasks 兼容的同步包装器"""
    _core_post_process(
        user_id=user_id,
        prompt_wav_bytes=prompt_wav_bytes,
        prompt_text=prompt_text,
        prompt_num=prompt_num,
        bypass_mode=bypass_mode,
        request_id=request_id,
        cosyvoice_frontend=cosyvoice_frontend,
        sample_rate_model=sample_rate_model,
    )


# ==============================================================================
# 4. 协议适配层：异步包装（供 gRPC asyncio.create_task 使用）
# ==============================================================================
import asyncio


async def run_denoise_and_register_in_background_async(
    user_id: str,
    prompt_wav_bytes: bytes,
    prompt_text: str,
    prompt_num: int,
    bypass_mode: int = 0,
    request_id: str = "null",
    cosyvoice_frontend=None,
    sample_rate_model: int = 16000,
):
    """asyncio 兼容的异步包装器"""
    await asyncio.to_thread(
        _core_post_process,
        user_id=user_id,
        prompt_wav_bytes=prompt_wav_bytes,
        prompt_text=prompt_text,
        prompt_num=prompt_num,
        bypass_mode=bypass_mode,
        request_id=request_id,
        cosyvoice_frontend=cosyvoice_frontend,
        sample_rate_model=sample_rate_model,
    )



