from io import BytesIO
import numpy as np
import torch
import torchaudio
import scipy, ffmpeg
from jttts.tts_common.logger import logger 
from jttts.tts_model.tts_utils import add_aigc_chunk_to_wav

def convert_audio_bytes_to_tensor(raw_audio: bytes) -> torch.Tensor:
    """同步音频转换方法"""
    return torch.from_numpy(np.array(np.frombuffer(raw_audio, dtype=np.float32))).unsqueeze(0)

def convert_audio_tensor_to_bytes(tensor: torch.Tensor, format: str = None, sample_rate=24000, aigc_data: dict = None) -> bytes:
    """将音频Tensor转换为指定格式的字节流

    Args:
        tensor: 输入音频Tensor，形状需为 (channels, samples) 或 (samples,)
        format: 目标格式，支持 'wav', 'pcm', 'mp3'
        sample_rate: 采样率（默认16000）

    Returns:
        bytes: 编码后的音频字节流
    """
    # 统一Tensor形状为 (channels, samples)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 2:
        if tensor.size(0) > tensor.size(1):  # 假设输入为 (samples, channels)
            tensor = tensor.permute(1, 0)
    else:
        raise ValueError("Invalid tensor shape")

    if not format:
        return _encode_pcm(tensor, 16)
    if  format == 'wav':
        # bits_per_sample: PCM/WAV的量化位数（16或32）
        return _encode_wav(tensor, sample_rate, 16, aigc_data)
    elif format == 'pcm':
        return _encode_pcm(tensor, 16)
    elif format == 'raw':
        return _encode_pcm(tensor, 16)
    elif format == 'mp3':
        return _encode_mp3(tensor, sample_rate)
    else:
        return _encode_pcm(tensor, 16)

def _encode_wav(tensor: torch.Tensor, sr: int, bits: int, aigc_data: dict = None) -> bytes:
    """编码WAV格式"""
    if bits == 16:
        encoding = "PCM_S"
        bits_depth = 16
    elif bits == 32:
        encoding = "PCM_F"
        bits_depth = 32
    else:
        raise ValueError("Only 16/32-bit WAV supported")
    io_buffer = BytesIO()
    torchaudio.save(
        io_buffer,
        tensor,
        sr,
        format="wav",
        encoding=encoding,
        bits_per_sample=bits_depth,
    )
    io_buffer.seek(0)

    origin_io_buffer = io_buffer
    # If AIGC data is provided, add it to the WAV bytes
    if aigc_data is not None:
        try:
            io_buffer.seek(0)  # Rewind to the beginning to read the data
            wav_bytes = io_buffer.read()
            wav_bytes = add_aigc_chunk_to_wav(wav_bytes, aigc_data)
            # Create the final BytesIO buffer and write the (modified) WAV bytes to it
            io_buffer = BytesIO()
            io_buffer.write(wav_bytes)
            io_buffer.seek(0)  # Rewind for reading by the caller
        except Exception as e:
            io_buffer = origin_io_buffer
            io_buffer.seek(0)
            # log.info(f"Failed to add AIGC to audio: \n{e}")


    return io_buffer.getvalue()

def _encode_pcm(tensor: torch.Tensor, bits: int) -> bytes:
    """编码原始PCM数据"""
    assert tensor.dtype == torch.float32 or tensor.dtype == torch.float64, "输入张量应为浮点类型"
    np_array = tensor.cpu().numpy()
    np_array = np.clip(np_array, -1.0, 1.0)

    # 量化到目标位深
    if bits == 16:
        np_array = (np_array * 32767.0).astype(np.int16)
    elif bits == 32:
        np_array = (np_array * 2147483647.0).astype(np.int32)
    else:
        raise ValueError("Only 16/32-bit PCM supported")
    return np_array.tobytes()

def _encode_mp3(tensor: torch.Tensor, sr: int) -> bytes:
    """编码MP3格式"""
    buffer = BytesIO()

    # 注意：需要安装支持MP3编码的后端（如ffmpeg, libsox）
    torchaudio.save(
        buffer,
        tensor,
        sr,
        format="mp3",
        encoding="MP3",
    )
    buffer.seek(0)
    return buffer.getvalue()


# ****************************************************************************************

from grpc import aio
import tts_interface_pb2
import tts_interface_pb2_grpc
import uuid, time
import grpc, os

async def Upload_Prompt_grpc_Warmup(base_url=None):
    input_text_file = './build_in_speaker/warm_up/warm_up.txt'
    with open(os.path.join(input_text_file),encoding="utf-8",) as ttf:
        lines = ttf.readlines()
        for idx in range(len(lines)):
            utt_id = lines[idx].strip().split()[0]
            prompt_text = lines[idx].strip()[len(utt_id):].strip()

            async with aio.insecure_channel(base_url) as channel:
                RequestDic={}
                RequestDic['user_id']      = utt_id
                RequestDic['prompt_num']   = 10000
                RequestDic['request_id']   = 'warmup_speaker_request'
                RequestDic['upload_prompt_text']   = prompt_text
                File_path   = os.path.join('./build_in_speaker/warm_up/', utt_id + '.wav')


                stub = tts_interface_pb2_grpc.cloneStub(channel)
                with open(File_path,'rb') as fileObj:
                    audio_bytes = fileObj.read()

                future = stub.RegisterSpk(
                    tts_interface_pb2.RegisterSpeakerRequest(
                            request_id  =   RequestDic['request_id'],
                            user_id     =   RequestDic['user_id'], 
                            upload_prompt_text=RequestDic['upload_prompt_text'], 
                            prompt_wav=audio_bytes, 
                            prompt_num=10000,
                            )
                        )
                response = await future
                if response.ResultStatus == 0:
                    logger.info(f"[warm_up] RegisterSpk  success")
                else:
                    logger.info(f"[warm_up] RegisterSpk  Failed: {response.Msg}")


async def Voice_Clone_grpc_Warmup(base_url = None):
    def construct_tts_request(RequestDic):
        request = tts_interface_pb2.TtsServiceRequest()
        if 'request_id' in RequestDic.keys():
            request.request_id      = 'request_001' + '_' + str(uuid.uuid4())
        
        if 'text' in RequestDic.keys():
            request.text            = RequestDic['text']

        if 'user_id' in RequestDic.keys():
            request.user_id         = RequestDic['user_id']
        
        if 'if_ver' in RequestDic.keys():
            request.if_ver         = RequestDic['if_ver']
        
        if 'speed_ratio' in RequestDic.keys():
            request.speed_ratio         = RequestDic['speed_ratio']
        
        if 'volume_ratio' in RequestDic.keys():
            request.volume_ratio         = RequestDic['volume_ratio']

        if 'sample_rate' in RequestDic.keys():
            request.sample_rate         = RequestDic['sample_rate']

        if 'streaming_mode' in RequestDic.keys():
            request.streaming_mode         = RequestDic['streaming_mode']

        if 'encoding' in RequestDic.keys():
            request.encoding         = RequestDic['encoding']

        if 'filter_bracket_content' in RequestDic.keys():
            request.filter_bracket_content = RequestDic['filter_bracket_content']
        
        return request

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

    for sub_text in Text_sets:
        RequestDic={}
        RequestDic['text']           = sub_text
        RequestDic['user_id']      =   'warm_up_exp1'
        RequestDic['request_id']   =   'request_001' + '_' + str(uuid.uuid4())[:4]
        # RequestDic['speed_ratio']        =   1.0 #
        # RequestDic['volume_ratio']       =   1.0
        # RequestDic['sample_rate']  =   24000
        RequestDic['streaming_mode']  =   False
        RequestDic['encoding']    =   'pcm'  

        async with aio.insecure_channel(base_url) as channel:
            stub = tts_interface_pb2_grpc.cloneStub(channel)
            try:
                request = construct_tts_request(RequestDic)
                response_stream = stub.Inference(request)

                async for response in response_stream:
                    if response.ResultStatus == 0:
                        logger.info(f"[warming_up] tts success: ")
                    else:
                        logger.info(f"[warming_up] tts Failed: {response.Msg}")

            except grpc.RpcError as e:
                logger.error(f"[warming_up] RPC error occurred: {e.code()}: {e.details()}")
            except Exception as e:
                logger.error(f"[warming_up] Processing failed: {str(e)}", exc_info=True)
