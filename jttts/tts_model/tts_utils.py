# coding=utf-8

from io import BytesIO
import soundfile as sf
import subprocess
import numpy as np
import shlex
import time
import os
from datetime import datetime
import json, scipy
import re
import ffmpeg
import subprocess
import wave

from jttts.tts_common.logger import logger as log

#全半角替换
def full2half(s):
    n = ''
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xFEE0
        num = chr(num)
        n += num
    return n






#把句子按字分开，中文按字分，英文按单词，数字按空格
def is_chinese(char):
    if char >= '\u4e00' and char <= '\u9fa5':
        return True
    else:
        return False

def count_english_letters(text):
    """
    统计文本中包含的英文字母个数。
    
    :param text: 待统计的文本
    :return: 英文字母的总数
    """
    # 使用正则表达式匹配所有英文字母（包括大写和小写）
    letters = re.findall(r'[a-zA-Z]', text)
    
    # 返回匹配到的字母数量
    return len(letters)

def contains_chinese(text):
    # 正则表达式匹配中文字符
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    
    # 搜索字符串中是否有中文字符
    if pattern.search(text):
        return True
    else:
        return False

def is_alphabet(char):
    if (char >= 'a' and char <= 'z') or (char >= 'A' and char <= 'Z'):
        return True
    else:
        return False

def is_other(char):
    if not (is_chinese(char) or is_alphabet(char)):
        return True
    else:
        return False



def remove_quotes(text):
    remove_list = ['"', "'"]
    out_text = ''
    for idx in range(len(text)):
        sub_text = text[idx]
        if sub_text in remove_list:
            if idx > 0 and is_alphabet(text[idx-1]):
                out_text += ' '
        else:
            out_text += sub_text
    return out_text


#判断文本是否合法
punc_en = '!"$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
punc_ch = '＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。°C'
punct_list = punc_en + punc_ch
def is_legal(word):
    for ch in word:
        if ('\u4e00' <= ch <= '\u9fff') or ('a' <= ch <= 'z') or ('A' <= ch <= 'Z') or ('0' <= ch <= '9') or (ch in punct_list) or (ch.isspace()):
            #print(ch)
            flag = True
        else:
            #print(ch)
            flag = False
            break
    return flag


def reserve_support_word(word):
    new_word = ''
    IsContainValidSymbolFlag = False
    for ch in word:
        if ('\u4e00' <= ch <= '\u9fff') or ('a' <= ch <= 'z') or ('A' <= ch <= 'Z') or ('0' <= ch <= '9') or (ch in punct_list) or (ch.isspace()):
            new_word += ch
        
        if ('\u4e00' <= ch <= '\u9fff') or ('a' <= ch <= 'z') or ('A' <= ch <= 'Z') or ('0' <= ch <= '9'):
            IsContainValidSymbolFlag = True
    return new_word, IsContainValidSymbolFlag


# def replace_emoji_with_comma(text):
#     # 匹配 Emoji 的正则表达式模式
#     emoji_pattern = re.compile("["
#                                u"\U0001F600-\U0001F64F"  # 表情符号
#                                u"\U0001F300-\U0001F5FF"  # 符号与图形
#                                u"\U0001F680-\U0001F6FF"  # 交通与地图符号
#                                u"\U0001F1E0-\U0001F1FF"  # 国旗与地区标志
#                                u"\U00002702-\U000027B0"
#                                u"\U000024C2-\U0001F251"
#                                "]+", flags=re.UNICODE)
#     return emoji_pattern.sub(',', text)


import unicodedata

def replace_emoji_with_comma(text):
    result = []
    for idx, char in enumerate(text):
        if unicodedata.category(char).startswith('So'):  # 'So' 类别通常包含 Emoji
            if idx >0 and text[idx-1] in punct_list:
                pass
            elif idx >0 and idx < len(text)-1 and text[idx+1] in punct_list:
                pass
            elif idx > 0:
                result.append(',')

        else:
            result.append(char)
    return ''.join(result)




def speed_change(input_audio:np.ndarray, speed:float, sr:int):
    # 将 NumPy 数组转换为原始 PCM 流
    raw_audio = input_audio.astype(np.int16).tobytes()

    # 设置 ffmpeg 输入流
    input_stream = ffmpeg.input('pipe:', format='s16le', acodec='pcm_s16le', ar=str(sr), ac=1)

    # 变速处理
    output_stream = input_stream.filter('atempo', speed)

    # 输出流到管道
    out, _ = (
        output_stream.output('pipe:', format='s16le', acodec='pcm_s16le')
        .run(input=raw_audio, capture_stdout=True, capture_stderr=True)
    )

    # 将管道输出解码为 NumPy 数组
    processed_audio = np.frombuffer(out, np.int16)

    return processed_audio


import struct
def add_aigc_chunk_to_wav(wav_data: bytes, aigc_data: dict) -> bytes:
    """
    Adds or replaces an 'AIGC' chunk in the WAV header (before the 'data' chunk).
    The AIGC chunk contains a JSON-encoded metadata dictionary.
    
    :param wav_data: Original WAV file as bytes.
    :param aigc_data: Dictionary to embed as AIGC metadata.
    :return: Modified WAV bytes with AIGC chunk in header.
    """
    # --- Input validation ---
    if len(wav_data) < 12:
        return wav_data
    try:
        riff, size, riff_type = struct.unpack('<4sI4s', wav_data[:12])
    except struct.error:
        return wav_data
    if riff != b'RIFF' or riff_type != b'WAVE':
        return wav_data

    # --- Parse chunks to find 'data' and existing 'AIGC' ---
    offset = 12
    data_chunk_offset = None
    existing_aigc_offset = -1
    existing_aigc_size = 0

    while offset < len(wav_data):
        if offset + 8 > len(wav_data):
            break
        try:
            chunk_id, chunk_size = struct.unpack('<4sI', wav_data[offset:offset+8])
        except struct.error:
            break

        # Safety: skip huge chunks
        if chunk_size > 0x10000000:
            chunk_total = 8 + chunk_size
            if chunk_total % 2 == 1:
                chunk_total += 1
            offset += chunk_total
            continue

        chunk_total = 8 + chunk_size
        if chunk_total % 2 == 1:
            chunk_total += 1

        if chunk_id == b'AIGC':
            existing_aigc_offset = offset
            existing_aigc_size = chunk_size
        elif chunk_id == b'data':
            data_chunk_offset = offset
            break  # We only care about first 'data' chunk; stop here to insert before it

        offset += chunk_total

    if data_chunk_offset is None:
        return wav_data  # No 'data' chunk found; invalid WAV

    # --- Build AIGC chunk ---
    try:
        aigc_json = json.dumps(aigc_data, ensure_ascii=False, separators=(',', ':'))
    except Exception:
        aigc_json = '{"Label":"1","error":"serialize_failed"}'
    
    aigc_bytes = aigc_json.encode('utf-8')
    MAX_AIGC_SIZE = 65536  # 64KB limit for safety
    if len(aigc_bytes) > MAX_AIGC_SIZE:
        fallback = {"Label": "1", "error": "metadata_too_large"}
        aigc_bytes = json.dumps(fallback, ensure_ascii=False).encode('utf-8')[:MAX_AIGC_SIZE]

    # Create AIGC chunk: 'AIGC' + size + data
    aigc_chunk = b'AIGC' + struct.pack('<I', len(aigc_bytes)) + aigc_bytes
    # Pad to even length
    if len(aigc_chunk) % 2 == 1:
        aigc_chunk += b'\x00'

    # --- Insert or replace ---
    if existing_aigc_offset != -1:
        # Replace existing AIGC chunk
        before = wav_data[:existing_aigc_offset]
        after = wav_data[existing_aigc_offset + 8 + existing_aigc_size:]
        # Adjust for padding in original
        orig_chunk_total = 8 + existing_aigc_size
        if orig_chunk_total % 2 == 1:
            orig_chunk_total += 1
        after = wav_data[existing_aigc_offset + orig_chunk_total:]
        new_data = before + aigc_chunk + after
    else:
        # Insert before 'data' chunk
        before_data = wav_data[:data_chunk_offset]
        after_data = wav_data[data_chunk_offset:]
        new_data = before_data + aigc_chunk + after_data

    # --- Update RIFF size ---
    new_size = len(new_data) - 8
    final_data = riff + struct.pack('<I', new_size) + riff_type + new_data[12:]

    return final_data




def pack_ogg(io_buffer:BytesIO, data:np.ndarray, rate:int):
    with sf.SoundFile(io_buffer, mode='w', samplerate=rate, channels=1, format='ogg') as audio_file:
        audio_file.write(data)
    return io_buffer


def pack_raw(io_buffer:BytesIO, data:np.ndarray, rate:int):
    io_buffer.write(data.tobytes())
    return io_buffer


def pack_wav(io_buffer:BytesIO, data:np.ndarray, rate:int, aigc_data: dict = None):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format='wav')

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
            log.info(f"Failed to add AIGC to audio: \n{e}")

    return io_buffer

def pack_aac(io_buffer:BytesIO, data:np.ndarray, rate:int):
    process = subprocess.Popen([
        'ffmpeg',
        '-f', 's16le',  # 输入16位有符号小端整数PCM
        '-ar', str(rate),  # 设置采样率
        '-ac', '1',  # 单声道
        '-i', 'pipe:0',  # 从管道读取输入
        '-c:a', 'aac',  # 音频编码器为AAC
        '-b:a', '192k',  # 比特率
        '-vn',  # 不包含视频
        '-f', 'adts',  # 输出AAC数据流格式
        'pipe:1'  # 将输出写入管道
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer

def pack_audio(io_buffer:BytesIO, data:np.ndarray, rate:int, media_type:str, aigc_data: dict = None):
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate, aigc_data)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer

# from https://huggingface.co/spaces/coqui/voice-chat-with-mistral/blob/main/app.py
def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    # This will create a wave header then append the frame input
    # It should be first on a streaming wav file
    # Other frames better should not have it (else you will hear some artifacts each chunk start)
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    return wav_buf.read()




def convert_audio_sr(audio_data, src_sr, dst_sr, request_id = 'null'):
    current_millis = int(round(time.time() * 1000))
    timestamp = time.strftime("%m-%d_%H-%M-%S", time.localtime())
    timestamp = timestamp + '-' + str(current_millis)[-3:]

    inputfile  = str(request_id) + '-' + timestamp + '_src.wav'
    outputfile = str(request_id) + '-' + timestamp + '_dst.wav'
    sf.write(inputfile, audio_data, src_sr, "PCM_16")
    # command = ('sox -c 1 -b 16 {} -t wav {} rate {}'.format(inputfile, outputfile, dst_sr))
    # subprocess.call(shlex.split(command))
    cmd = f'sox {inputfile} -r {dst_sr} {outputfile}'
    os.system(cmd)

    converted_audio, sr = sf.read(outputfile)
    os.remove(inputfile)
    os.remove(outputfile)
    return converted_audio, sr




def audio_post_process(audio_data, origin_sample_rate,  req:dict):
    sr = origin_sample_rate
    if sr != req['sample_rate']:
        audio_data = scipy.signal.resample(audio_data, audio_data.shape[0] * req['sample_rate'] // sr)
        sr = req['sample_rate']
    audio_data_vol = audio_data * float(req['volume'])
    if np.max(np.abs(audio_data_vol)) > 0.97:
        audio_data = audio_data * 0.97 / np.max(np.abs(audio_data)) 
    else:
        audio_data = audio_data_vol
    audio_data = (audio_data * 32768).astype(np.int16)
    audio_data = np.ascontiguousarray(audio_data, dtype=np.int16)

    try:
        if float(req['speed']) != 1.0:
            audio_data = speed_change(audio_data, speed=float(req['speed']), sr=int(sr))
    except Exception as e:
        log.info(f"Failed to change speed of audio: \n{e}")
    
    audio_data = (audio_data).astype(np.int16)

    return audio_data, sr


def is_boolean(value):
    pattern = r"^(False|True)$"
    return bool(re.match(pattern, value, re.IGNORECASE))

def get_aigc_metadata(protocol='http', req= {}):
    now = datetime.now()
    sample_rate = req.get('sample_rate', 24000)
    encoding    = req.get('encoding', 'defaultencoding')
    current_time = now.strftime("%Y%m%d%H%M%S") + f"{now.strftime('%f')[:3]}" 
    aigc_ProduceID = f'clone6p1_{protocol}_{current_time}'
    aigc_ReserveCode1 = str(0) + str(sample_rate) + str(encoding)

    aigc_PropagatorID = f'clone6p1_{protocol}_prog_{current_time}'
    aigc_ReserveCode2 = str(0) + str(sample_rate) + str(encoding)

    aigc_metadata = {
        "Label": "1",
        "ContentProducer": "001191110102MAEP6BJ33600000",
        "ProduceID": aigc_ProduceID,  # 内容制作编码
        "ReserveCode1": aigc_ReserveCode1,
        "ContentPropagator": "001191110102MAEP6BJ33600000",
        "PropagatorID": aigc_PropagatorID,
        "ReserveCode2": aigc_ReserveCode2
    }
    aigc_metadata = None
    return aigc_metadata



# *****************************************************************************************************************************************

def text_split_as_punc(text: str, min_len = 25) :

    sentences = []
    punc_split_en = '!?:'
    punc_split_zh = '，。！？：'
    punc_split = punc_split_en + punc_split_zh
    sub_str = ''
    for sub_word in text:
        if sub_word in punc_split and len(sub_str) > min_len:
            sub_str += sub_word
            sentences.append(sub_str)
            sub_str = ''
        else:
            sub_str += sub_word

    if len(sub_str) > 0:
        sentences.append(sub_str)

    return sentences

# *****************************************************************************************************************************************

def split_text_to_word(sentence):
    # 匹配中文字符、英文单词、数字和标点符号的正则表达式
    pattern = r'([\u4e00-\u9fff])|([a-zA-Z0-9]+(?:[\'\-][a-zA-Z0-9]+)*)|([^\u4e00-\u9fff0-9a-zA-Z\s])'
    result = []
    
    # 处理所有匹配结果
    for match in re.finditer(pattern, sentence):
        # 中文单字
        if match.group(1):
            text = match.group(1)
        # 英文单词或数字
        elif match.group(2):
            text = match.group(2)
        # 标点符号
        elif match.group(3):
            text = match.group(3)
        else:
            continue
            
        result.append({
            'text': text,
            'start': match.start(),
            'end': match.end()
        })
    
    return result

def long_sentence_split(sentence, max_len=100,):
    # return [sentence]
    # pdb.set_trace()
    split_words = split_text_to_word(sentence)
    num = len(sentence) // max_len + 1
    end_index = len(sentence) // num

    result = []
    tmp_idx = 0
    tmp_endindex = end_index
    for item in split_words:
        if item['end'] > tmp_endindex:
            result.append(sentence[tmp_idx:item['end']].strip())
            tmp_idx = item['end']
            tmp_endindex += end_index

    if tmp_idx < split_words[-1]['end']:
        result.append(sentence[tmp_idx:split_words[-1]['end']].strip())

    return result



        # punctuation_mapping = {
        #     '，': ',',
        #     '。': '.',
        #     '！': '!',
        #     '？': '?',
        #     '；': ';',
        #     '：': ':',
        #     '“': '"',
        #     '”': '"',
        #     '‘': '\'',
        #     '’': '\'',
        #     '（': '(',
        #     '）': ')',
        #     '【': '[',
        #     '】': ']',
        #     '《': '<',
        #     '》': '>',
        #     '……': '...',
        #     '—': '-'
        # }

def split_text_by_semantics(text, min_len=30, max_len=90, should_merge_len=6):
    """
    将长文本按语义切割为指定长度范围的片段
    
    参数:
    text (str): 待分割的长文本
    min_len (int): 最小长度，默认为20
    max_len (int): 最大长度，默认为100
    
    返回:
    list: 分割后的文本片段列表
    """
    # 定义三级断句符号
    level1_punct = r'[。.！!？?;；\n]'  # 一级：句子结束符
    level2_punct = r'[，,、：:]'    # 二级：句子内部较大停顿
    level3_punct = r'[（(）)《》〈〉【】「」﹃﹄“”‘’〝〞﹏—]'  # 三级：括号、引号、破折号等
    
    # 按一级标点符号进行初步分割
    sentences = re.split(f'({level1_punct})', text)
    
    # 合并句子和标点符号
    merged_sentences = []
    for i in range(0, len(sentences), 2):
        if i < len(sentences) - 1:
            merged_sentences.append(sentences[i] + sentences[i+1])
        else:
            merged_sentences.append(sentences[i])
    
    result = []
    
    for sentence in merged_sentences:
        # 处理空句子
        if not sentence.strip():
            continue
            
        # 如果句子长度在范围内，直接添加
        if min_len <= len(sentence) <= max_len:
            result.append(sentence.strip())
            continue
            
        # 如果句子长度小于最小长度，尝试与前一个片段合并
        if len(sentence) < min_len:
            if result and len(result[-1]) + len(sentence) <= max_len:
                result[-1] = result[-1] + sentence
            else:
                result.append(sentence.strip())  # 不足最小长度但无其他选择
            continue
            
        # 按二级标点分割
        parts = re.split(f'({level2_punct})', sentence)
        current_part = ""
        
        for i in range(0, len(parts), 2):
            if i < len(parts) - 1:
                part = parts[i] + parts[i+1]  # 内容 + 标点
            else:
                part = parts[i]  # 最后一部分可能没有标点
                
            # 如果加入当前部分会超过最大长度，且当前已有内容，添加到结果
            if len(current_part) + len(part) > max_len and current_part:
                # 尝试按三级标点进一步分割当前部分
                sub_parts = re.split(f'({level3_punct})', current_part)
                if len(sub_parts) > 1:  # 如果能分割
                    temp_part = ""
                    for j in range(0, len(sub_parts), 2):
                        if j < len(sub_parts) - 1:
                            sub = sub_parts[j] + sub_parts[j+1]
                        else:
                            sub = sub_parts[j]
                            
                        if len(temp_part) + len(sub) <= max_len:
                            temp_part += sub
                        else:
                            if temp_part:
                                result.append(temp_part.strip())
                            temp_part = sub
                    
                    if temp_part:
                        current_part = temp_part
                    else:
                        current_part = ""
                        
                if current_part:
                    result.append(current_part.strip())
                current_part = part
            else:
                current_part += part
                
            # 如果当前部分达到最小长度，且下一部分存在，且加入下一部分会超过最大长度
            if (len(current_part) >= min_len and 
                i < len(parts) - 2 and 
                len(current_part) + len(parts[i+2]) > max_len):
                # 尝试按三级标点进一步分割
                sub_parts = re.split(f'({level3_punct})', current_part)
                if len(sub_parts) > 1:
                    temp_part = ""
                    for j in range(0, len(sub_parts), 2):
                        if j < len(sub_parts) - 1:
                            sub = sub_parts[j] + sub_parts[j+1]
                        else:
                            sub = sub_parts[j]
                            
                        if len(temp_part) + len(sub) <= max_len:
                            temp_part += sub
                        else:
                            if temp_part:
                                result.append(temp_part.strip())
                            temp_part = sub
                    
                    if temp_part:
                        result.append(temp_part.strip())
                        current_part = ""
                    else:
                        result.append(current_part.strip())
                        current_part = ""
                else:
                    result.append(current_part.strip())
                    current_part = ""
                
        # 添加剩余部分
        if current_part.strip():
            # 尝试按三级标点进一步分割
            sub_parts = re.split(f'({level3_punct})', current_part)
            if len(sub_parts) > 1:
                temp_part = ""
                for j in range(0, len(sub_parts), 2):
                    if j < len(sub_parts) - 1:
                        sub = sub_parts[j] + sub_parts[j+1]
                    else:
                        sub = sub_parts[j]
                        
                    if len(temp_part) + len(sub) <= max_len:
                        temp_part += sub
                    else:
                        if temp_part:
                            result.append(temp_part.strip())
                        temp_part = sub
                
                if temp_part:
                    result.append(temp_part.strip())
            else:
                result.append(current_part.strip())

    # 合并相邻的过短片段（如果有）
    final_result = []
    temp = ""
    for seg in result:
        if len(seg) < should_merge_len:
            temp += seg
        else:
            if temp:
                if len(temp.strip()) > max_len:
                    tmp_split_sentences = long_sentence_split(temp.strip(), max_len=max_len)
                    final_result.extend(tmp_split_sentences)
                else:
                    final_result.append(temp.strip())
            temp = seg
    
    if temp:
        if len(temp.strip()) > max_len:
            tmp_split_sentences = long_sentence_split(temp.strip(), max_len=max_len)
            final_result.extend(tmp_split_sentences)
        else:
            final_result.append(temp.strip())
    
    return final_result

# ******************************************************************************************************************
from pydub.silence import split_on_silence
from pydub import AudioSegment
import io


def RemoveSil_Process(audio, sample_rate):
    """
    去除音频中的静音段并将非静音片段拼接在一起。

    Args:
        audio (np.ndarray): 音频数据，一维数组（单声道）或二维数组（多声道），数值范围通常为 [-1, 1] 或整数 PCM。
        sample_rate (int): 采样率，例如 16000, 44100 等。

    Returns:
        tuple: (processed_audio, sample_rate) 处理后的音频数据（np.ndarray）和采样率。
            如果处理失败，返回 (None, None)。
    """
    # 确保音频是单声道，并转换为 16-bit PCM 格式（pydub 要求）
    if len(audio.shape) > 1:
        # 多通道：取平均值转为单声道
        audio = audio.mean(axis=1)

    # 假设输入是 float32 [-1.0, 1.0] 范围
    if audio.dtype != np.int16:
        audio = (audio * 32767).astype(np.int16)

    try:
        # 创建 AudioSegment 对象
        audio_segment = AudioSegment(
            audio.tobytes(),
            frame_rate=sample_rate,
            sample_width=audio.dtype.itemsize,
            channels=1
        )
    except Exception as e:
        print(f"Error creating AudioSegment: {e}")
        return None, None

    # 按静音分割
    segments = split_on_silence(
        audio_segment,
        min_silence_len=200,      # 最小静音长度（毫秒）
        silence_thresh=-50,       # 静音阈值（dBFS）
        seek_step=100,            # 搜索步长
        keep_silence=100          # 保留每段前后 100ms 的静音
    )

    if not segments:
        print("No non-silent segments found.")
        return None, None

    # 拼接所有非静音段
    combined = segments[0]
    for seg in segments[1:]:
        combined += seg

    # 将处理后的音频导出为内存中的 wav 数据
    buffer = io.BytesIO()
    combined.export(buffer, format="wav")

    # 重新从 buffer 读取为 AudioSegment，以便提取数据
    buffer.seek(0)
    processed_segment = AudioSegment.from_wav(buffer)

    # 提取音频数据为 numpy 数组
    raw_data = np.frombuffer(processed_segment.raw_data, dtype=np.int16)
    processed_audio = raw_data.astype(np.float32) / 32767.0  # 转回 [-1, 1]

    # 返回处理后的音频数据和采样率
    return processed_audio, processed_segment.frame_rate



def generate_unique_id():
    """
    生成一个基于时间戳的唯一ID，精确到微秒。
    格式示例: 20260205_150322_123456
    """
    # 获取当前时间戳（浮点数，包含小数部分的秒）
    now_timestamp = time.time()
    
    # 将时间戳转换为本地时间元组
    local_time = time.localtime(now_timestamp)
    
    # 格式化基础时间部分
    time_str = time.strftime("%Y%m%d_%H%M%S", local_time)
    
    # 计算微秒部分 (0-999999)
    microseconds = int((now_timestamp - int(now_timestamp)) * 1_000_000)
    
    # 拼接成唯一的ID
    unique_id = f"{time_str}_{microseconds:06d}"
    
    return unique_id
