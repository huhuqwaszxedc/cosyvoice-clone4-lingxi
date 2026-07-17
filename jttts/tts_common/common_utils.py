# coding=utf-8

import os, json, re
from jttts.tts_common.logger import logger
import time, glob

import subprocess

def load_json_file(json_file, is_log = False):
    user_define_polyphone_dict = {}
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r', encoding="utf-8") as fjson:
                cn_polyphone_info = json.load(fjson)
            user_define_polyphone_dict.update(cn_polyphone_info)
            if is_log:
                logger.info(f"load  {json_file} success")
        except:
            if is_log:
                logger.info(f"load  {json_file} failed")
    else:
        if is_log:
            logger.info(f"{json_file} is not exists !")
    return user_define_polyphone_dict

def is_valid_requestid(s):
    """
    检查字符串是否只包含大小写字母、数字、下划线和短横线。
    
    参数:
        s (str): 需要检查的字符串
    
    返回:
        bool: 如果字符串只包含允许的字符，则返回 True；否则返回 False。
    """
    # 正则表达式模式：只能包含大小写字母、数字、下划线和短横线
    pattern = r'^[a-zA-Z0-9_-]+$'
    
    return bool(re.match(pattern, s))



zh_pattern = re.compile("[\u4e00-\u9fa5]")
def is_chinese(word):
    global zh_pattern
    match = zh_pattern.search(word)
    return match is not None
def is_alphabet(char):
    if (char >= 'a' and char <= 'z') or (char >= 'A' and char <= 'Z'):
        return True
    else:
        return False

def is_engchar(c):
    if c >= 'a' and c <= 'z':
        return True
    elif c >= 'A' and c <= 'Z':
        return True
    else:
        return False

def count_chinese_characters(text):
    # 使用正则表达式匹配所有中文汉字
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    return len(chinese_pattern.findall(text))


def count_elements(sentence):
    # 统计中文汉字：匹配Unicode中的中文字符
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', sentence)
    chinese_count = len(chinese_chars)
    
    # 统计英文单词：匹配由字母组成的单词，不区分大小写
    # 先移除句子中的标点符号（保留数字和字母）
    cleaned_sentence = re.sub(r'[^\w\s\d]', ' ', sentence)
    # 查找所有英文单词
    english_words = re.findall(r'[a-zA-Z]+', cleaned_sentence)
    english_count = len(english_words)
    
    # 统计数字：每个数字字符单独计数（如123算3个）
    digits = re.findall(r'\d', sentence)
    digit_count = len(digits)
    
    return english_count, chinese_count, digit_count


def filter_punc(text): # remove the punc in the text
    new_text = ''
    for sub_word in text:
        if is_chinese(sub_word) or is_engchar(sub_word) or sub_word in [' ']:
            new_text += sub_word

    return new_text


def cer(r: list, h: list):
    """
    Calculation of CER with Levenshtein distance.
    """
    # initialisation
    import numpy
    d = numpy.zeros((len(r) + 1) * (len(h) + 1), dtype=numpy.uint16)
    d = d.reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i
    # computation
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    return d[len(r)][len(h)] / float(len(r))





def check_contain_chinese(check_str):
    if not isinstance(check_str, str):
        return False    
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

def check_contain_english(check_str):
    if not isinstance(check_str, str):
        return False    
    contain_en = bool(re.search('[a-zA-Z]', check_str))    
    return contain_en

def check_contain_digits(check_str):
    if not isinstance(check_str, str):
        return False    

    contain_di = bool(re.search('[0-9]', check_str))
    return contain_di

def check_contain_Korea(text):
    """
    检查文本中是否包含韩语字符。
    
    :param text: 待检查的文本字符串
    :return: 如果文本中包含韩语字符，则返回 True，否则返回 False
    """
    hangul_range_start = 0xAC00
    hangul_range_end = 0xD7AF
    
    for char in text:
        if hangul_range_start <= ord(char) <= hangul_range_end:
            return True
    return False

def check_contain_japanese(text):
    """
    检查文本中是否包含日语字符。
    
    :param text: 待检查的文本字符串
    :return: 如果文本中包含日语字符，则返回 True，否则返回 False
    """
    hiragana_range_start = 0x3040
    hiragana_range_end = 0x309F
    katakana_range_start = 0x30A0
    katakana_range_end = 0x30FF
    kanji_range_start = 0x4E00
    kanji_range_end = 0x9FFF
    
    for char in text:
        char_code = ord(char)
        if (hiragana_range_start <= char_code <= hiragana_range_end or
            katakana_range_start <= char_code <= katakana_range_end or
            kanji_range_start <= char_code <= kanji_range_end):
            return True
    return False



def check_contain_valid_str(check_str):
    if not isinstance(check_str, str):
        return False    
    valid_res = check_contain_english(check_str) or check_contain_chinese(check_str) or check_contain_digits(check_str) or check_contain_Korea(check_str) or check_contain_japanese(check_str)
    return valid_res


def get_dir_all_files(path, extension=None):
    """
    获取指定目录下的所有文件，并可选择按指定扩展名过滤。

    :param path: 要搜索的目录路径
    :param extension: 文件扩展名（例如 '.wav'），默认为 None，表示不进行扩展名过滤
    :return: 包含所有匹配文件路径的列表
    """
    filenames = []
    
    try:
        # 构建匹配模式
        if extension is not None:
            pattern = os.path.join(path, '**', f'*{extension}')
        else:
            pattern = os.path.join(path, '**', '*')
        
        # 使用 glob 搜索文件
        for filename in glob.iglob(pattern, recursive=True):
            if os.path.isfile(filename):
                filenames.append(filename)
        
        # 对文件名进行排序
        filenames = sorted(filenames)
        
        return filenames
    
    except FileNotFoundError:
        print(f"目录 {path} 不存在")
        return []
    except PermissionError:
        print(f"没有权限访问目录 {path}")
        return []



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

