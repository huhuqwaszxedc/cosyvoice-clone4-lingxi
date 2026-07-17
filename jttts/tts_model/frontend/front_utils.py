#!/usr/bin/python
# -*- encoding: utf-8 -*-

import re
from jttts.tts_common.logger import logger
import jttts.tts_model.frontend.config_frontend as cfg
import pdb
import os

def contains_letter_or_digit(s):
    pattern = r'[a-zA-Z0-9]'
    return bool(re.search(pattern, s))

def is_emoji(char):
    """
    判断给定的字符是否为 Emoji 表情。
    
    :param char: 单个字符
    :return: 如果是 Emoji 表情则返回 True，否则返回 False
    """
    # 获取字符的 Unicode 码点
    code_point = ord(char)
    
    # 检查是否在 Emoji Unicode 范围内
    emoji_ranges = [
        (0x1F600, 0x1F64F),  # Emoticons
        (0x1F300, 0x1F5FF),  # Miscellaneous Symbols and Pictographs
        (0x1F680, 0x1F6FF),  # Transport and Map Symbols
        (0x1F900, 0x1F9FF),  # Supplemental Symbols and Pictographs
        (0x2600, 0x26FF),    # Miscellaneous Symbols
        (0x2700, 0x27BF),    # Dingbats
        (0xFE00, 0xFE0F),    # Variation Selectors
        (0x1F1E6, 0x1F1FF),  # Flags (iOS)
    ]
    
    for start, end in emoji_ranges:
        if start <= code_point <= end:
            return True
    return False

def remove_commas_from_numbers(text):
    # 使用正则表达式找到所有形如带有逗号的数字，并替换掉逗号
    return re.sub(r'(\d),(\d)', r'\1\2', text)

def remove_markdown_characters(text):
    # 移除连续的两个或多个#
    text = re.sub(r'#{2,}', '', text)
    
    # 移除连续的两个或多个*
    text = re.sub(r'\*{2,}', '', text)
    
    # 移除形如 [数字] 的部分
    # 注意这里使用了非贪婪匹配 .*? 来确保只匹配最短的数字串
    text = re.sub(r'\[(\d+)]', '', text)
    
    # 清理多余的空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def convert_FullEnglish_to_Half(text):
    result = ""
    for char in text:
        code = ord(char)
        # 检查是否为全角英文字母
        if 0xFF21 <= code <= 0xFF3A or 0xFF41 <= code <= 0xFF5A:
            # 转换为半角英文字母
            code -= 0xFEE0
        result += chr(code)
    return result

# Regular expression matching whitespace:
def collapse_whitespace(text):
    _whitespace_re = re.compile(r"\s+")
    text = re.sub(_whitespace_re, " ", text) # 多个空格变成一个空格
    return text





if __name__ == '__main__':
    # sentence = "凡是过去30多年来在大陆发展的台湾同胞，不管哪个世代不管身在大陆何地，绝大多数都有一致高度的共识就是祖国必须统一也必然统一。"
    sentence = "一个人的世界也要 enjoy everyday"
    tn_sentence = ["everyday"]
    start_index = 0

