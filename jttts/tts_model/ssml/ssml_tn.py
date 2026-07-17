#!/usr/bin/python
# -*- encoding: utf-8 -*-

import re, pdb
from typing import List
from jttts.tts_model.frontend.zh_normalization.num import verbalize_cardinal, verbalize_digit


def normalize_score_format(text):
    # 正则表达式匹配 "数字:数字" 格式，数字最多10位
    pattern = re.compile(r'(\d{1,10})[:：](\d{1,10})')
    
    def score_replacer(match):
        # 提取匹配的组
        first_number, second_number = match.groups()
        # 将时间转换为比分读法
        return f'{int(first_number)}比{int(second_number)}'
    
    # 替换文本中的 "数字:数字" 格式为比分读法
    normalized_text = pattern.sub(score_replacer, text)
    return normalized_text

def ssml_say_as_score(text):

    def replace_score(match):
            # 获取匹配的两个数字部分
            left, right = match.groups()
            left    = verbalize_cardinal(left)
            right   = verbalize_cardinal(right)
            return f"{left}比{right}"

    # 定义正则表达式模式，确保两边都是至少1位的数字
    pattern = r'(\d+)\:(\d+)'
    
    # 使用 re.sub 进行替换
    normalized_text = re.sub(pattern, replace_score, text)

    return normalized_text


def ssml_say_as_characters(text):
    # 定义标点符号到发音的映射
    punctuation_to_word = {
        '.': '句号', ',': '逗号', '?': '问号', '!': '感叹号', ':': '冒号', ';': '分号',
        '-': '短横线', '—': '破折号', '(': '左括号', ')': '右括号', '[': '左方括号',
        ']': '右方括号', '{': '左花括号', '}': '右花括号', '"': '引号', "'": '撇号',
        '/': '斜杠', '\\': '反斜杠', '@': '艾特', '#': '井号', '$': '美元符号',
        '%': '百分号', '^': '插入符号', '&': '俺得', '*': '星号', '+': '加号',
        '=': '等于', '<': '小于', '>': '大于', '_': '下划线', '~': '波浪线',
        '`': '重音符号', '|': '竖线'
    }


    result = []
    for char in text:
        if char.isalpha():  # 英文字母
            result.append(str(char))
        elif char in punctuation_to_word:  # 标点符号
            result.append(punctuation_to_word[char])
        elif not char.isspace():  # 忽略空格和其他空白字符
            result.append(char)
    
    return ' '.join(result)


def read_date_part(match):
    num = str(match.group(1))
    unit = match.group(2)

    if '年' in unit:
        num = verbalize_digit(num)
        return f"{num}年"
    elif '月' in unit:
        num = verbalize_cardinal(num)
        return f"{num}月"
    elif '日' in unit or '号' in unit:
        num = verbalize_cardinal(num)
        return f"{num}日" if '日' in unit else f"{num}号"


def ssml_say_as_date(text):
    text = text.replace('~', '至')
    # 定义正则表达式模式
    patterns = [
        (r'(\d{4})[./](\d{1,2})[./](\d{1,2})', r'\1年\2月\3日'),  # 处理"2018/03/03"、"1997.9.9"等格式
        (r'(\d{1,2})[./](\d{1,2})[./](\d{4})', r'\3年\1月\2日'),  # 处理"10/20/2018"格式
        (r'(\d{1,2})[./](\d{1,2})', r'\1月\2日'),  # 处理"10/20"格式
        (r'(\d+)(年|月|日)', r'\1\2')  # 处理"年"、"月"、"日"前面的数字
    ]
    
    # 依次应用每个正则表达式模式
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    
    # 定义正则表达式模式，确保匹配年、月、日前的数字
    pattern = r'(\d{1,4})(年|月|日|号)'
    return re.sub(pattern, read_date_part, text)
    
