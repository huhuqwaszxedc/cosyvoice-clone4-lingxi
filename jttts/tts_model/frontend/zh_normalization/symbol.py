#!/usr/bin/python
# -*- encoding: utf-8 -*-

import re
from jttts.tts_common.logger import logger
import pdb


# ****************************将成对括号中的内容移除******************************************************************
# 定义支持的括号对映射（左括号: 右括号）
BRACKET_PAIRS = {
    '(': ')',
    '[': ']',
    '{': '}',
    '（': '）',
    '【': '】',
    '｛': '｝'
}

# 生成匹配所有成对括号的正则表达式
# 对每种括号对生成专用模式，确保左右括号匹配
patterns = []
for left, right in BRACKET_PAIRS.items():
    # 转义特殊字符，匹配成对括号及其中内容（非贪婪模式）
    patterns.append(re.escape(left) + r'.*?' + re.escape(right))

# 合并所有模式，编译为正则对象
RE_PAIRED_BRACKET = re.compile('|'.join(patterns))

def replace_paired_bracket(match):
    """替换函数：将匹配到的成对括号及内容替换为空字符串"""
    return ''

def remove_paired_bracket_content(text):
    """移除文本中所有成对出现的括号及其内部内容，保留不成对的括号"""
    # 处理嵌套括号：循环替换直到无匹配（每次处理最内层成对括号）
    while RE_PAIRED_BRACKET.search(text):
        # 使用指定格式：text = RE.sub(replace_func, text)
        text = RE_PAIRED_BRACKET.sub(replace_paired_bracket, text)
    return text


# ****************************移除不成对的括号左边或者右边的内容******************************************************************
# 反向映射（右括号: 左括号）
BRACKET_REVERSE_MAP = {v: k for k, v in BRACKET_PAIRS.items()}

def handle_incomplete_brackets(text):
    """
    专门处理因断句错误导致的不完整括号问题。
    策略：如果发现孤立的左括号，移除其后所有内容；如果发现孤立的右括号，移除其前所有内容。
    （如果同时存在孤立的左括号和右括号，且右括号在左括号之后，则处理逻辑会截取中间部分）
    
    Args:
        text (str): 输入的原始文本片段。
        
    Returns:
        str: 处理后的文本，移除了由不完整括号引发的问题部分。
    """
    # 寻找所有括号的位置和类型
    bracket_positions = []
    for i, char in enumerate(text):
        if char in BRACKET_PAIRS or char in BRACKET_REVERSE_MAP:
             bracket_positions.append((i, char))

    # 如果没有括号，直接返回原文本
    if not bracket_positions:
        return text

    # 使用集合来标记需要移除的字符索引
    to_remove_indices = set()

    # 正向扫描寻找孤立的左括号
    open_bracket_stack = []
    for pos, char in bracket_positions:
        if char in BRACKET_PAIRS:
            # 遇到左括号，入栈
            open_bracket_stack.append(pos)
        elif char in BRACKET_REVERSE_MAP:
            # 遇到右括号
            if open_bracket_stack and text[open_bracket_stack[-1]] == BRACKET_REVERSE_MAP[char]:
                # 匹配成功，弹出栈顶
                open_bracket_stack.pop()
            else:
                # 发现孤立的右括号，标记其前所有内容（包括自身）为待移除
                for p in range(pos + 1):
                    to_remove_indices.add(p)

    # 扫描结束后，栈中剩余的都是孤立的左括号
    for pos in open_bracket_stack:
        # 发现孤立的左括号，标记其后所有内容（包括自身）为待移除
        for p in range(pos, len(text)):
            to_remove_indices.add(p)

    # 按索引顺序构建新字符串
    result = ''.join([char for i, char in enumerate(text) if i not in to_remove_indices])
    
    # 可选：移除首尾空白
    # result = result.strip()
    
    return result




# ****************************类似这种GLM-4.6，移除横线 ******************************************************************

def replace_hyphen_after_capital_word(text: str) -> str:
    """
    纯标准 re 实现：查找「首大写字母 + 至少一个字母」后跟 '-' 的位置，逐个替换。
    兼容 Python 内置 re，无需安装 regex。
    """
    # 匹配：一个大写字母 + 至少一个字母 + '-'（捕获前面的单词和 '-'）
    # 使用非贪婪匹配，避免跨词
    pattern = r'([A-Z][A-Za-z]+)-'
    
    def replacer(match):
        word = match.group(1)
        return word + ' '  # 把 "GLM-" → "GLM "
    
    return re.sub(pattern, replacer, text)
