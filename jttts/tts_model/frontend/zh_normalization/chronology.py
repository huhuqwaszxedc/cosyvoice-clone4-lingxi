#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File    :   arpabet.py
@Time    :    2025.02   
@Author  :   yanghuibao
@Version :   1.0
@Brief   :    
@Contact :   yanghuibao
'''
import re

from .num import DIGITS
from .num import num2str
from .num import verbalize_cardinal
from .num import verbalize_digit
import pdb

def _time_num2str(num_string: str) -> str:
    """A special case for verbalizing number in time."""
    result = num2str(num_string.lstrip('0'))
    if num_string.startswith('0'):
        result = DIGITS['0'] + result
    return result





# 时间范围，如8:30-12:30
RE_TIME_RANGE = re.compile(r'([0-1]?[0-9]|2[0-4])'
                           r':([0-5][0-9])'
                           r'(:([0-5][0-9]))?'
                           r'(~|-)'
                           r'([0-1]?[0-9]|2[0-4])'
                           r':([0-5][0-9])'
                           r'(:([0-5][0-9]))?')

# *******************比值表达式(当句子中包含“比分|比例|比值”等关键词时)*********************************************************

# 定义关键字
ratio_keywords = "比分|比例|比值|比赛|球队|上半场|下半场|分数|得分"
# 编译关键字匹配正则
RATIO_KEYWORD = re.compile(ratio_keywords)
# 编译比分匹配正则（支持多种冒号类型）
RE_RATIO = re.compile(r'(\d+)[\:\∶：](\d+)')

def replace_ratio(match):
    """替换函数：将匹配到的比分 (X:Y) 转换为中文读法 (X比Y)"""
    num1, num2 = match.groups()
    ch_num1 = num2str(num1)
    ch_num2 = num2str(num2)
    return f"{ch_num1}比{ch_num2}"


# *******************时刻表达式*********************************************************
RE_TIME = re.compile(r'([0-1]?[0-9]|2[0-4])'
                     r'[:∶：]([0-5][0-9])'
                     r'([:∶：]([0-5][0-9]))?')

def replace_time(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    # pdb.set_trace()
    is_range = len(match.groups()) > 5

    hour = match.group(1)
    minute = match.group(2)
    second = match.group(4)


    if is_range:
        hour_2 = match.group(6)
        minute_2 = match.group(7)
        second_2 = match.group(9)

    if hour == '2':
        result = f"两点"
    else:
        result = f"{num2str(hour)}点"
    if minute.lstrip('0'):
        if int(minute) == 30:
            result += "半"
        else:
            result += f"{_time_num2str(minute)}分"
    if second and second.lstrip('0'):
        result += f"{_time_num2str(second)}秒"

    if is_range:
        result += "至"
        if hour_2 == '2':
            result += f"两点"
        else:
            result += f"{num2str(hour_2)}点"
        if minute_2.lstrip('0'):
            if int(minute) == 30:
                result += "半"
            else:
                result += f"{_time_num2str(minute_2)}分"
        if second_2 and second_2.lstrip('0'):
            result += f"{_time_num2str(second_2)}秒"

    return result

RE_TRANS_PINYIN = re.compile(r'[abcdefgɡhijklmnopqrstuvwxyz]*[àáèéìíòóùúüāăēěīōŏūŭǎǐǒǔǘǚǜǹḿ][abcdefgɡhijklmnopqrstuvwxyz]*')
RE_TRANS_JIANPIN = re.compile(r'(简\s*拼[：:]\s*)([abcdefghijklmnopqrstuvwxyz]+)')
RE_YEAR_RANGE = re.compile(r'(?<![-~—])(\d{4})[-~—](\d{4})年')
RE_YEAR_MAN_RANGE = re.compile(r'[(（]([-])?(\d{2,4})(年)?[-~—]([-])?(\d{2,4})(年)?[)）]')

RE_MONTH_RANGE = re.compile(r'(?<![-~—])(0?[1-9]|1[0-2])[-~—](0?[1-9]|1[0-2])月')

RE_DAY_RANGE = re.compile(r'(?<![-~—])((0?[1-9])|((1|2)[0-9])|30|31)[-~—]((0?[1-9])|((1|2)[0-9])|30|31)([日号])')

# RE_DATE = re.compile(r'(\d{4}|\d{2})年'
# 未来10年,以15年为期
# RE_DATE = re.compile(r'(?<![-~—\d来])([-——]+)?(\d{4}|[6-9]\d)年(?![以来中里代])'
#                      r'(?<![-~—\d])((0?[1-9]|1[0-2])月)?'
#                      r'(?<![-~—\d])(((0?[1-9])|((1|2)[0-9])|30|31)([日号]))?')


RE_DATE = re.compile(r'(?<![-~—\d])([-——]+)?(\d{4}|[6-9]\d)年(?![代])'
                     r'(?<![-~—\d])((0?[1-9]|1[0-2])月)?'
                     r'(?<![-~—\d])(((0?[1-9])|((1|2)[0-9])|30|31)([日号]))?')


RE_EVENT = re.compile(r'(0?[1-9]|1[0-2])[·]((((1|2)[0-9])|30|31)|(0?[1-9]))')
RE_EVENT2 = re.compile(r'(0?[1-9]|1[0-2])((((1|2)[0-9])|30|31)|(0?[1-9]))(声明|案件|事件|法案)')






# 带声调字符。
phonetic_symbol = {
    "ā": "a1", "á": "a2", "ǎ": "a3", "à": "a4","ɑ": "a5",
    "ē": "e1",  "é": "e2", "ě": "e3",  "è": "e4",
    "ō": "o1",  "ó": "o2", "ǒ": "o3", "ò": "o4",
    "ī": "i1", "í": "i2", "ǐ": "i3", "ì": "i4",
    "ū": "u1", "ú": "u2", "ǔ": "u3", "ù": "u4",
    # üe
    "ü": "v", "ǖ": "v1", "ǘ": "v2", "ǚ": "v3", "ǜ": "v4",
    "ń": "n2", "ň": "n3", "ǹ": "n4",
    "m̄": "m1", "ḿ": "m2", "m̀": "m4",  
    "ê̄": "ê1", "ế": "ê2", "ê̌": "ê3", "ề": "ê4",
}


# 带声调字符与数字表示声调的对应关系
PHONETIC_SYMBOL_DICT = phonetic_symbol.copy()
PHONETIC_SYMBOL_DICT_KEY_LENGTH_NOT_ONE = dict(
    (k, v)
    for k, v in PHONETIC_SYMBOL_DICT.items()
    if len(k) > 1
)


# 匹配带声调字符的正则表达式
RE_PHONETIC_SYMBOL = re.compile(
    r'[{0}]'.format(
        re.escape(''.join(x for x in PHONETIC_SYMBOL_DICT if len(x) == 1))
    )
)


def replace_symbol_to_number(pinyin):
    """把声调替换为数字"""
    def _replace(match):
        symbol = match.group(0)  # 带声调的字符
        # 返回使用数字标识声调的字符
        return PHONETIC_SYMBOL_DICT[symbol]

    # 替换拼音中的带声调字符
    value = RE_PHONETIC_SYMBOL.sub(_replace, pinyin)
    for symbol, to in PHONETIC_SYMBOL_DICT_KEY_LENGTH_NOT_ONE.items():
        value = value.replace(symbol, to)

    return value


def replace_pinyin(match) -> str:
    strs = match.group(0)
    #将在字母上的音调放到字母后面
    ton2 = replace_symbol_to_number(strs)
    tong2_num = re.findall('\d+',ton2,re.S)
    #将声调后置
    if tong2_num:
        result = ton2.replace(tong2_num[0],'') + tong2_num[0]
    else:
        result = strs+'5'
    return result.replace("ɡ", "g")


def replace_jianpin(match) -> str:
    strs = match.group(1)
    letter = match.group(2)
    result = strs + " ".join(letter.upper())
    return result


def replace_year(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    year_s = match.group(1)
    year_e = match.group(2)
    result = f"{verbalize_digit(year_s)}到{verbalize_digit(year_e)}年"
    return result

def replace_man_year(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    unit_s = match.group(1)
    year_s = match.group(2)
    unit_nian_f = match.group(3)
    unit_e = match.group(4)
    year_e = match.group(5)
    unit_nian = match.group(6)
    result = "、"
    if unit_s:
        result += "公元前"
    result += verbalize_digit(year_s)
    if unit_nian_f:
        result += unit_nian_f
    result += "到"
    if unit_e:
        result += "公元前"
    result += verbalize_digit(year_e)
    result += "年、"
    return result

def replace_month(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    month_s = match.group(1)
    month_e = match.group(2)
    result = f"{num2str(month_s)}到{num2str(month_e)}月"
    return result

def replace_day(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    day_s = match.group(1)
    day_e = match.group(5)
    unit = match.group(9)
    result = f"{num2str(day_s)}到{num2str(day_e)}{unit}"
    return result

def replace_date(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    unit = match.group(1)
    year = match.group(2)
    month = match.group(4)
    day = match.group(6)
    result = ""
    if unit:
        result += f"公元前"
    if year:
        result += f"{verbalize_digit(year)}年"
    if month:
        result += f"{verbalize_cardinal(month)}月"
    if day:
        result += f"{verbalize_cardinal(day)}{match.group(10)}"
    return result

def replace_event(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    month = match.group(1)
    day = match.group(2)
    result = ""
    if month:
        result += f"{verbalize_digit(month, alt_one=True)}"
    if day:
        result += f"{verbalize_digit(day, alt_one=True)}"
    return result


def replace_event2(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    month = match.group(1)
    day = match.group(2)
    name = match.group(7)
    result = ""
    if month:
        result += f"{verbalize_digit(month, alt_one=True)}"
    if day:
        result += f"{verbalize_digit(day, alt_one=True)}"
    result += name
    return result


# 用 / 或者 - 分隔的 YY/MM/DD 或者 YY-MM-DD 日期
RE_DATE2 = re.compile(
    r'(\d{4})([- /.])(0[1-9]|1[012])\2(0[1-9]|[12][0-9]|3[01])')


def replace_date2(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    year = match.group(1)
    month = match.group(3)
    day = match.group(4)
    result = ""
    if year:
        result += f"{verbalize_digit(year)}年"
    if month:
        result += f"{verbalize_cardinal(month)}月"
    if day:
        result += f"{verbalize_cardinal(day)}日"
    return result


# RE_PERCENT_RANGE = re.compile(r'([+-]?\d+(?:\.\d+)?%)[-~—～]([+-]?\d+(?:\.\d+)?%)')

# def replace_pencent_range(match) -> str:
#     """
#     Args:
#         match (re.Match)
#     Returns:
#         str
#     """
#     year = match.group(1)
#     month = match.group(3)
#     day = match.group(4)
#     result = ""
#     if year:
#         result += f"{verbalize_digit(year)}年"
#     if month:
#         result += f"{verbalize_cardinal(month)}月"
#     if day:
#         result += f"{verbalize_cardinal(day)}日"
#     return result
