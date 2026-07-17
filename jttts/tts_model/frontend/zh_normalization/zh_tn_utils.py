#!/usr/bin/python
# -*- encoding: utf-8 -*-
import re
from .num import verbalize_digit, verbalize_cardinal





from datetime import datetime
def datetime_to_chinese(datetime_str):
    if ' ' not in datetime_str:
            datetime_str = datetime_str[:10] + ' ' + datetime_str[10:]
    # 定义时间各部分
    dt = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
    year, month, day = dt.year, dt.month, dt.day
    hour, minute, second = dt.hour, dt.minute, dt.second
    
    year = str(int(year))
    month = str(int(month))
    day = str(int(day))

    hour = str(int(hour))
    minute = str(int(minute))
    second = str(int(second))
    
    if second == '0':
        chinese_datetime = (
            f"{verbalize_digit(year)}年"
            f"{verbalize_cardinal(month)}月"
            f"{verbalize_cardinal(day)}日 "
            f"{verbalize_cardinal(hour)}点"
            f"{verbalize_cardinal(minute)}分"
        )

    else:
        chinese_datetime = (
            f"{verbalize_digit(year)}年"
            f"{verbalize_cardinal(month)}月"
            f"{verbalize_cardinal(day)}日"
            f"{verbalize_cardinal(hour)}点"
            f"{verbalize_cardinal(minute)}分"
            f"{verbalize_cardinal(second)}秒"
        )
    
    return chinese_datetime

def remove_commas_from_numbers(text):
    # 使用正则表达式找到所有形如带有逗号的数字，并替换掉逗号
    return re.sub(r'(\d),(\d)', r'\1\2', text)


def fix_date_format_normalize(text):

    # 定义正则表达式模式以匹配日期时间
    # time_pattern = r'(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})'
    time_pattern = r'(\d{4})-(\d{2})-(\d{2})\s*(\d{2}):(\d{2}):(\d{2})'

    # 替换匹配到的时间字符串
    try:
        new_text = re.sub(time_pattern, lambda m: datetime_to_chinese(m.group()), text)
        return new_text
    except:
        return text


RE_SHORT_Hengxian = re.compile(r'(?<=[\u4e00-\u9fa5])\s*-\s*|(?<!\S)\s*-\s*(?=[\u4e00-\u9fa5])')

def replace_short_hengxian(match) -> str:
    return ','