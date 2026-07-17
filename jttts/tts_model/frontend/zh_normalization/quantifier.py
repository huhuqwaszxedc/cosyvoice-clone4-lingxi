#!/usr/bin/python
# -*- encoding: utf-8 -*-

import re
from .num import num2str
from jttts.tts_common.logger import logger
import pdb
# 温度表达式，温度会影响负号的读法
# -3°C 零下三度
RE_TEMPERATURE_RANGE = re.compile(r'(?<!\d)(-?)(\d+(\.\d+)?)\s*([-])\s*(\d+(\.\d+)?)(°C|℃|度|摄氏度)')

# 温度表达式，温度会影响负号的读法
# -3°C 零下三度
RE_TEMPERATURE = re.compile(r'(?<!\d)(-?)(\d+(\.\d+)?)(°C|℃|度|摄氏度)')
#复杂，长度较长的单位写在前面
RE_MEASURE = re.compile(r'(\d+(\.\d+)?)(cm2|cm²|cm3|cm³|min|cm|db|kg|km|m2|m²|m³|m3|ml|mm|ds|m|s|h|l|L)')
#复杂，长度较长的单位写在前面
RE_MEASURE_RANGE = re.compile(r'(\d+(\.\d+)?)[-~—](\d+(\.\d+)?)(cm2|cm²|cm3|cm³|min|cm|db|kg|km|m2|m²|m³|m3|ml|mm|ds|m|s|h|l|L)')
measure_dict = {
    "cm2": "平方厘米",
    "cm²": "平方厘米",
    "cm3": "立方厘米",
    "cm³": "立方厘米",
    "cm": "厘米",
    "db": "分贝",
    "kg": "千克",
    "km": "千米",
    "m2": "平方米",
    "m²": "平方米",
    "m³": "立方米",
    "m3": "立方米",
    "l": "升",
    "L": "升",
    "ml": "毫升",
    "m": "米",
    "ds": "毫秒",
    "mm": "毫米",
    "s": "秒",
    "h": "小时",
    "min": "分"
}


def replace_temperature(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    sign = match.group(1)
    temperature = match.group(2)
    unit = match.group(4)

    sign: str = "零下" if sign else ""
    temperature: str = num2str(temperature)
    unit: str = "摄氏度" if unit == "摄氏度" else "度"
    result = f"{sign}{temperature}{unit}"
    return result

def replace_temperature_range(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    sign = match.group(1)
    start_tem = match.group(2)
    temperature = match.group(5)
    unit = match.group(7)

    start_temperature: str = num2str(start_tem)
    sign: str = "零下" if sign else ""
    temperature: str = num2str(temperature)
    unit: str = "摄氏度" if unit == "摄氏度" else "度"
    result = f"{sign}{start_temperature}{'到'}{temperature}{unit}"
    return result


def replace_measure(match) -> str:
    measure = match.group(1)
    unit = match.group(3)

    measure: str = num2str(measure)
    result = f"{measure}{measure_dict[unit]}"
    return result

def replace_measure_range(match) -> str:
    measure1 = match.group(1)
    measure2 = match.group(3)
    unit = match.group(5)

    measure1: str = num2str(measure1)
    measure2: str = num2str(measure2)
    result = f"{measure1}到{measure2}{measure_dict[unit]}"
    return result


