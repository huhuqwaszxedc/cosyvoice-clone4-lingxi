#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
Rules to verbalize numbers into Chinese characters.
https://zh.wikipedia.org/wiki/中文数字#現代中文
"""
import re
from collections import OrderedDict
from typing import List
import pdb
DIGITS = {str(i): tran for i, tran in enumerate('零一二三四五六七八九')}
UNITS = OrderedDict({
    1: '十',
    2: '百',
    3: '千',
    4: '万',
    8: '亿',
})



# 分数表达式
RE_FRAC = re.compile(r'(-?)(\d+)/(\d+)')


def replace_frac(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    sign = match.group(1)
    nominator = match.group(2)
    denominator = match.group(3)
    sign: str = "负" if sign else ""
    nominator: str = num2str(nominator)
    denominator: str = num2str(denominator)
    result = f"{sign}{denominator}分之{nominator}"
    return result


# 百分数表达式
RE_PERCENTAGE = re.compile(r'(-?)(\d+(\.\d+)?)\s*%')


def replace_percentage(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    sign = match.group(1)
    percent = match.group(2)
    sign: str = "负" if sign else ""
    percent: str = num2str(percent)
    result = f"{sign}百分之{percent}"
    return result

# 百分数范围 比例在50%-90%之间
RE_PERCENT_RANGE = re.compile(r'([+-]?\d+(?:\.\d+)?%)[-~—～]([+-]?\d+(?:\.\d+)?%)')

def replace_percentage_range(match) -> str:

    first, second = match.group(1), match.group(2)
    first = RE_PERCENTAGE.sub(replace_percentage, first)
    second = RE_PERCENTAGE.sub(replace_percentage, second)
    result = f"{first}到{second}"

    return result


# 整数表达式
# 带负号的整数 -10
RE_INTEGER = re.compile(r'(?<![\d])(-)(\d+)')


def replace_negative_num(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    sign = match.group(1)
    number = match.group(2)
    sign: str = "负" if sign else ""
    number: str = num2str(number)
    result = f"{sign}{number}"
    return result

# *****************************以0开头的数字*************************************************************
# 00078
RE_LEADING_ZERO = re.compile(r'(?<![\d.])(0\d+)(?!\d)')
def replace_leading_zero(match):
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    number = match.group(1)
    return verbalize_digit(number, alt_one=True)


# 数字表达式
# 纯小数
RE_DECIMAL_NUM = re.compile(r'(?<!\d)(-?)((\d+)(\.\d+))' r'|(\.(\d+))')



# *****************************正整数 + 关键字 , 按照电报读法读*************************************************************
KEYWORD_SUBFIX_DIGIT ='(会议室|房间|宿舍|教室|病房|诊室|中国|北京|上海|杭州|南京|西安)'
RE_NUM_KEYWORD_DIGIT = re.compile(r"(\d+)" + KEYWORD_SUBFIX_DIGIT)
def replace_num_keyword_digit(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    # pdb.set_trace()
    number = match.group(1)
    quantifiers: str = match.group(2)
    number: str = verbalize_digit(number, alt_one=True)
    result = f"{number}{quantifiers}"
    return result

# *****************************正整数 + 关键字 , 按照幂数读法读*************************************************************

# COM_QUANTIFIERS = '(种|平方|项|封|艘|把|目|套|段|人|所|朵|匹|张|座|回|场|尾|条|个|首|阙|阵|网|炮|顶|丘|棵|只|支|袭|辆|挑|担|颗|壳|窠|曲|墙|群|腔|砣|座|客|贯|扎|捆|刀|令|打|手|罗|坡|山|岭|江|溪|钟|队|单|双|对|出|口|头|脚|板|跳|枝|件|贴|针|线|管|名|位|身|堂|课|本|页|家|户|层|楼|丝|毫|厘|分|钱|两|斤|担|铢|石|钧|锱|忽|(千|毫|微)克|毫|厘|(公)分|分|寸|尺|丈|里|公里|寻|常|铺|程|(千|分|厘|毫|微)米|米|撮|勺|合|升|斗|石|盘|碗|碟|叠|桶|笼|盆|盒|杯|钟|斛|锅|簋|篮|盘|桶|罐|瓶|壶|卮|盏|箩|箱|煲|啖|袋|钵|年|月|日|季|刻|时|周|天|秒|分|小时|旬|纪|岁|世|更|夜|春|夏|秋|冬|代|伏|辈|丸|泡|粒|颗|幢|堆|条|根|支|道|面|片|张|颗|块|元|(亿|千万|百万|万|千|百)|(亿|千万|百万|万|千|百|美|)元|(亿|千万|百万|万|千|百|十|)吨|(亿|千万|百万|万|千|百|)块|角|毛|分)'

KEYWORD_SUBFIX_CARDINAL = '(种|公顷|平方|项|封|艘|把|目|套|段|人|所|朵|匹|张|座|回|场|条|个|首|阵|网|炮|顶|棵|只|支|辆|挑|担|颗|壳|曲|墙|群|腔|砣|座|贯|扎|捆|刀|令|打|手|队|单|对|口|头|板|跳|枝|件|贴|针|线|管|名|位|堂|课|本|页|家|户|层|楼|毫|厘|分|钱|两|斤|担|铢|石|(千|毫|微)克|毫|厘|(公)分|分|寸|尺|丈|里|公里|铺|程|(千|分|厘|毫|微)米|米|撮|勺|合|升|斗|盘|碗|碟|叠|桶|笼|盆|盒|杯|钟|锅|簋|篮|罐|瓶|壶|卮|盏|箩|箱|煲|啖|袋|钵|年|月|日|季|刻|时|周|天|秒|分|小时|旬|纪|岁|世|夜|代|伏|辈|丸|泡|粒|颗|幢|堆|条|根|支|道|面|片|张|颗|块|元|(亿|千万|百万|万|千|百)|(亿|千万|百万|万|千|百|美|)元|(亿|千万|百万|万|千|百|十|)吨|(亿|千万|百万|万|千|百|)块|角|毛|分|点)'

# 关键字 + 量词
RE_NUM_KEYWORD_CARDINAL = re.compile(r"(\d+)([多余几\+])?" + KEYWORD_SUBFIX_CARDINAL)
sevice_hotline = ['10086', '12306', '10010', '10085', '1008611', '10080', '13800138000', '12580',
                '110', '119', '122', '120', '12315', '12333', '12366', '12369', '12301', '12305',
                '10000', '10050', '95588', '95599', '95566', '95533', '95580', '95555', '95519',
                '95500', '95511', '95589', '95518', 
                '600941','600050','601728','9981', # 中国移动、联通、电信的股票代码 
                ]
def replace_num_keyword_cardinal(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    # pdb.set_trace()
    number = match.group(1)
    match_2 = match.group(2)
    if match_2 == "+":
        match_2 = "多"
    match_2: str = match_2 if match_2 else ""
    quantifiers: str = match.group(3)
    if len(match_2) == 0 and number in sevice_hotline and quantifiers in ['人', '客']:
        number: str = verbalize_digit(number, alt_one=True)
    elif len(match_2) == 0 and number in ['2'] and quantifiers in ['点','个', '人', ]:
        number = '两'
    elif len(match_2) == 0 and re.search(r'\b\d{4}\b', number) and quantifiers in ['世', ]:
        number: str = verbalize_digit(number, alt_one=True)
    else:
        number: str = num2str(number)
    result = f"{number}{match_2}{quantifiers}"
    return result

# *****************************关键字 + 正整数 , 按照电报读法读*************************************************************
KEYWORD_PREFIXES_DIGIT = r'((?:客服|客服号码|电话|号码|热线|股票代码|证券号|证券代码|大楼|会议室|大厦|代码|会议)(?:为|是)?)'
# |G|D|K|k|CA|CZ|MU|HU|MF|ZH|SC|FM|BK|HO|SQ|TG|JL
RE_KEYWORD_NUM_DIGIT = re.compile(KEYWORD_PREFIXES_DIGIT + r'(\d+)(?![\d.])')


def replace_keyword_num_digit(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    # pdb.set_trace()
    keyword = match.group(1)
    number = match.group(2)
    number: str = verbalize_digit(number, alt_one=True)
    result = f"{keyword}{number}"
    return result

# *****************************关键字 + 正整数 , 按照幂数读法读*************************************************************
KEYWORD_PREFIXES_CARDINAL = r'((?:金额|收款|方式)(?:为|是)?)'
RE_KEYWORD_NUM_CARDINAL = re.compile(KEYWORD_PREFIXES_CARDINAL + r'(\d+)(?![\d.])')
def replace_keyword_num_cardinal(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    # pdb.set_trace()
    keyword = match.group(1)
    number = match.group(2)
    number: str = num2str(number)
    result = f"{keyword}{number}"
    return result


# ******************************************************************************************
RE_NUMBER = re.compile(r'(?<!\d)(-?)((\d+)(\.\d+)?)' r'|(\.(\d+))')
#要考虑中英混的场景，数字前后不可以是字母或者空格
# RE_NUMBER = re.compile(r'(-?)((\d+)(\.\d+)?)' r'|(\.(\d+))')

def replace_number(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    # pdb.set_trace()
    sign = match.group(1)
    number = match.group(2)
    pure_decimal = match.group(5)
    if pure_decimal: # 纯小数
        result = num2str(pure_decimal)
    elif len(sign)==0 and number in sevice_hotline:
        result: str = verbalize_digit(number, alt_one=True)
    else:
        if sign:
            sign: str = "负" 
            number: str = num2str(number)
            result = f"{sign}{number}"
        else:
            sign = ''
            # number: str = verbalize_digit(number, alt_one=False)
            number: str = num2str(number)
            result = f"{sign}{number}"

    return result



def replace_number_default(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    # pdb.set_trace()
    sign = match.group(1)
    number = match.group(2)
    pure_decimal = match.group(5)
    if pure_decimal: # 纯小数
        result = num2str(pure_decimal)
    elif len(sign)==0 and number in sevice_hotline:
        result: str = verbalize_digit(number, alt_one=True)
    else:
        if sign:
            sign: str = "负" 
            number: str = num2str(number)
            if len(number) >=2 and number[:2] in ['二百','二千', '二万', '二亿']:
                number =  '两' + number[1:]
            result = f"{sign}{number}"
        else:
            sign = ''
            # number: str = verbalize_digit(number, alt_one=False)
            number: str = num2str(number)
            if len(number) >=2 and number[:2] in ['二百','二千', '二万', '二亿']:
                number =  '两' + number[1:]
            result = f"{sign}{number}"

    return result


# 范围表达式
# match.group(1) and match.group(8) are copy from RE_NUMBER
#数学表达式会被误匹配,例如："加多了8-3=5"
# RE_RANGE = re.compile(
#     r'((-?)((\d+)(\.\d+)?)|(\.(\d+)))[-~—]((-?)((\d+)(\.\d+)?)|(\.(\d+)))\s*(?![a-zA-Z=])')

# RE_RANGE = re.compile(
#     r'((-?)((\d+)(\.\d+)?)|(\.(\d+)))[-~—]((-?)((\d+)(\.\d+)?)|(\.(\d+)))\s*(?![\d=])')

RE_RANGE = re.compile(
    r'((-?)((\d+)(\.\d+)?)|(\.(\d+)))\s*[-~—]\s*((-?)((\d+)(\.\d+)?)|(\.(\d+)))\s*(?![\d=])'
)

def replace_range(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    first, second = match.group(1), match.group(8)
    first = RE_NUMBER.sub(replace_number, first)
    second = RE_NUMBER.sub(replace_number, second)
    result = f"{first}到{second}"
    return result

# ******************************************************************************************
operators = r'[+\-*/]'  # 匹配 +, -, *, / （注意 - 需要转义）
# 运算符到中文的映射
op_to_chinese = {
    '+': '加',
    '-': '减',
    '*': '乘',
    '/': '除'
    # 可以根据需要添加，如 '=': '等于'
}
variables_with_boundary = r'\b(?:[a-z]|\d+)\b'
RE_PURE_MATH = re.compile(rf'({variables_with_boundary})({operators})({variables_with_boundary})(\s*=\s*)?({variables_with_boundary})?')

def replace_pure_math(match):
    """
    替换函数：将匹配到的数学表达式转换为中文读法。
    """
    left, op, right, equals, result = match.groups()
    
    # 转换左边部分
    if left.isalpha(): # 如果是字母
        ch_left = left # 使用字母映射
    else: # 如果是数字
        ch_left = num2str(left)      # 使用数字转换函数
    
    # 转换运算符
    ch_op = op_to_chinese.get(op, op)
    
    # 转换右边部分
    if right.isalpha():
        ch_right = right
    else:
        ch_right = num2str(right)
    
    # 构建基础表达式
    expr = f"{ch_left}{ch_op}{ch_right}"
    
    # 如果有等号和结果
    if equals is not None and result is not None: # 注意：equals 可能包含空格，result 是变量/数字
        if result.isalpha():
            ch_result = result
        else:
            ch_result = num2str(result)
        expr = f"{expr}等于{ch_result}"
    
    return expr


# ******************************************************************************************
RE_MATH = re.compile(
    r'([+*/\-]?\s*[\(\[\{])?(\s*-\s*)?(\s*[+*/\-\d]+\s*)(\s*[\)\]\}]\s*)?([=\s]*)')

def is_math_element(sentence, idx):
    if sentence.split()[idx] in ['a', 'b', 'c', 'd', 'x', 'y', 'z'] :
        return True
    else:
        return False

#该处需要优化
def replace_math(sentence) -> str:
    """
    Args:
        str
    Returns:
        str
    """
    end_index = 0
    result_str = ""
    for re_match in RE_MATH.finditer(sentence):
        # pdb.set_trace()
        #先判别是否为数学表达式
        #首先获取匹配到的文本
        match_txt = re_match.group()
        match_span = re_match.span()

        if is_math_element(sentence[0:match_span[0]], -1) is False:
            return sentence[0:match_span[0]] + sentence[match_span[1]:]
        
        if is_math_element(sentence[match_span[1]:], 0) is False:
            return sentence[0:match_span[0]] + sentence[match_span[1]:]

        #不是数学表达式，则进行返回
        if len(re.sub(r"[\d\s\-/]","",match_txt)) == 0:
            if len(re.sub(r"[\d]","",match_txt)) == 0:
                continue
            #看看文本中是否有与数学表达相关的描述
            elif len(re.sub(r"[^加减乘除等于大小=><]","",sentence)) == 0:
                continue
        #是数学表达是，进行转换
        result_str += sentence[end_index:match_span[0]]
        groups_info = re_match.groups()
        if groups_info[0]:
            #进行运算符号替换
            info = groups_info[0].replace("+","加")
            info = info.replace("-","减")
            info = info.replace("*","乘")
            info = info.replace("/","除")
            info = info.replace("(","左括号")
            info = info.replace("[","左中括号")
            info = info.replace("{","左花括号")
            result_str += info
        if groups_info[1]:
            #进行运算符号替换
            result_str += "负"
        if groups_info[2]:
            #进行运算符号替换
            info = groups_info[2].replace("+","加")
            info = info.replace("-","减")
            info = info.replace("*","乘")
            info = info.replace("/","除")
            result_str += info
        if groups_info[3]:
            #进行运算符号替换
            info = groups_info[3].replace(")","右括号，")
            info = info.replace("]","右中括号，")
            info = info.replace("}","右花括号，")
            result_str += info
        if groups_info[4]:
            info = groups_info[4].replace("=","等于")
            result_str += info
        span = re_match.span()
        result_str = result_str.replace(" ","")
        #对句首的"减号"替换为"负"
        result_str = result_str[0].replace("减","负") + result_str[1:]
        end_index = match_span[1]
    result_str += sentence[end_index:]
    return result_str


def _get_value(value_string: str, use_zero: bool=True) -> List[str]:
    stripped = value_string.lstrip('0')
    if len(stripped) == 0:
        return []
    elif len(stripped) == 1:
        if use_zero and len(stripped) < len(value_string):
            return [DIGITS['0'], DIGITS[stripped]]
        else:
            return [DIGITS[stripped]]
    else:
        largest_unit = next(
            power for power in reversed(UNITS.keys()) if power < len(stripped))
        first_part = value_string[:-largest_unit]
        second_part = value_string[-largest_unit:]
        return _get_value(first_part) + [UNITS[largest_unit]] + _get_value(
            second_part)


def verbalize_cardinal(value_string: str) -> str:
    if not value_string:
        return ''
    # 000 -> '零' , 0 -> '零'
    value_string = value_string.lstrip('0')
    if len(value_string) == 0:
        return DIGITS['0']

    result_symbols = _get_value(value_string)
    # verbalized number starting with '一十*' is abbreviated as `十*`
    if len(result_symbols) >= 2 and result_symbols[0] == DIGITS['1'] and result_symbols[1] == UNITS[1]:
        result_symbols = result_symbols[1:]
    
    return ''.join(result_symbols)


def verbalize_digit(value_string: str, alt_one=False) -> str:
    result_symbols = [DIGITS[digit] for digit in value_string]
    result = ''.join(result_symbols)
    if alt_one:
        result = result.replace("一", "幺")
    return result


def num2str(value_string: str) -> str:
    # pdb.set_trace()
    integer_decimal = value_string.split('.')
    if len(integer_decimal) == 1:
        integer = integer_decimal[0]
        decimal = ''
    elif len(integer_decimal) == 2:
        integer, decimal = integer_decimal
    else:
        # raise ValueError(f"The value string: '${value_string}' has more than one point in it.")
        integer = '1'
        decimal = '1'

    result = verbalize_cardinal(integer)

    # decimal = decimal.rstrip('0') # 小数部分，移除最右边的0，不读出来
    # 像百度的文心大模型5.0，为了解决此类问题，将上面代码注释掉
    if decimal:
        # '.22' is verbalized as '零点二二'
        # '3.20' is verbalized as '三点二
        result = result if result else "零"
        result += '点' + verbalize_digit(decimal)
    return result

# **************************************************************************************

# 2508会议室, 306房间
chars_after_digit = ['会议室', '房间','宿舍', '教室', '病房', '诊室',]
RE_DIGIT_CHARS = re.compile(r'(\d+)\s*({})'.format('|'.join(map(re.escape, chars_after_digit))))

# 创新大楼2506， 创新大厦2369
chars_before_digit = ['大楼', '会议室', '大厦']
RE_CHARS_DIGIT = re.compile(r'({})\s*(\d+)'.format('|'.join(map(re.escape, chars_before_digit))))

def replace_digit_char(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    num_part = match.group(1)
    char_part = match.group(2)
    chinese_num = verbalize_digit(num_part, alt_one=True)
    return f"{chinese_num}{char_part}"

def replace_char_digit(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    char_part = match.group(1)
    num_part = match.group(2)
    
    chinese_num = verbalize_digit(num_part, alt_one=True)
    return f"{char_part}{chinese_num}"

# ***********************************************************************************
# CA1589  京ABD987  航班号。车牌号等等
RE_CAPITAL_EN_DIGIT = re.compile(r'([A-Z]+)(\d+)')

enname_to_cnname_dict ={
    'G': '高',
    'K': '快',
    'k': '快',
    'D': '动'
}
def replace_capital_english_digit(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    char_part = match.group(1)
    num_part = match.group(2)
    if char_part in enname_to_cnname_dict.keys():
        char_part = enname_to_cnname_dict[char_part]
    # pdb.set_trace()
    if char_part in ['F'] and num_part in ['1']:
        chinese_num = verbalize_digit(num_part, alt_one=False)
    else:
        chinese_num = verbalize_digit(num_part, alt_one=True)

    char_list = []
    for sub_char in char_part:# 将英文字母之间增加空格
        char_list.append(sub_char)
    char_part_str = ' '.join(char_list)
    return f"{char_part_str}{chinese_num}"


# ***********************************************************************************
keywords = ["股票代码", "股票", "交易所", "证券"]
RE_GUPIAO_DIGIT = re.compile(r'(.*?(' + '|'.join(keywords) + r').*?)(\d+)') # "上海证券交易所，股票代码600941"

def replace_gupiao_digit(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    char_part = match.group(1)
    num_part = match.group(3)
    
    chinese_num = verbalize_digit(num_part, alt_one=True)

    char_list = []
    for sub_char in char_part:
        char_list.append(sub_char)
    char_part_str = ' '.join(char_list)
    return f"{char_part_str}{chinese_num}"


# ***********************************************************************************
RE_ZEROSTART_DIGIT = re.compile(r'^([^0-9]*)(0\d+)$') # 以0作为开头的数字，需要进行分离

def replace_zerostart_digit(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    char_part = match.group(1)
    num_part = match.group(3)
    
    chinese_num = verbalize_digit(num_part, alt_one=True)

    char_list = []
    for sub_char in char_part:
        char_list.append(sub_char)
    char_part_str = ' '.join(char_list)
    return f"{char_part_str}{chinese_num}"


# ************************************************************************************
# 适配“会议号为660 235 325”、“云视讯是660-235-325”、“云视讯660-235-325”、“云视讯660235325”
meeting_prefixes = ["会议号", "会议", "腾讯会议", "云视讯","云视讯号"]
# 将前缀列表转换为正则表达式模式（使用|分隔）
prefix_pattern = '|'.join(re.escape(p) for p in meeting_prefixes)
pattern = rf'({prefix_pattern})(?:是|为)?(\d{{3}}[- ]\d{{3}}[- ]\d{{3}})'
RE_MEETING_DIGIT = re.compile(rf'(?:{prefix_pattern})(?:是|为)?(\d{{3}}[- ]?\d{{3}}[- ]\d{{3}})') 
RE_MEETING_DIGIT = re.compile(rf'(?:{prefix_pattern})(?:是|为)?(\d{{3}}[- ]?\d{{3}}[- ]?\d{{3}})') 

def replace_meeting_digit(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    full_match = match.group(0)
    num_part1 = match.group(1)
    num_part2 = num_part1.replace(' ', '').replace('-', '')
    chinese_num = verbalize_digit(num_part2, alt_one=True)
    return full_match.replace(num_part1, chinese_num)



# ************************************************************************************
# 正则表达式：匹配"参考"或"详见"+[ref_doc_数字]，捕获关键字和数字
RE_KEYWORD_REFDOC = re.compile(r'(参考|详见)\[ref_doc_(\d+)\]')
def replace_keyword_refdoc(match):
    # pdb.set_trace()
    num = match.group(2)  # 获取数字部分
    chinese_num = verbalize_cardinal(str(num))
    return f"{match.group(1)}文档{chinese_num}"  # 末尾保留空格


# 正则表达式：匹配[ref_doc_数字] + '提到'或者'指出'，捕获关键字和数字
RE_REFDOC_KEYWORD = re.compile(r'\[ref_doc_(\d+)\](提到|指出|提及)')
def replace_refdoc_keyword(match):
    # pdb.set_trace()
    num = match.group(1)  # 获取数字部分
    chinese_num = verbalize_cardinal(str(num))
    return f"文档{chinese_num}{match.group(2)}"  # 末尾保留空格

# 正则表达式：剩余[ref_doc_数字],全部移除
RE_REFDOC_BLANK = re.compile(r'\[ref_doc_(\d+)\]')
def replace_refdoc_blank(match):
    return f""  # 末尾保留空格


# ************************************************************************************

# 正则：匹配 (1)、（2）、1)、2）、[3] 等编号格式
# 使用非捕获组和多个选项，捕获数字部分
RE_NUMBERING = re.compile(
    r'\b(?:'
    r'\(?(\d+)[\)）]?'          # 匹配 1) 或 1） 或 (1)
    r'|'
    r'[（\[](\d+)[）\]]'        # 匹配 （1） 或 [1]
    r')\b'
)

def replace_numbering(match) -> str:
    # 从多个捕获组中提取数字（总有一个是非 None）
    groups = match.groups()
    num_str = next(g for g in groups if g is not None)
    chinese = num2str(num_str)
    return f"{chinese},"



# ***********************匹配全国的车牌号*************************************************************

# 定义省份简称字符集
PROVINCE_CHARS = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领港澳"

# 编译正则表达式对象（用于sub）
RE_CAR_PLATE_NUMBER = re.compile(rf'([{PROVINCE_CHARS}])[A-Z][0-9A-Z]+')


def replace_car_plate_number(match) -> str:
    plate = match.group(0)  # 完整匹配的车牌字符串，如 "京A12345"
    if not plate:
        return plate

    # 分解：第一个字符是省份，第二个是城市字母，后面是主体
    province = plate[0]      # 如 "京"
    city_code = plate[1]     # 如 "A"
    body = plate[2:]         # 如 "12345"

    normalized_parts = []
    normalized_parts.append(province)
    normalized_parts.append(city_code)
    for sub_str in body:
        if sub_str >='A' and sub_str <= 'Z':
            normalized_parts.append(sub_str)
        else:
            normalized_parts.append(verbalize_digit(sub_str, alt_one=True))

    out_str = ' ' + ' '.join(normalized_parts) + ' '

    return out_str


# ***********************匹配固定的带数字的关键字，并将其中的数字按照电报读法去读*************************************************************
fixText_num2digit_list = [
    '中国新声代2026',
    '新说唱2024',
    '新说唱2025',
    '歌手2024',
    '歌手2025',
    '有歌2024',
    '有歌2026',
    '声鸣远扬2025',
    '中国好声音2023',
    '萌探2024',
    '爱情保卫战2025',
    '爱情保卫战2026',
    '男生女生向前冲2025',
    '男生女生向前冲2026',
    '非你莫属2025',
    '非你莫属2026',
    '创造营2021',
    '群英会2025',
    '群英会2026',
    '跨时代战书2026',
    '创业中国人2026',
    '跑男来了2025',
    '你好，星期六2025',
    '你好，星期六2026',
    '披荆斩棘2025',
    '乘风2025',
    '我家那闺女2025',
    '全员加速中2025',
    '非诚勿扰2025',
    '非诚勿扰2026',
    '开门大吉2026',
    '开门大吉2026',
    '星光大道2025',
    '星光大道2026',
    '越战越勇2026',
    '华西论健2026',
    '非常话题2026',
    '笑逐言开2026',
    '星推荐2026',
    '幸福账单2025',
    '大明王朝1566',
    '特赦1959',
    '致1999年的自己',
    '唐探1900',
    '749局',
    '胜利1945',
    '绝密1950',
    '1945黎明之战',
    '银翼杀手2049',
    '2001太空漫游',
    '请回答1988',
    '红楼梦87版',
    '末路1997',
    '刀锋1937',
    '暗杀1932',
    '1950他们正年轻',
    '我的1919',
    '警察故事2013',    
]


# 1. 构建正则表达式模式
# 使用 re.escape 确保列表中的字符串包含特殊正则字符时也能被正确匹配
escaped_items = [re.escape(item) for item in fixText_num2digit_list]
# 使用非捕获组 (?:...) 和 | 操作符连接所有选项
# 括号是必要的，用于捕获整个匹配项
pattern = r'(' + '|'.join(escaped_items) + r')'
RE_FIX_TEXT_NUMBER2DIGIT = re.compile(pattern)


def replace_num2digits_in_fix_text(match):
    """
    re.sub 的替换函数，用于将匹配到的数字转换为电报读法。
    """
    matched_text = match.group(0)
    # 在匹配到的文本中，将所有数字序列 (\d+) 替换为电报读法
    # 使用 lambda 表达式，将每个找到的数字传递给 verbalize_digit
    return re.sub(r'\d+', lambda m: verbalize_digit(m.group(0), alt_one=False), matched_text)


