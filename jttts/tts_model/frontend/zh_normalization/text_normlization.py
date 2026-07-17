#!/usr/bin/python
# -*- encoding: utf-8 -*-

import re
from typing import List

from .char_convert import tranditional_to_simplified, trans_to_identify
from .chronology import RE_DATE
from .chronology import RE_YEAR_RANGE, RE_TRANS_PINYIN, RE_TRANS_JIANPIN
from .chronology import RE_YEAR_MAN_RANGE
from .chronology import RE_MONTH_RANGE
from .chronology import RE_DAY_RANGE
from .chronology import RE_RATIO, RATIO_KEYWORD
from .chronology import RE_DATE2
from .chronology import RE_EVENT, RE_EVENT2
from .chronology import RE_TIME
from .chronology import RE_TIME_RANGE
from .chronology import replace_date
from .chronology import replace_ratio
from .chronology import replace_year, replace_pinyin, replace_jianpin
from .chronology import replace_man_year
from .chronology import replace_month
from .chronology import replace_day
from .chronology import replace_date2
from .chronology import replace_event, replace_event2
from .chronology import replace_time
from .constants import F2H_ASCII_LETTERS
from .constants import F2H_DIGITS
from .constants import F2H_SPACE
from .num import RE_DECIMAL_NUM
from .num import RE_LEADING_ZERO
from .num import RE_FRAC
from .num import RE_INTEGER
from .num import RE_NUMBER
from .num import RE_PERCENTAGE, RE_PERCENT_RANGE
from .num import RE_NUM_KEYWORD_DIGIT, RE_NUM_KEYWORD_CARDINAL, replace_num_keyword_cardinal, replace_num_keyword_digit
from .num import RE_RANGE, RE_MATH, RE_PURE_MATH, RE_MEETING_DIGIT, replace_meeting_digit
from .num import replace_leading_zero
from .num import replace_frac
from .num import replace_negative_num
from .num import replace_number, replace_number_default
from .num import replace_percentage, replace_percentage_range
from .num import replace_range, replace_math, replace_pure_math
from .num import RE_DIGIT_CHARS, RE_CHARS_DIGIT, replace_digit_char, replace_char_digit
from .num import RE_KEYWORD_NUM_DIGIT, replace_keyword_num_digit, RE_KEYWORD_NUM_CARDINAL, replace_keyword_num_cardinal
from .num import RE_KEYWORD_REFDOC, replace_keyword_refdoc, RE_REFDOC_KEYWORD, replace_refdoc_keyword, RE_REFDOC_BLANK, replace_refdoc_blank
from .num import RE_NUMBERING, replace_numbering
from .symbol import remove_paired_bracket_content, replace_hyphen_after_capital_word, handle_incomplete_brackets
from .phonecode import RE_MOBILE_PHONE
from .phonecode import RE_NATIONAL_UNIFORM_NUMBER
from .phonecode import RE_TELEPHONE
from .phonecode import RE_TELEPHONE_SPECIAL
from .phonecode import replace_mobile
from .phonecode import replace_phone
from .quantifier import RE_TEMPERATURE
from .quantifier import RE_TEMPERATURE_RANGE
from .quantifier import RE_MEASURE, RE_MEASURE_RANGE
from .quantifier import replace_temperature
from .quantifier import replace_temperature_range
from .quantifier import replace_measure, replace_measure_range
from .zh_tn_utils import fix_date_format_normalize, RE_SHORT_Hengxian, replace_short_hengxian, remove_commas_from_numbers
from .num import RE_CAPITAL_EN_DIGIT, replace_capital_english_digit, RE_GUPIAO_DIGIT, replace_gupiao_digit
from .num import RE_CAR_PLATE_NUMBER, replace_car_plate_number
from .num import RE_FIX_TEXT_NUMBER2DIGIT, replace_num2digits_in_fix_text


import os
import jttts.tts_model.frontend.config_frontend as cfg
from jttts.tts_common.logger import logger
from bs4 import BeautifulSoup
import pdb
from jttts.tts_common.tts_fileencoder import check_encrypt_file_valid, text_file_decrypt


class TextNormalizer():
    def __init__(self):
        self.SENTENCE_SPLITOR = re.compile(r'([，；。？！,;?!][”’]?)')
        # self.SELECT_REGEX = re.compile(r'(^|\s|[^a-zA-Z])([A-Za-z])([.\s]+)?([^a-zA-Z])')
        self.SELECT_REGEX = re.compile(r'(^|[^a-zA-Z\s])\s*([A-Ga-g])([.\s]+)')
        self.SENTENCE_SPLITOR_level_1 = re.compile(r'(.{8,}?[^a-zA-Z\d])([,，。？！?!…；;][”’]?)(?![\sa-zA-Z\d])')
        # self.SENTENCE_SPLITOR_level_1 = re.compile(r'([,，。？！?!…；;][”’]?)')
        self.SENTENCE_SPLITOR_level_2 = re.compile(r'(.{25,}?)([)）\]\}>》:：][”’]?)(?!\s*[\d=])')
        # self.SENTENCE_SPLITOR_NUM = re.compile(r'(^[A-Z])[.]')
        self.SENTENCE_SPLITOR_NUM = re.compile(r'((^[A-Z])|([^a-zA-Z][a-zA-Z]))[.]')
        self.SENTENCE_SPLITOR_dot = re.compile(r"((?<=^)|\s)([A-Za-z.]{0,}[A-Za-z][.])")
        self.ABBR_PATH  = os.path.join(cfg.base_path, "TextNormalize", "en_abbr.txt")
        self.ABBR_PATH, ABBR_is_encrypt = check_encrypt_file_valid(self.ABBR_PATH, loginfo='ABBR_PATH')

        if not os.path.exists(self.ABBR_PATH):
            logger.info("缩写词典路径不正确:{}".format(self.ABBR_PATH))
            exit()
        self.ABBR_DICT = {}
        
        if ABBR_is_encrypt:
            txt_list = text_file_decrypt(self.ABBR_PATH)
        else:
            with open(self.ABBR_PATH,"r",encoding="utf-8") as readf:
                txt_list = readf.readlines()
        for txt in txt_list:
            k,v = txt.strip().split("\t")
            if k not in self.ABBR_DICT:
                self.ABBR_DICT[k] = v

        del txt_list

    def _replace_dot(self, txt) -> str:
        """
        Args:
            match (re.Match)
        Returns:
            str
        """
        result = txt
        for re_dot in self.SENTENCE_SPLITOR_dot.finditer(txt):
            if re_dot:
                word = re_dot.group()
                if word not in self.ABBR_DICT:
                    result = result.replace(word,word[:-1]+"\n")
                else:
                    result = result.replace(word,self.ABBR_DICT[word])
        return result

        
    def _split(self, text: str, lang="mix") -> List[str]:
        """Split long text into sentences with sentence-splitting punctuations.
        Args:
            text (str): The input text.
        Returns:
            List[str]: Sentences.
        """
        text = re.sub(r"[.。?？!！，,;；]*<([/\s]*)br([/\s]*)>", "。\n", text)
        text = BeautifulSoup(text, "html.parser").get_text()
        # Only for pure Chinese here
        if lang == "zh":
            text = text.replace(" ", "")
            # text = re.sub(r'[——《》【】<=>{}()（）#&@“”^_|…\\]', '', text)
            text = re.sub(r'[=#&@^_|…\\]', '', text)
        elif lang == "en":
            # text = self.SENTENCE_SPLITOR.sub(r' \1\n', text)
            # text = self.SENTENCE_SPLITOR_level_1.sub(r' \1\n', text)
            text = self.SENTENCE_SPLITOR_level_1.sub(r'\1 \2\n', text)
        else:
            #判断'.'是否为句号
            #句首是A.Mrs. wu,
            # text = self.SENTENCE_SPLITOR_NUM.sub(r'\1 ,', text)
            text = self.SELECT_REGEX.sub(r'\1,\2,', text)
            if len(text) > 0 and text[0] == ",":
                text = text[1:]
            text = self._replace_dot(text)
            if "…" in text:
                text = re.sub(r'…+',"…", text)
            
            # text = self.SENTENCE_SPLITOR.sub(r'\1\n', text)
            # text = self.SENTENCE_SPLITOR_level_1.sub(r'\1\n', text)
            text = self.SENTENCE_SPLITOR_level_1.sub(r'\1\2\n', text)
        text = text.strip()
        sentences_1 = [sentence.strip() for sentence in re.split(r'\n+', text)]
        sentences = []
        for text in sentences_1:
            if len(text) > 25:
                # Only for pure Chinese here
                if lang == "zh":
                    text = text.replace(" ", "")
                    # text = re.sub(r'[——《》【】<=>{}()（）#&@“”^_|…\\]', '', text)
                    text = re.sub(r'[=#&@^_|…\\ ]', '', text)
                elif lang == "en":
                    # text = self.SENTENCE_SPLITOR.sub(r' \1\n', text)
                    text = self.SENTENCE_SPLITOR_level_2.sub(r'\1 \2\n', text)
                else:
                    # text = self.SENTENCE_SPLITOR.sub(r'\1\n', text)
                    text = self.SENTENCE_SPLITOR_level_2.sub(r'\1\2\n', text)
                    if len(text) > 40:
                        text = re.sub(r"(.{25,})([^a-zA-Z\d.]) ([^a-zA-Z\d.])",r'\1\2\n\3', text)
                sentences += [sen.strip() for sen in re.split(r'\n+', text)]
            else:
                if lang == "en":
                    text = self.SENTENCE_SPLITOR_level_2.sub(r'\1 \2\n', text)
                sentences.append(text)
        return sentences



    def chinese_punct_to_english(self, text):
        # 定义中文标点到英文标点的映射
        punctuation_mapping = {
            '，': ',',
            '。': '.',
            '！': '!',
            '？': '?',
            '；': ';',
            '：': ':',
            '“': '"',
            '”': '"',
            '‘': '\'',
            '’': '\'',
            '（': '(',
            '）': ')',
            '【': '[',
            '】': ']',
            '《': '<',
            '》': '>',
            '……': '...',
            '—': '-'
        }
        # 遍历映射字典，将文本中的中文标点替换为英文标点
        for chinese_punct, english_punct in punctuation_mapping.items():
            text = text.replace(chinese_punct, english_punct)
        return text


    def _split_as_punc(self, text: str, lang="mix") -> List[str]:
        # text = self.chinese_punct_to_english(text)

        sentences = []
        punc_split_en = '!?'
        punc_split_zh = '，。！？'
        punc_split = punc_split_en + punc_split_zh
        sub_str = ''
        for sub_word in text:
            if sub_word in punc_split and len(sub_str) > 30:
                sub_str += sub_word
                sentences.append(sub_str)
                sub_str = ''
            else:
                sub_str += sub_word

        if len(sub_str) > 0:
            sentences.append(sub_str)
        return sentences

    

    def _post_replace(self, sentence: str) -> str:

        sentence = sentence.replace('（', '')
        sentence = sentence.replace('）', '')
        sentence = sentence.replace('(', '')
        sentence = sentence.replace(')', '')
        sentence = sentence.replace('+', '加')
        # 例如：beyond-大地
        # sentence = sentence.replace('-', '减')
        sentence = sentence.replace('-', ' ')
        # sentence = re.sub(r"([\d])[Xx]([\d])",r"\1乘\2",sentence)
        # sentence = sentence.replace('x', '乘')
        # sentence = sentence.replace('×', '乘') # 也就是2×2=4了
        sentence = sentence.replace('=', '等于')
        # sentence = sentence.replace('/', '每')
        sentence = sentence.replace('÷', '除')
        # sentence = sentence.replace('~', '至')
        # sentence = sentence.replace('～', '至')
        sentence = sentence.replace('Ⅰ', '一')
        sentence = sentence.replace('Ⅱ', '二')
        sentence = sentence.replace('Ⅲ', '三')
        sentence = sentence.replace('Ⅳ', '四')
        sentence = sentence.replace('Ⅴ', '五')
        sentence = sentence.replace('Ⅵ', '六')
        sentence = sentence.replace('Ⅶ', '七')
        sentence = sentence.replace('Ⅷ', '八')
        sentence = sentence.replace('Ⅸ', '九')
        sentence = sentence.replace('Ⅹ', '十')
        sentence = sentence.replace('①', '一')
        sentence = sentence.replace('②', '二')
        sentence = sentence.replace('③', '三')
        sentence = sentence.replace('④', '四')
        sentence = sentence.replace('⑤', '五')
        sentence = sentence.replace('⑥', '六')
        sentence = sentence.replace('⑦', '七')
        sentence = sentence.replace('⑧', '八')
        sentence = sentence.replace('⑨', '九')
        sentence = sentence.replace('⑩', '十')
        #引用角标，去除掉
        sentence = sentence.replace('⑴', '')
        sentence = sentence.replace('⑵', '')
        sentence = sentence.replace('⑶', '')
        sentence = sentence.replace('⑷', '')
        sentence = sentence.replace('⑸', '')
        sentence = sentence.replace('⑹', '')
        sentence = sentence.replace('⑺', '')
        sentence = sentence.replace('⑻', '')
        sentence = sentence.replace('⑼', '')
        sentence = sentence.replace('⑽', '')
        sentence = sentence.replace('⑾', '')
        sentence = sentence.replace('⑿', '')
        sentence = sentence.replace('⒀', '')
        sentence = sentence.replace('⒁', '')
        sentence = sentence.replace('⒂', '')
        sentence = sentence.replace('⒃', '')
        sentence = sentence.replace('⒄', '')
        sentence = sentence.replace('⒅', '')
        sentence = sentence.replace('⒆', '')
        sentence = sentence.replace('⒇', '')
        sentence = sentence.replace('α', '阿尔法')
        sentence = sentence.replace('β', '贝塔')
        sentence = sentence.replace('γ', '伽玛').replace('Γ', '伽玛')
        sentence = sentence.replace('δ', '德尔塔').replace('Δ', '德尔塔')
        sentence = sentence.replace('ε', '艾普西龙')
        sentence = sentence.replace('ζ', '捷塔')
        sentence = sentence.replace('η', '依塔')
        sentence = sentence.replace('θ', '西塔').replace('Θ', '西塔')
        sentence = sentence.replace('ι', '艾欧塔')
        sentence = sentence.replace('κ', '喀帕')
        sentence = sentence.replace('λ', '拉姆达').replace('Λ', '拉姆达')
        sentence = sentence.replace('μ', '缪')
        sentence = sentence.replace('ν', '拗')
        sentence = sentence.replace('ξ', '克西').replace('Ξ', '克西')
        sentence = sentence.replace('ο', '欧米克伦')
        sentence = sentence.replace('π', '派').replace('Π', '派')
        sentence = sentence.replace('ρ', '肉')
        sentence = sentence.replace('ς', '西格玛').replace('Σ', '西格玛').replace(
            'σ', '西格玛')
        sentence = sentence.replace('τ', '套')
        sentence = sentence.replace('υ', '宇普西龙')
        sentence = sentence.replace('φ', '服艾').replace('Φ', '服艾')
        sentence = sentence.replace('χ', '器')
        sentence = sentence.replace('ψ', '普赛').replace('Ψ', '普赛')
        sentence = sentence.replace('ω', '欧米伽').replace('Ω', '欧米伽')
        # re filter special characters, have one more character "-" than line 68
        #部分标点会被用作韵律，jieb分词会有部分标点不支持，在这里替换为同等级可以支持的标点符号
        # sentence = re.sub(r'[-——《》【】<=>{}()（）#&@“”^_|…\\]', '', sentence)
        sentence = re.sub(r'[=#&@“”^_|…\\]', ' ', sentence)
        #对中英混的英文部分标点与单词之间添加空格
        sentence = re.sub(r'([a-zA-Z]+)([、.,\):!?-])', r'\1 \2 ', sentence)
        #不对标点进行替换
        # sentence = re.sub(r'[-——《》【】<>{}()（）“”^_|…\\]', '、', sentence)
        #对汉字前后的空格进行去除
        #"马上故事新编， 说： “你说的这事是有的——不是我喜欢她，";不然空格在bert韵律模型会被清除掉，导致"unk"获取不准确
        sentence = re.sub(r'([ \s]*)([\u4e00-\u9fa5])([ \s]*)', r'\2', sentence)
        return sentence

    def normalize_sentence(self, sentence: str) -> str:
        sentence = tranditional_to_simplified(sentence)
        #将不支持的符号转换为支持的符号
        sentence = trans_to_identify(sentence)
        sentence = sentence.translate(F2H_ASCII_LETTERS).translate(F2H_DIGITS).translate(F2H_SPACE)
        try:
            # pdb.set_trace()
            # basic character conversions
            # 固定格式： 2025-04-09 11:43:00
            sentence = fix_date_format_normalize(sentence)

            # 移除不成对的括号左边或者右边的内容
            sentence = handle_incomplete_brackets(sentence)
            
            # 匹配车牌号 京A12345
            sentence = RE_CAR_PLATE_NUMBER.sub(replace_car_plate_number, sentence)

            # 适配“会议号为660 235 325”、“云视讯是660-235-325”、““云视讯660-235-325””
            sentence = RE_MEETING_DIGIT.sub(replace_meeting_digit, sentence)

            # 匹配"参考"或"详见"+[ref_doc_数字]
            sentence = RE_KEYWORD_REFDOC.sub(replace_keyword_refdoc, sentence)
            # 匹配[ref_doc_数字] + '提到'或者'指出'
            sentence = RE_REFDOC_KEYWORD.sub(replace_refdoc_keyword, sentence)
            # 匹配[ref_doc_数字] , 全部移除
            sentence = RE_REFDOC_BLANK.sub(replace_refdoc_blank, sentence)


            # "2020-2025年","2010~2015年","2000—2005年",
            sentence = RE_YEAR_RANGE.sub(replace_year, sentence)
            # "(2020-2025)","(20-25年)","（-2020~-2025）","(2020年—2025)",
            sentence = RE_YEAR_MAN_RANGE.sub(replace_man_year, sentence)
            # "1-2月","03~05月","10—12月",
            sentence = RE_MONTH_RANGE.sub(replace_month, sentence)
            # "1-2日","05~10号","20—31日",
            sentence = RE_DAY_RANGE.sub(replace_day, sentence)

            # number related NSW verbalization
            # "2024年","2024年10月","2024年10月15日","60年5月20号","前面有-2024年","前面有2024年",
            sentence = RE_DATE.sub(replace_date, sentence)
            # "2024-05-15","2023/12/31","2022.01.01",
            sentence = RE_DATE2.sub(replace_date2, sentence)
            # "1·5","03·20","12·31",
            sentence = RE_EVENT.sub(replace_event, sentence)
            # "15声明","0320案件","1231事件",
            sentence = RE_EVENT2.sub(replace_event2, sentence)

            # range first
            # 时间范围，如8:30-12:30
            sentence = RE_TIME_RANGE.sub(replace_time, sentence)
            # pdb.set_trace()
            # "比例3:2", 内容中要包含"比例"等关键字，此字出现的位置不关键
            if RATIO_KEYWORD.search(sentence):
                sentence = RE_RATIO.sub(replace_ratio, sentence)

            # "09:30","23∶59","12:00:30","2:15",
            sentence = RE_TIME.sub(replace_time, sentence)
            # "-25-30°C","20 - 25.5度","0 - 10摄氏度",
            sentence = RE_TEMPERATURE_RANGE.sub(replace_temperature_range, sentence)
            # "-25°C","30℃","22.5度","40摄氏度",
            sentence = RE_TEMPERATURE.sub(replace_temperature, sentence)
            # "1.5~2.5kg","50—60min",
            sentence = RE_MEASURE_RANGE.sub(replace_measure_range, sentence)
            # "25cm","1.5kg","60min","3.14m²",
            sentence = RE_MEASURE.sub(replace_measure, sentence)
            # "1/2","-3/4","123/456",
            sentence = RE_FRAC.sub(replace_frac, sentence)

            # "50%-90%"，"50.25%~90.56%"
            sentence = RE_PERCENT_RANGE.sub(replace_percentage_range, sentence)

            # "25%","-10%","25.5%","30 %",
            sentence = RE_PERCENTAGE.sub(replace_percentage, sentence)
            # "13812345678","+86 13812345678","8613812345678",
            sentence = RE_MOBILE_PHONE.sub(replace_mobile, sentence)
            #先处理一些特殊的电话号码，例如：86-27-85855900
            sentence = RE_TELEPHONE_SPECIAL.sub(replace_phone, sentence)
            # "010-1234567","02112345678","1234567",
            sentence = RE_TELEPHONE.sub(replace_phone, sentence)
            # 全国统一的号码400开头
            sentence = RE_NATIONAL_UNIFORM_NUMBER.sub(replace_phone, sentence)

            # '- 南段：前门大街-南中轴御道-天桥-北京古代建筑博物馆-永定门公园-永定门城楼',将-替换成，
            sentence = RE_SHORT_Hengxian.sub(replace_short_hengxian, sentence)
            # 匹配，GLM-4.6 等，将 - 移除
            sentence = replace_hyphen_after_capital_word(sentence)

            # pdb.set_trace()
            #这里有问题，遇到单纯的数字会出现问题，暂时先注释掉
            # if RE_MATH.search(sentence):
            #     sentence = replace_math(sentence)

            # "1-5","-2.5~3.7",".5—2","1-56"
            sentence = RE_RANGE.sub(replace_range, sentence)
            # 带负号的整数 "-123","-5",
            sentence = RE_INTEGER.sub(replace_negative_num, sentence)
            
            # 纯小数 "12.34","-5.67",".8","-.9",
            sentence = RE_DECIMAL_NUM.sub(replace_number, sentence)
            # pdb.set_trace()
            # 正整数 + 关键字, 按照电报读法读 
            sentence = RE_NUM_KEYWORD_DIGIT.sub(replace_num_keyword_digit, sentence)
            # 正整数 + 关键字, 按照幂数读法读
            sentence = RE_NUM_KEYWORD_CARDINAL.sub(replace_num_keyword_cardinal, sentence)

            # pdb.set_trace()
            # 关键字 + 正整数 , 按照电报读法读
            sentence = RE_KEYWORD_NUM_DIGIT.sub(replace_keyword_num_digit, sentence)
            # 关键字 + 正整数 , 按照幂数读法读
            sentence = RE_KEYWORD_NUM_CARDINAL.sub(replace_keyword_num_cardinal, sentence)

            # pdb.set_trace()
            # # 2508会议室
            # sentence = RE_DIGIT_CHARS.sub(replace_digit_char, sentence)
            # # 创新大楼2508
            # sentence = RE_CHARS_DIGIT.sub(replace_char_digit, sentence)

            # CA1589  京ABD987  航班号MF7654 车牌号等等
            sentence = RE_CAPITAL_EN_DIGIT.sub(replace_capital_english_digit, sentence)

            # # "上海证券交易所，股票代码600941"
            # sentence = RE_GUPIAO_DIGIT.sub(replace_gupiao_digit, sentence)

            # “今天的收益是23,456美元”, 将上述的逗号移除
            sentence = remove_commas_from_numbers(sentence)
            # 以0开头的数字，均按照电报读法,比如00783
            sentence = RE_LEADING_ZERO.sub(replace_leading_zero, sentence)

            # 匹配 (1)、（2）、1)、2）、[3] 等编号格式
            sentence = RE_NUMBERING.sub(replace_numbering, sentence)

            # 只匹配 x+y=z 或者  1+2=3 或者 x+2=4的形式
            sentence = RE_PURE_MATH.sub(replace_pure_math, sentence)

            # 匹配固定的带数字的关键字，并将其中的数字按照电报读法去读
            sentence = RE_FIX_TEXT_NUMBER2DIGIT.sub(replace_num2digits_in_fix_text, sentence)

            # "123","-123","123.45","-123.45",".45","-.45",
            sentence = RE_NUMBER.sub(replace_number_default, sentence)

            # """移除文本中所有成对出现的括号及其内部内容，保留不成对的括号"""
            sentence = remove_paired_bracket_content(sentence)


            
            sentence = self._post_replace(sentence)
            #添加对大模型输出的拼音进行转换，例如：wáng
            sentence = RE_TRANS_PINYIN.sub(replace_pinyin, sentence)
            # 简 拼: abc   简拼：xyz
            sentence = RE_TRANS_JIANPIN.sub(replace_jianpin, sentence)
            return sentence
        except Exception as e:
            # pdb.set_trace()
            # print(e)
            logger.info(f"TextNormalize error :{e}")
            return sentence
    
    def split_sentence(self, text: str) -> List[str]:
        # sentences = self._split(text)
        sentences = self._split_as_punc(text=text)
        return sentences



if __name__ == '__main__':
    test_text = "我有-2348.9345元钱,还有12345.64567元存款,利率是5.35%,女士占2/3，计算300.23+200.69,等于多少,理论上-30.08加-500.369，今天日期2024/12/6，高度是300~500米， 8:30-12:30, 结果是6:15,手机号15010954372，电话010-68928732"
    # print(f'origin text: {test_text}')
    # result = text_normalizer.normalize_sentence(test_text)
    
    text_normalizer = TextNormalizer()
    test_text = '根据网页内容，2025年4月3日A股市场的状况如下'
    result = text_normalizer.normalize_sentence(test_text)
    print(result)


    # print(f'result: {result}')