#!/usr/bin/python
# -*- encoding: utf-8 -*-

import re
from typing import Dict
from typing import List

import numpy as np

from jttts.tts_model.ssml.xml_processor import MixTextProcessor
from jttts.tts_common.logger import logger
from jttts.tts_model.frontend.zh_normalization.text_normlization import TextNormalizer
from jttts.tts_model.frontend.en_normalization import en_norm
from jttts.tts_model.frontend.generate_lexicon import generate_tn_lexicon
from jttts.tts_model.frontend.front_utils import remove_markdown_characters, convert_FullEnglish_to_Half, collapse_whitespace
from jttts.tts_model.frontend.zh_normalization.symbol import remove_paired_bracket_content, replace_hyphen_after_capital_word, handle_incomplete_brackets

import os
import time
import pdb

class MixFrontend():
    def __init__(self,
                 front_type:int=0,
                 resource_path=None,
                 phone_vocab_path=None,
                 tone_vocab_path=None,
                 PHONE_TYPE:str="CMU",
                 PHONE_NATION:str="US",
                 use_rhy = False):

        self.front_type = front_type
        self.resource_path = resource_path
        self.tn_force_replace_path = os.path.join(self.resource_path, "TextNormalize", "tn_force_replace.txt")
        self.tn_force_replace_dict = {}
        is_success = self.update_tn_force_replace_dict()
        if is_success is False:
            logger.info("load tn_force_replace_dict failed,please check...")
            exit()

        self.text_normalizer = TextNormalizer()
        self.en_text_normalizer = en_norm


    def update_tn_force_replace_dict(self):
        try:
            self.tn_force_replace_dict   = generate_tn_lexicon(self.tn_force_replace_path)
            return True
        except Exception as e:
            logger.info("update tn_force_replace_dict failed:{},please check...".format(e))
            return False

    def is_chinese(self, char):
        if char >= '\u4e00' and char <= '\u9fa5':
            return True
        else:
            return False

    def is_alphabet(self, char):
        if (char >= '\u0041' and char <= '\u005a') or (char >= '\u0061' and
                                                       char <= '\u007a'):
            return True
        else:
            return False

    def is_english_alphabet(self, char):
        if (char >= 'a' and char <= 'z') or (char >= 'A' and char <= 'Z'):
            return True
        else:
            return False

    def is_other(self, char):
        if not (self.is_chinese(char) or self.is_alphabet(char)):
            return True
        else:
            return False
        
    def get_segment_language(self, text: str) -> List[str]:
        # sentence --> [ch_part, en_part, ch_part, ...]
        #输入为纯数字，按照中文读
        if re.sub(r"\d+","",text) == "":
            return "zh"
        #输入为非字母，按照中文读
        elif re.sub(r"[^a-zA-Z]","",text) == "":
            return "zh"
        #输入为字母和数字，按照英文读
        elif re.sub(r"[\u4e00-\u9fa5]","",text) == text:
            return "en"
        #数字、字母、数字都有，按照中英混读
        else:
            return "mix"

    def get_segment(self, text: str) -> List[str]:
        # sentence --> [ch_part, en_part, ch_part, ...]
        segments = []
        types = []
        flag = 0
        temp_seg = ""
        temp_lang = ""
        pinyin_span = []
        # pdb.set_trace()
        for pinyin_g in re.finditer(r"([a-z]+\d\s*)+",text):
            span = pinyin_g.span()
            for i in range(span[0], span[1]):
                pinyin_span.append(i)

        # Determine the type of each character. type: blank, chinese, alphabet, number, unk and point.
        for i, ch in enumerate(text):
            if i in pinyin_span:
                # print(i)
                types.append("zh_py")
            elif self.is_chinese(ch):
                types.append("zh")
            elif self.is_alphabet(ch):
                types.append("en")
            else:
                types.append("other")

        assert len(types) == len(text)

        for i in range(len(types)):
            # find the first char of the seg
            if flag == 0:
                temp_seg += text[i]
                temp_lang = types[i]
                flag = 1
            else:
                if temp_lang == "other":
                    if types[i] == temp_lang:
                        temp_seg += text[i]
                    else:
                        temp_seg += text[i]
                        temp_lang = types[i]
                else:
                    if types[i] == temp_lang:
                        temp_seg += text[i]
                    elif types[i] == "other":
                        temp_seg += text[i]
                    else:
                        segments.append((temp_seg, temp_lang))
                        temp_seg = text[i]
                        temp_lang = types[i]
                        flag = 1
        segments.append((temp_seg, temp_lang))
        return segments

    def split_tn_rhy(self, sentence: str):
        sentence = sentence.strip()
        sentence = convert_FullEnglish_to_Half(sentence)
        sentence = collapse_whitespace(text=sentence)
        sentence = remove_markdown_characters(sentence)
        sentence = handle_incomplete_brackets(sentence)

        d_inputs = MixTextProcessor.get_dom_split(sentence) # d_inputs = ['在创新大楼2506房间开会']
        tmpSegments = []
        text_after_tn = ''
        # pdb.set_trace()
        for instr in d_inputs:
            ''' 暂时只支持 say-as '''
            if instr.lower().startswith("<say-as"):
                tmpSegments.append((instr, "zh"))
            else:
                if len(instr) == 0:
                    continue
                # pdb.set_trace()
                for tnkey, tnvalue in self.tn_force_replace_dict.items():
                    tnvalue0 = tnvalue[0]
                    instr = instr.replace(tnkey, tnvalue0)

                # pdb.set_trace()
                instrs = self.text_normalizer.split_sentence(instr) # 切句后，保留原始的标点符号
                instrs_normal = []
                for ins in instrs:
                    if len(ins) == 0:
                        continue
                    lang = self.get_segment_language(ins)
                    if lang == "en":
                        ins = self.en_text_normalizer(ins)
                    else:
                        ins = self.text_normalizer.normalize_sentence(ins)
                    instrs_normal.append(ins)
                text_after_tn = ''.join(instrs_normal)
                text_after_tn = re.sub(r"\s+"," ",text_after_tn)
                instrs = instrs_normal
                for ins in instrs:
                    tmpSegments.append(self.get_segment(ins)) # tmpSegments = [[('在%创新%大楼`二千%五百%零六%房间%开会~', 'zh')]]

        return tmpSegments, text_after_tn
