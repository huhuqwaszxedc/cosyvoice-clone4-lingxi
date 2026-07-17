#!/usr/bin/python
# -*- encoding: utf-8 -*-

import re
from collections import OrderedDict
from jttts.tts_common.logger import logger
import pdb
from jttts.tts_common.tts_fileencoder import check_encrypt_file_valid, text_file_decrypt

def generate_tn_lexicon(file_path:str):
    """Generate tn lexicon."""
    tn_dict = {}
    file_path, is_encrypt = check_encrypt_file_valid(file_path, loginfo='tn')
    if is_encrypt:
        txt_list = text_file_decrypt(file_path)
    else:
        with open(file_path,"r",encoding="utf-8") as readf:
            txt_list = readf.readlines()
    # pdb.set_trace()
    for txt in txt_list:
        txt_split = txt.strip().split("\t")
        if len(txt_split) <2 or len(txt_split) > 3:
            continue

        if len(txt_split) == 2:
            origin_text = txt_split[0].strip()
            dst_text = txt_split[1].strip()
            tn_level = 0
        else:
            origin_text = txt_split[0].strip()
            dst_text = txt_split[1].strip()
            tn_level = txt_split[2].strip()

        if origin_text[0] == '#' or len(origin_text) == 0 or origin_text == ' ' or len(dst_text) ==0 or dst_text == ' ':
            continue
        tn_dict[origin_text] = [dst_text, tn_level]
    return tn_dict



# Regular expression matching whitespace:
def collapse_whitespace(text):
    _whitespace_re = re.compile(r"\s+")
    text = re.sub(_whitespace_re, " ", text) # 多个空格变成一个空格
    return text
