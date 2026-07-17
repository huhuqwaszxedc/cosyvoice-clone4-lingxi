#!/usr/bin/python
# -*- encoding: utf-8 -*-
import sys
sys.path.append('./')
import os
import time
from typing import Optional
from jttts.tts_common.logger import logger
from typing import Dict, List, Optional

from jttts.tts_model.frontend.front_utils import contains_letter_or_digit, is_emoji
import re, pdb
from jttts.tts_model.ssml.ssml_processor import ssml_tag_processor
from collections import OrderedDict
from jttts.tts_model.frontend.mix_frontend import MixFrontend
from jttts.tts_model.frontend.zh_normalization.chronology import  RE_TRANS_PINYIN, replace_pinyin



class TTSFrontExecutor():
    def __init__(
            self,
            resource_path: Optional[os.PathLike]=None,
            phones_dict: Optional[os.PathLike]=None,
            tones_dict: Optional[os.PathLike]=None,
            wave_path: Optional[os.PathLike]=None,
            lang: str='mix',
            front_type: int=0, # 0--normal,  1--zipvoice pinyin
            PHONE_TYPE:str="CMU",
            PHONE_NATION:str="US"
            ):
        """
        Init model and other resources from a specific path.
        """
        #设置输出音频路径
        self.wave_path = wave_path
        self.front_type = front_type

        if resource_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            resource_path = os.path.join(current_dir, 'resource')
        self.use_rhy = False
        self.frontend = MixFrontend(front_type = self.front_type, resource_path=resource_path,
            phone_vocab_path=phones_dict, tone_vocab_path=tones_dict,PHONE_TYPE=PHONE_TYPE, PHONE_NATION=PHONE_NATION, use_rhy=self.use_rhy)
        self.ssml_tag_process = ssml_tag_processor()


    def remove_emoj(self, text):
        out_str = ''
        for idx in range(len(text)):
            if is_emoji(text[idx]):
                pass
            else:
                out_str += text[idx]
        return out_str
    
    def only_tn_rhy(self, text: str,):
        text = self.remove_emoj(text=text)
        out_segments, text_after_tn = self.frontend.split_tn_rhy(text)
        return out_segments, text_after_tn

    def input_text_pinyins_preprocess(self, text_pinyins:str= ''):
        #添加对大模型输出的拼音进行转换，例如：wáng
        text_pinyins = RE_TRANS_PINYIN.sub(replace_pinyin, text_pinyins)

        text_pinyins_list = text_pinyins.strip().split()
        out_text_pinyins = []
        check_pass_flag = True
        out_notpass_pinyins = []
        for idx, sub_pinyin in enumerate(text_pinyins_list):
            if sub_pinyin[-1] == '5':
                if sub_pinyin[:-1] in self.backend_vocab:
                    out_text_pinyins.append(sub_pinyin[:-1])
                else:
                    check_pass_flag = False
                    out_notpass_pinyins.append(sub_pinyin)
            elif sub_pinyin in self.backend_vocab:
                out_text_pinyins.append(sub_pinyin)
            elif not contains_letter_or_digit(sub_pinyin):
                out_text_pinyins.append(sub_pinyin)
            else:
                check_pass_flag = False 
                out_notpass_pinyins.append(sub_pinyin)

        if check_pass_flag:
            out_str = ' '.join(out_text_pinyins)
            return True, out_str
        else:
            out_str = ' '.join(out_notpass_pinyins)
            return False, out_str


    def contains_chinese(self, text):
        # 正则表达式匹配中文字符
        pattern = re.compile(r'[\u4e00-\u9fa5]')
        
        # 搜索字符串中是否有中文字符
        if pattern.search(text):
            return True
        else:
            return False

    def is_alphabet(self, char):
        if (char >= 'a' and char <= 'z') or (char >= 'A' and char <= 'Z'):
            return True
        else:
            return False

    def lookup_forward_valid_symbol(self, pinyin_list):
        '''
            从末尾向前查找最后一个非空字符
        '''
        out_str = ''
        for idx in range(len(pinyin_list)):
            idx_2 = len(pinyin_list) -1 - idx
            if pinyin_list[idx_2] != ' ':
                if len(pinyin_list[idx_2]) == 1 and is_emoji(pinyin_list[idx_2]):
                    pass
                else:
                    out_str = pinyin_list[idx_2]
                    break
        return out_str

    def lookup_backword_valid_symbol(self, pinyin_list):
        '''
            从开头向后查找最后一个非空字符
        '''
        out_str = ''
        for idx in range(len(pinyin_list)):
            if pinyin_list[idx] != ' ':
                if len(pinyin_list[idx]) == 1 and is_emoji(pinyin_list[idx]):
                    pass
                else:
                    out_str = pinyin_list[idx]
                    break
        return out_str


    def replace_emoji_with_punc(self, pinyin_list):
        # pdb.set_trace()
        out_list = []
        for idx, sub_pinyin in enumerate(pinyin_list):
            if  len(sub_pinyin) == 1 and is_emoji(sub_pinyin):
                if len(out_list) == 0:
                    pass
                elif idx == len(sub_pinyin) - 1: # 最后一个是emoji表情
                    out_str_1 = self.lookup_forward_valid_symbol(out_list)
                    if contains_letter_or_digit(out_str_1): # 前面是pinyin序列
                        if len(out_str_1) > 1: # 中文
                            out_list.append('。')
                        else: # 英文
                            out_list.append('.')
                else:
                    out_str_1 = self.lookup_forward_valid_symbol(out_list)
                    out_str_2 = self.lookup_backword_valid_symbol(pinyin_list[idx+1:])
                    if contains_letter_or_digit(out_str_1)  and contains_letter_or_digit(out_str_2): # 前面和后面都是pinyin序列
                        if len(out_str_1) > 1: # 中文
                            out_list.append('，')
                        else: # 英文
                            out_list.append(',')
                    
            else:
                out_list.append(sub_pinyin)
        return out_list

    def match_infer(self, text, text_pinyins=None, text_pho=None, lang="mix"):
        # pdb.set_trace()
        valid_pinyins_list, valid_span_list = self.infer(text=text, text_pinyins= text_pinyins, wave_path=self.wave_path, lang= lang)
        # pdb.set_trace()
        if self.front_type == 1:
            out_pinyin_list = self.zip_zh_pinyin(valid_pinyins_list)

            # 如果text是最后一个是标点符号，则需要确保text和out_pinyin_list的最后标点符号保持一致。
            # if text[-1] != out_pinyin_list[-1]:
            #     if self.contains_chinese(text[-1]) or self.is_alphabet(text[-1]): # contains_letter_or_digit(out_pinyin_list[-1]) and
            #         pass
            #     else:
            #         if contains_letter_or_digit(out_pinyin_list[-1]):
            #             out_pinyin_list.append(text[-1])
            #         else:
            #             out_pinyin_list[-1] = text[-1]

            out_pinyin_list = self.replace_emoji_with_punc(out_pinyin_list)

            # 句子结尾的地方，必须是标点符号,如果没有，自动添加
            if contains_letter_or_digit(out_pinyin_list[-1]):
                if len(out_pinyin_list[-1]) > 1:
                    out_pinyin_list.append('。')
                else:
                    out_pinyin_list.append('.')

        else:
            if self.use_rhy is False:
                out_pinyin_list = []
                for sub_py in valid_pinyins_list:
                    if '#' not in sub_py:
                        out_pinyin_list.append(sub_py)
                    
            else:
                out_pinyin_list = valid_pinyins_list

        return out_pinyin_list


if __name__ == '__main__':
    import jttts.tts_model.frontend.config_frontend as cfg

    #获取脚本所在路径
    base_path = cfg.base_path
    #音素转id词典
    phones_dict = cfg.phones_dict
    #暂时不设置音调的字典
    tones_dict = cfg.tones_dict
    #init音频存放地址
    wave_path = cfg.wave_path


    #音标类型,选填值为：["CMU","IPA"] 默认"CMU"
    PHONE_TYPE = cfg.PHONE_TYPE
    #英语类型,选填值为：["US","UK"] 默认"US"
    PHONE_NATION = cfg.PHONE_NATION
    #初始化引擎
    logger.info("初始化引擎...")

    if 0:
        # if only_front:
        #     front_tts = TTSFrontExecutor(resource_path=base_path, phones_dict=phones_dict,tones_dict=None,vits_config=vits_config,
        #                             vits_path=vits_path,wave_path=wave_path,only_front=only_front,PHONE_TYPE=PHONE_TYPE,PHONE_NATION=PHONE_NATION)
        #     app = FastAPI() # 创建api对象
        #     @app.get("/text={text}") # 根路由
        #     def front(text):
        #         text_pho,cost_time = front_tts.front_process(text=text)
        #         return {"front_phonemes": text_pho,"cost_time(秒)":cost_time}
        #     uvicorn.run(app=app,host="0.0.0.0",port=7860,workers=1)
        # else:
        #     import gradio as gr
        #     init_tts = TTSFrontExecutor(resource_path=base_path, phones_dict=phones_dict,tones_dict=None,vits_config=vits_config,
        #                         vits_path=vits_path,wave_path=wave_path,only_front=only_front,PHONE_TYPE=PHONE_TYPE,PHONE_NATION=PHONE_NATION)
        #     with gr.Blocks() as demo:
        #         logger.info("开启语音合成服务...")
        #         text = [gr.Textbox(label="请输入需要合成的文本"),gr.Textbox(label="或请输入需要合成的音素(*有音素时合成音素，无音素时合成文本)")]
        #         output = [gr.Audio(label="合成语音",value=wave_path),gr.Textbox(label="前端生成音素")]
        #         greet_btn = gr.Button("开始合成")
        #         greet_btn.click(fn=init_tts.greet, inputs=text, outputs=output)
        #     demo.launch(server_name="127.0.0.1",server_port=7861, share=True)
        pass
    else:
        only_front = True
        lang="mix"
        front_type = 0
        front_tts = TTSFrontExecutor(resource_path=base_path, phones_dict=phones_dict,tones_dict=None,
                wave_path=wave_path, lang = lang, front_type = front_type, PHONE_TYPE=PHONE_TYPE,PHONE_NATION=PHONE_NATION)

        text = '九九提示你，今天晚上19:45,在创新大楼2506房间开会。'
        text = "<speak>要一起去<phoneme alphabet='py' ph='chi1'>吃</phoneme>饭吗，今天晚上<say-as interpret-as='score'>12:30</say-as>，,在创新大楼<say-as interpret-as='digits'>2506</say-as>房间开会, \
            a vast network and over 900 million users . </speak>"
        text = '在创新大楼2506房间路转溪头忽见开会,hello world,nice to meet you.'
        text = 'hello world,nice to meet you.中国目前已开始部署近地小行星防御系统，全球科学家也正以行星防御为纽带展开协作。'

        if front_type == 1:
            f5_pinyin_list = front_tts.match_infer(text=text, lang=lang)
            # pdb.set_trace()
            f5_pinyin_str = ' '.join(f5_pinyin_list)
            print(f"原始文本序列: {text}")
            print(f"f5输出的文本的拼音序列：{f5_pinyin_list}")

            from test_tools.test_f5_front import convert_char_to_pinyin
            origin_f5_pinyin_list = convert_char_to_pinyin(text_list=[text])
            origin_f5_pinyin_str = ' '.join(origin_f5_pinyin_list)
            print(f"原始f5输出的文本的拼音序列：{origin_f5_pinyin_list}")
            pdb.set_trace()
        else:
            fshi_pinyin_list = front_tts.match_infer(text=text, lang=lang)
            # pdb.set_trace()
            fshi_pinyin_str = ' '.join(fshi_pinyin_list)
            print(f"原始文本序列: {text}")
            print(f"fshi输出的文本的拼音序列：{fshi_pinyin_str}")


