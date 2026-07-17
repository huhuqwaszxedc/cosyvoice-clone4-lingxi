#!/usr/bin/python
# -*- encoding: utf-8 -*-

import sys
sys.path.append('./')
import xml.etree.ElementTree as ET
import re, pdb
from jttts.tts_model.ssml.ssml_tn import ssml_say_as_score, ssml_say_as_characters, ssml_say_as_date
from jttts.tts_model.frontend.zh_normalization.num import num2str, verbalize_digit

class ssml_tag_processor():
    def parse_ssml(self, ssml):
        ssml = ssml.strip()
        # 检查字符串是否以<speak>开头并以</speak>结尾
        if not ssml.startswith('<speak>') or not ssml.endswith('</speak>'):
            return ssml, False

        # 去掉<speak>和</speak>标签
        ssml_content = ssml[len('<speak>'):-len('</speak>')]
        
        # 定义正则表达式模式来匹配指定的标签
        phoneme_pattern = re.compile(r'(<phoneme.*?>)(.*?)(</phoneme>)', re.DOTALL)
        say_as_pattern = re.compile(r'(<say-as.*?>)(.*?)(</say-as>)', re.DOTALL)
        sub_pattern = re.compile(r'(<sub.*?>)(.*?)(</sub>)', re.DOTALL)
        
        # 初始化结果列表
        content = []
        
        # 使用正则表达式查找并解析指定的标签
        pos = 0
        while pos < len(ssml_content):
            phoneme_match = phoneme_pattern.search(ssml_content, pos)
            say_as_match = say_as_pattern.search(ssml_content, pos)
            sub_match = sub_pattern.search(ssml_content, pos)
            
            # 找到最近的匹配
            matches = [m for m in [phoneme_match, say_as_match, sub_match] if m]
            if not matches:
                break
            
            # 按匹配位置排序
            matches.sort(key=lambda m: m.start())
            match = matches[0]
            
            # 提取匹配前的文本并添加到结果
            if match.start() > pos:
                content.append({
                    'type': 'text',
                    'content': ssml_content[pos:match.start()]
                })
            
            # 提取匹配的标签内容
            tag = match.group(1)
            inner_text = match.group(2)
            end_tag = match.group(3)
            
            if 'phoneme' in tag:
                # alphabet_match = re.search(r'alphabet="(.*?)"', tag)
                alphabet_match = re.search(r'alphabet=[\'"](.*?)[\'"]', tag)
                ph_match = re.search(r'ph="(.*?)"', tag)
                if alphabet_match and ph_match:
                    alphabet = alphabet_match.group(1)
                    ph = ph_match.group(1)
                    content.append({
                        'type': 'phoneme',
                        'alphabet': alphabet,
                        'ph': ph,
                        'content': inner_text,
                    })
                else:
                    content.append({
                        'type': 'text',
                        'content': inner_text
                    })
            elif 'say-as' in tag:
                # interpret_as_match = re.search(r'interpret-as="(.*?)"', tag)
                interpret_as_match = re.search(r'interpret-as=[\'"](.*?)[\'"]', tag)
                
                if interpret_as_match:
                    interpret_as = interpret_as_match.group(1)
                    content.append({
                        'type': 'say-as',
                        'interpret-as': interpret_as,
                        'content': inner_text,
                    })
                else:
                    content.append({
                        'type': 'text',
                        'content': inner_text
                    })
            elif 'sub' in tag:
                # alias_match = re.search(r'alias="(.*?)"', tag)
                alias_match = re.search(r'alias=[\'"](.*?)[\'"]', tag)
                if alias_match:
                    alias = alias_match.group(1)
                    content.append({
                        'type': 'sub',
                        'alias': alias,
                        'content': inner_text,
                    })
                else:
                    content.append({
                        'type': 'text',
                        'content': inner_text
                    })
            
            # 更新位置
            pos = match.end()
        
        # 添加剩余的文本内容
        if pos < len(ssml_content):
            content.append({
                'type': 'text',
                'content': ssml_content[pos:]
            })
        
        return content, True

    def convert(self, text):
        if len(text.strip()) ==0:
            return text

        text , ssml_flag = self.parse_ssml(text)
        if ssml_flag is False:
            return text
        else:
            output_str = ''
            for idx, sub_text in enumerate(text):
                if sub_text['type'] == 'text':
                    output_str += sub_text['content']
                elif sub_text['type'] == 'sub':
                    output_str += sub_text['alias']
                elif sub_text['type'] == 'say-as':
                    if sub_text['interpret-as'] == 'cardinal': # 按整数或小数发音
                        output_str += num2str(sub_text['content'])
                    elif sub_text['interpret-as'] == 'digits': # 按数字发音。
                        output_str += verbalize_digit(sub_text['content'])
                    elif sub_text['interpret-as'] == 'telephone': # 按电话号码常用方式发音
                        output_str += verbalize_digit(sub_text['content'], alt_one = True)
                    elif sub_text['interpret-as'] == 'name': # 按人名发音(暂不支持)
                        pass
                    elif sub_text['interpret-as'] == 'address': # 按地址发音。
                        output_str += verbalize_digit(sub_text['content'], alt_one = True)

                    elif sub_text['interpret-as'] == 'id': # 适用于账户名、昵称等(暂不支持)
                        pass
                    elif sub_text['interpret-as'] == 'characters': # 将标签内的文本按字符一一读出。
                        output_str += ssml_say_as_characters(sub_text['content'])

                    elif sub_text['interpret-as'] == 'date': # 将标签内的文本按字符一一读出。
                        output_str += ssml_say_as_date(sub_text['content'])

                    elif sub_text['interpret-as'] == 'score': # 比分 12:30-->12比30
                        output_str += ssml_say_as_score(sub_text['content'])
                    else:
                        output_str += sub_text['content']
                else:
                    output_str += sub_text['content']
        
        return output_str


if __name__ == '__main__':
    ssml_string = '<speak>要一起去<phoneme alphabet="py" ph="chi1">吃</phoneme>饭吗, <哈哈> today 1244 and \
        <say-as interpret-as="score">12:30</say-as>，\
        <say-as interpret-as="cardinal">69.236</say-as>，\
        <say-as interpret-as="digits">9569</say-as>，\
        <say-as interpret-as="telephone">137889569</say-as>，\
        <say-as interpret-as="characters">english 5G</say-as>，\
        <say-as interpret-as="date">2018/03/03~1997.9.9</say-as>，\
        今天天气<sub alias="语音合成标记语言">SSML</sub>不错 </speak>'


    ssml_string = '<speak>要一起去<phoneme alphabet="py" ph="chi1">吃</phoneme>饭吗，今天晚上<say-as interpret-as="score">12:30</say-as>，,在创新大楼<say-as interpret-as="digits">25006</say-as>房间开会</speak>'

    ssml_processor_inst = ssml_tag_processor()
    merged_content = ssml_processor_inst.convert(ssml_string)
    print(merged_content)