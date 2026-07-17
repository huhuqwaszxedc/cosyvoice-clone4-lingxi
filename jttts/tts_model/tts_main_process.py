#!/usr/bin/python
# -*- encoding: utf-8 -*-
import sys
sys.path.append('./')
import os
import time
from typing import Optional
from jttts.tts_common.logger import logger
from typing import Dict

import re, pdb, torch

import soundfile as sf
import numpy as np
from collections import OrderedDict
import jttts.tts_model.frontend.config_frontend as config_frontend


from jttts.config import GlobalConfigInst


import uuid
import threading
import json

class TTSModelMainExecutor():
    def __init__(
            self,
            front_type = 1,
            only_front = False
            ):
        """
        Init model and other resources from a specific path.
        """

        # ******************frontend init*********************
        #获取脚本所在路径
        base_path = config_frontend.base_path
        #音素转id词典
        phones_dict = config_frontend.phones_dict
        #暂时不设置音调的字典
        tones_dict = config_frontend.tones_dict
        #init音频存放地址
        wave_path = config_frontend.wave_path

        #音标类型,选填值为：["CMU","IPA"] 默认"CMU"
        PHONE_TYPE = config_frontend.PHONE_TYPE
        #英语类型,选填值为：["US","UK"] 默认"US"
        PHONE_NATION = config_frontend.PHONE_NATION
        lang="mix"
        self.front_type = front_type
        self.only_front = only_front
        self.front_model = None

        # remove jieba.cache file
        os.system(f'rm -rf /tmp/jieba* ')
        from jttts.tts_model.frontend.tts_frontend import TTSFrontExecutor
        self.front_model = TTSFrontExecutor(resource_path=base_path, phones_dict=phones_dict,tones_dict=None,
                wave_path=wave_path, lang = lang, front_type = self.front_type, PHONE_TYPE=PHONE_TYPE,PHONE_NATION=PHONE_NATION)
        
        

        if self.only_front:
            return

        current_file_dir = os.path.dirname(__file__)
        sys.path.append(os.path.join('./', 'jttts/tts_model/backend/'))
        # sys.path.append(os.path.join('./', './jttts/tts_engine/jtaudio/'))

        

    def text_preprocess(self, text):
        if self.front_type == 1:
            text = text.replace("，", ",")
            text = text.replace("。", ".")
            text = text.replace("！", "!")
            text = text.replace("？", "?")
            text = text.replace("；", ";")
            text = text.replace("：", ":")
            text = text.replace("、", ",")
            text = text.replace("‘", "'")
            text = text.replace("“", '"')
            text = text.replace("”", '"')
            text = text.replace("’", "'")
            text = text.replace("⋯", "…")
            text = text.replace("···", "…")
            text = text.replace("・・・", "…")
            text = text.replace("...", "…")
        return text



    def text_prenormlize(self, text):
        return text

    def get_pinyins(self, text=None, text_pinyins=None,):
        # pdb.set_trace()
        if not text_pinyins:
            text = self.text_preprocess(text)
            text = self.text_prenormlize(text)
        else:
            is_check_pass, text_pinyins = self.front_model.input_text_pinyins_preprocess(text_pinyins=text_pinyins)
            if is_check_pass is False:
                print(f'The pinyins is not support ! {text_pinyins}')
        phones_list = self.front_model.match_infer(text=text, text_pinyins=text_pinyins)
        return phones_list


    def actual_get_userinfo(self, user_id = '', req: dict ={}):

        json_file = os.path.join(GlobalConfigInst.speaker_info_dir, user_id + ".json")
        speaker_info_dir = GlobalConfigInst.speaker_info_dir
        if os.path.exists(json_file) is False:
            json_file = os.path.join(GlobalConfigInst.build_in_speaker_info_dir, user_id + ".json")
            speaker_info_dir = GlobalConfigInst.build_in_speaker_info_dir

        if os.path.exists(json_file) is False:
            return None, None

        with open(json_file, 'r', encoding="utf-8") as fjson:
            spk_info = json.load(fjson)

        if './speaker_info/' in spk_info['prompt_wav']:
            spk_info['prompt_wav'] = spk_info['prompt_wav'].replace('speaker_info/', '')

        refer_wav_path = os.path.join(speaker_info_dir, spk_info['prompt_wav']) 
        if os.path.exists(refer_wav_path) is False:
            return None, None

        return spk_info['prompt_text'], refer_wav_path

    def actual_infer(self, text=None, user_id='F', req: dict = {}):

        if req['debug_mode'] == 2:
            pass
        else:
            text = self.text_preprocess(text)
            text = self.text_prenormlize(text)
        request_id_2 = 'model_init' + '_' + str(uuid.uuid4())

        prompt_text, ref_audio_path = self.actual_get_userinfo(user_id=user_id, req=req)

        audio_data = self.inference(request_id_2, ref_audio_path, prompt_text, text, req=req)

        return audio_data, self.sample_rate

    def only_tn_rhy(self, text = None):
        out_segments = self.front_model.only_tn_rhy(text=text)
        return out_segments



class ali_ttsfrd():
    def __init__(self) -> None:
        import ttsfrd # pip install https://modelscope.oss-cn-beijing.aliyuncs.com/releases/dependencies/ttsfrd/linux/ttsfrd-0.2.1-cp310-cp310-linux_x86_64.whl

        self.current_conda_env = os.getenv('CONDA_DEFAULT_ENV')

        if self.current_conda_env == 'cosv':
            RESOURCE_DIR = '/mnt/d/work/git_code/pretrained_models/CosyVoice-ttsfrd/resource'
        elif self.current_conda_env == 'ttsfront':
            RESOURCE_DIR = '/mnt/d/work/git_code/pretrained_models/speech_ptts_autolabel_16k_ttsfrd/resource'
        else:
            print(f'conda env is not support !')
            exit()

        self.fe_ins = ttsfrd.TtsFrontendEngine()
        self.fe_ins.initialize(RESOURCE_DIR) # final_dir = '/root/.cache/modelscope/hub/damo/speech_ptts_autolabel_16k/model/audio2phone_assets/ttsfrd/resource'
        # default Zh-CN
        self.fe_ins.enable_erhua(True)
        
        if self.current_conda_env == 'cosv':
            self.fe_ins.set_lang_type('pinyinvg')
        elif self.current_conda_env == 'ttsfront':
            self.fe_ins.set_lang_type("Zh-CN")



    def infer(self, text):
        text_prosody = ''
        # text_prosody = self.fe_ins.get_frd_extra_info(text, "finalbreak")
        # text_prosody = '另外#1韩国#1某#1private#1seelbank#1保管了#1李英爱的#1脐待鞋#4'
        text_prosody = text_prosody.replace("\n", "")
        # text_prosody = text_prosody.replace("#3", "#1")
        text_pinyin = self.fe_ins.get_frd_extra_info(text, "finalpron")
        # text_pinyin = 'ling4 wai4 han2 guo2 mou3 / P R AY1 . V AH0 T / S IY1 L . B AE1 NG K / bao2 guan3 le5 li3 ying1 ai4 de5 qi2 dai4 xie2 '
        text_pinyin = text_pinyin.strip(" /")
        text_pinyin = text_pinyin.replace("\n", "")

        return text_prosody, text_pinyin

    def text_normalize(self, text):
        if self.current_conda_env != 'cosv':
            print(f'conda env is not support !')
            exit()
        # pdb.set_trace()
        out_text = json.loads(self.fe_ins.do_voicegen_frd(text))["sentences"]
        out_text = out_text[0]['text']
        return out_text





if __name__ == '__main__':

    # setting
    only_front = True
    only_tn    = True
    front_type = 2 # 0--normal, 1-f5, 2-only tn
    use_ali_front = False

    # pdb.set_trace()
    current_file_dir = os.getcwd()
    sys.path.append(os.path.join(current_file_dir, 'jttts/tts_model/backend/'))
    sys.path.append(os.path.join(current_file_dir, './jttts/tts_engine/jtaudio/'))

    TTSModelInst = TTSModelMainExecutor(front_type=front_type, only_front = only_front)
    # pdb.set_trace()
    ali_ttsfrd_inst = None
    if use_ali_front:
        ali_ttsfrd_inst = ali_ttsfrd()

    text_pinyins = None
    # text = '请稍等，(联网查询中'
    # text = '根据网页内容，以下是2025年中国财政的一些数据：'
    # text = '哎呀，快把我吓死了，狼肚子里真黑呀。从经济角度来看，提前还贷的前提是要对比其他投资机会，如果发现提前还贷的性价比更高，才会选择这种反向操作，但当前环境下，这种操作可能并不是最优选择。'
    # text = '哎呀，快把我吓死了，狼肚子里真黑呀。Promote 598362 communication and exchange between countries and regions through industry connections'
    text = '希望你以后能够做的比我还好呦。昨天，北京低空湿度条件不利，it is expected to become even more integrated into daily life,但动力条件真好，最后是大力出奇迹，就像拧毛巾，最后拧出来了。'
    text = "Yes, it's Friday,March 7th,2025.Anything special planned for today?"
    text = '哎呀，👟🌂😊是不是想说温度 17°C 到 5°C 呢👟🌂😊'
    text = '温度挺低的，只有0到9℃哦  😌 ? 别忘穿暖些！😌'
    text = '未能找到直接关于西安上周(2025-02-17至2025-02-23)的具体财经新闻报道'
    text = '后来他遇到了一只小兔子，小兔子说：“你的刺好特别，就像你的名片一样，让我一眼就能认出你。”小刺猬听了很开心，从此不再烦恼啦眨眼,'
    text = "我不会主动关闭自己哦，不过如果用户不想和我聊天了，我就会乖乖的待着啦~你是不是想让我“休息”一下呀?"
    text = 'The English spelling is E-N-G-L-I-S-H. '
    text = 'B-A-N-A-N-A,就是这六个字母啦！是不是挺简单的？那你还想知道其他水果的英文拼写吗？'
    text = '看到小猫咪的样子，笑着说：“小猫咪，你变成这样也好可爱呀！”小猫咪听了，good morning and hello world.'
    text = '\n3.**科技领域投资激增**，提到了DeepSeek的火爆引发了对人工智能、云计算等领域的大量投资，这预示着科技行业可能迎来新一轮的变革和突破。'
    text = ' 9. 青田迎来2025年首场降雪：虽然是在春季，但仍给当地带来了独特的景色。'
    text = '好嘛，那我给你讲个 小兔子找朋友 的故事吧🐰 小兔子想找朋友，它遇到了小松鼠、小刺猬和小乌龟，你猜它最后和谁成为了好朋友呢。'
    text = '比例范围是-50.23%-90.89%，具体以测试为准。'
    text = '青田迎来20%-30%的首场降雪,但仍给当地1998年带来了独特的景色。'
    text = '提到了HelloWorld的火爆引发了对人工智能、云计算等领域的大量投资，这预示着科技行业可能迎来新一轮的变革和突破。'
    text = '发改委的主任是谁。'
    text = '清清爽爽的一整天, Good morning and hello world。'
    text = '好嘛，那下次再聊咯。'
    text = '那就来一道简单的吧：2x + y = 7，x - y = 1，求x和y的值。'
    text = '来了!x+y+z=6,2x-y+z=3,3x+2y-z=4。解法类似哦，先消去一个未知数，变成二元一次方程组，再解就好啦。'
    # text = 'zhǎng 这个读音呀，像小树苗慢慢长大，花朵绽放，都是 zhǎng 的过程呢，可神奇了。'
    # text = '长有两个读音呢，一个是 cháng ，表示长度、时间等，另一个是 zhǎng ，表示生长、增加等。'
    # text = '那卷卷的羽毛可真好看。'
    # text = '中国和柬埔寨签署了关于构建新时代全天候中柬命运共同体的联合声明。'
    text = '上海证券交易所，公司号码0600941.'
    # text = '新闻总是不断更新。'
    # text = 'x+y+z=6，2x-y+z=3，3x+2y-z=4。'
    # text = '那让我上网查查看今天的天气咋样。'
    # text = '- 南段：前门大街-南中轴御道-天桥-北京古代建筑博物馆-永定门公园-永定门城楼'
    # text = '那就来一道简单的吧：2x + y = 7，x - y = 1，求x和y的值。'
    text = '这是一个YUERW89723航班号 MF8199 的示例, 车牌号为京AB86387。'

    # text = '我是您的个人助手，你安排的日程：会议，将于2025-04-09 11:43:00开始，请您别忘了参加。'
    # text = '- 10:45，又在创新大楼 2508有一个会议。还在2365会议室也有会。'
    # text = '不过要是你说的是 1033 和 18，那平均值就是 (1033+18) ÷ 2 = 1050.5'
    # text = '孙悟空的笑话：土地公公，没有轻重音.'
    text = '中国移动,在上海证券交易所的代码是600941,电信的股票代码是601728.'
    # text = '帮我上网查查今天的天气,太阳公公也很不错哦.'
    text = '创新908会议室'
    # text = '我这边没有具体的版本号信息哦。'
    # text = '深夜的舞池里，电子乐强劲的鼓点与闪烁的霓虹交织，将每个人的热情都点燃到了极致。'
    text = '客服号码为10086,股票代码是08937, 在创新大楼2508开会，请到569房间.付款金额为234,总共有12306个人。'
    text = '今天下午2点开会。'
    text = 'F1赛车比赛,在G201上,航班号MF7654'
    text = '中国移动Jiutian大模型。'
    text = '抱歉，目前没有关于7月30日的天气预报数据。'
    text = ' 一个成年人在地球的重力大概在400 - 600牛顿。'
    # text = 'SO₂、NO₂、O₃和CO的浓度分别为1、4、31和5微克/立方米。'
    text = '现在是下午2:30'
    text = '上半场的为3:4,上述代码.'
    text = '合成 deepseek-R1 deepseek-V3 横线读成了减'
    text = '本田汽车CRX-V'
    text = '来了!x+y=6,x-y=3,的解法'
    text = ' 温度范围。30°C 至 23°C。'
    text = '最高温度为30℃，最低温度为23℃。'
    text = '夜间温度。23℃'
    text = '中午12点有个需求会，会议号为660-235-325，已经提醒过了,会议号660 235 325, 云视讯号是123456789'
    # text = '2025世界人工智能大会,在21世纪初期。2026中国必胜。'
    # text = '参加中俄海上联合-2025联合演习的双方舰艇编队完成海上科目演练，转入海上联合巡航[ref_doc_4]。'
    # text = '使读者能够全面了解新闻事件的各个方面参考[ref_doc_1][ref_doc_4][ref_doc_5]。'
    # text = '使读者能够全面了解新闻事件的各个方面参考[ref_doc_1],在这里[ref_doc_4]提到了这个问题[ref_doc_1][ref_doc_4][ref_doc_5].'
    text = '有个读法是忌(鸡)嘴压（呀）缩，是忌(鸡嘴【压】呀）缩.'
    text = '1)有个读法是忌鸡)嘴压呀）缩,(3)今天的天气.'
    text = '接口参数包括request_id、user_id等等.'

    text = '昨天，你来了吗，it is expected to become even more integrated into daily life,但动力条件真好。'
    # text = '让我查查今天的天气，你说说小狐狸这个事情。'
    # text = '那让我上网查查今天的天气咋样，太阳公公都快出来了。'
    text = '现在是下午2:30，帮京G21您核对鲁K23，车牌号是豫N125ABG1CDE 120个人，豫G23K23,请问对吗9981？'
    text = '🔔 建议关注：12月30日即将举行的“**2026科学跨年之夜**”，'
    text = '### 🕊️ 一、俄乌冲突与和平谈判新进展\n'
    text = '📅 **2025年12月31日 今日重点科技新闻汇总**  '
    text = '天津五大道超有韵味，2030多栋欧式小洋楼，像走进“万国建筑博物馆”,2口人.'
    text = '现在国内大模型真是百花齐放～像百度的文心大模型5.0，参数高达2.4万亿，'
    text = '还有智谱的GLM-4.6，写代码特别溜～要说谁最厉害，'
    text = "我最喜欢看《中国新声代2026》和《幸福账单2025》，期待《新说唱2025》的到来，新说唱2025个人。节目《2001太空漫游》很精彩。还有一个例子是《新说唱2024再战2024》。"
    text = "这是第一句话。(这是一半"
    text = "type=manus&sessionId=e4cfd7dd-c104-423c-b5c4-a1ec10d905e2&txt=%E5%B8%AE%E6%88%91%E8%B0%83%E7%A0%94%E4%B8%80%E4%B8%8B%E4%B8%AD%E5%9B%BD%E7%A7%BB%E5%8A%A8%E4%B9%9D%E5%A4%A9%E5%92%8C%E6%95%B0%E6%99%BA%E5%8C%96%E9%83%A8%E5%90%88%E5%B9%B6%E7%9A%84%E5%89%8D%E6%99%AF%EF%BC%8C%E7%94%9F%E6%88%90%E4%B8%80%E4%BB%BD%E6%B7%B1%E5%BA%A6%E5%88%86%E6%9E%90%E6%8A%A5%E5%91%8A&from=9981)"
    text = '任务即将开始执行，预计用时10分钟左右，执行完成后我们会短信通知您，您也可随时点击链接查看详情：[深度报告](https://jiutian.10086.cn/jiujiuassist/chat?'
    text = '你问的是豆包1.0、1.1到1.5的基础路线'
    text = '天津五大道超有韵味，2000多栋欧式小洋楼，像走进“万国建筑博物馆”'
    text = '受此带动，半导体设备ETF易方达（159558）连续3日净流入超2.66亿元，'
    text = '受此带动，半导体设备ETF易方达(159558)连续3日净流入超2.66亿元，'

    # text = "Yes, it's Friday,March 7th,2025.Anything special planned for today?"
    # text_pinyins_list = [' ', 'fa1', 'zhan3', 'zhong1', 'ri4', 'guan1', 'xi4', 'xu1', 'ke4', 'fu2', 'kun4', 'nan2', '、', 'jie3', 'jue2', 'wen4', 'ti2', '、', 'jia1', 'bei4', 'nu3', 'li4', '。']
    # text_pinyins       = ' '.join(text_pinyins_list)
    # text_pinyins       = 'zuò fǎ zì bì'


    if only_tn:
        tn_result = TTSModelInst.only_tn_rhy(text = text)
        print(f'origin_text: {text}')
        print(f'tn_result  : {tn_result[-1]}')
        exit()

        texts = [
            '今年夏天，备受瞩目的歌手2025与披荆斩棘2025将同期播出，引发观众热议。',
            '中国新声代2026作为一档面向青少年的音乐节目，旨在挖掘下一代的音乐潜力。',
            '他不仅是新说唱2025的冠军，也在有歌2024中留下了令人深刻的作品。',
            '萌探2024的嘉宾阵容每年都在变化，总能带来意想不到的化学反应。',
            '爱情保卫战2026节目中，情感导师们为无数迷茫的情侣指点迷津。',
            '男生女生向前冲2026以其独特的竞技模式吸引了大量年轻观众。',
            '非你莫属2026作为求职类节目，为职场新人提供了展示自我的平台。',
            '创造营2021虽然已经收官很久，但许多选手的发展轨迹仍是粉丝关注的焦点。',
            '群英会2026汇聚了各行各业的精英人士，共同探讨前沿话题。',
            '跨时代战书2026以新颖的赛制连接了不同年代的音乐风格。',
            '创业中国人2026聚焦创业者的故事，传递商业智慧与奋斗精神。',
            '跑男来了2025的户外挑战总是充满乐趣和惊喜。',
            '你好，星期六2025是周末放松心情的不错选择，充满了欢声笑语。',
            '非诚勿扰2025的舞台上，单身男女勇敢地追求着自己的幸福。',
            '星光大道2026为广大草根歌手提供了一个实现梦想的广阔舞台。',
            '华西论健2026邀请权威专家普及健康知识，深受观众信赖。',
            '声鸣远扬2025专注于推广具有地域特色的民族音乐。',
            '我家那闺女2025真实记录了明星父母与子女的日常生活。',
            '全员加速中2025紧张刺激的游戏环节让观众大呼过瘾。',
            '越战越勇2026鼓励普通人勇敢面对生活中的困难与挑战。',
            '非常话题2026深入探讨社会热点，引发公众思考。',
            '笑逐言开2026汇集了众多喜剧人才，为观众带来欢乐。',
            '星推荐2026是了解最新影视资讯的重要窗口。',
            '幸福账单2025通过游戏的方式帮助普通人实现心愿。',
            '大明王朝1566和特赦1959都以其深刻的历史题材和精湛的制作赢得了好评。',
            '电影银翼杀手2049与2001太空漫游都是科幻电影史上的里程碑之作。',
            '请回答1988和致1999年的自己这两部剧都以细腻的情感描绘了青春记忆。',
            '从红楼梦87版到现代剧，经典作品总能跨越时间的考验。末路1997也是一部值得回味的经典港片。',
            '他喜欢观看历史题材的电视剧，像刀锋1937和暗杀1932这样的剧集他都看过好几遍。同样，绝密1950和1945黎明之战也令他印象深刻。',
            '中国好声音2023、新说唱2024、有歌2026以及乘风2025等音乐综艺，共同构成了近年来华语乐坛的一道亮丽风景线。此外，爱情保卫战2025、男生女生向前冲2025、非你莫属2025、群英会2025、你好，星期六2026、非诚勿扰2026、开门大吉2026（此处重复，但算作两次提及）、星光大道2025、非常话题2026、笑逐言开2026、星推荐2026、唐探1900、749局、胜利1945、1950他们正年轻、我的1919和警察故事2013也都是各具特色的热门节目或影片，丰富了大众的文化生活。',

        ]

        for text in texts:
            tn_result = TTSModelInst.only_tn_rhy(text = text)
            print(f'origin_text: {text}')
            print(f'tn_result  : {tn_result[-1]}')
            print(f'*******************************************************************')


        exit()
    if  only_front and front_type == 1:

        ours_pinyins_list = TTSModelInst.get_pinyins(text = text, text_pinyins=text_pinyins)

        if not text_pinyins:
            print(f'origin_text: {text}')
        else:
            print(f'origin_text_pinyins: {text_pinyins}')
        print(f'ours_pinyin: {ours_pinyins_list}')

        # if not text_pinyins:
        #     # f5
        #     f5_output_list = TTSModelInst.zip_origin_convert_text_to_phones(text)
        #     # pdb.set_trace()
        #     output_str = f5_output_list
        #     output_str2 = ''.join(output_str) 
        #     print(f'f5or_pinyin: {f5_output_list}')
        #     # print(output_str2)

        #     if len(ours_pinyins_list) == len(f5_output_list):
        #         print(f'The length is same !')
        #     else:
        #         print(f'[error] The length is Not same ! ours:f5= {len(ours_pinyins_list)} : {len(f5_output_list)}')
        #     err_count = 0
        #     if ours_pinyins_list == f5_output_list:
        #         print(f'check success!')
        #     else:
        #         for idx in range(min(len(ours_pinyins_list), len(f5_output_list))):
        #             pass
        #             if ours_pinyins_list[idx] != f5_output_list[idx]:
        #                 print(f'error,idx = {idx} , {ours_pinyins_list[idx]} != {f5_output_list[idx]} ')
        #                 err_count += 1
        #                 if err_count > 4:
        #                     print(f'more error ')
        #                     exit()

    # elif only_front and front_type == 0:
    #     ours_pinyins_list1 = TTSModelInst.front_model.match_infer(text=text, text_pinyins=None)
    #     print(ours_pinyins_list1)

    #     # exit()
    #     pinyin_compatible_dict = TTSModelInst.front_model.frontend.zh_frontend.pinyin_compatible_dict
    #     poly_dict              = TTSModelInst.front_model.frontend.zh_frontend.poly_dict

    #     import re
    #     def contains_letter_or_digit(s):
    #         pattern = r'[a-zA-Z0-9]'
    #         return bool(re.search(pattern, s))
        
    #     def convert_pinyin_to_ours(pinyin):
    #         pinyin = pinyin.replace('6', '2')
    #         tone = pinyin[-1]
    #         py_wot = pinyin[:-1]
    #         if py_wot in pinyin_compatible_dict.keys():
    #             py_wot =  pinyin_compatible_dict[py_wot]
            
    #         return py_wot+tone

    #     zh_pattern = re.compile("[\u4e00-\u9fa5]")
    #     def is_chinese(word):
    #         global zh_pattern
    #         match = zh_pattern.search(word)
    #         return match is not None

    #     def text_clean(text):
    #         out_text = ''
    #         for sub_word in text:
    #             if is_chinese(sub_word):
    #                 out_text += sub_word
    #         return out_text

    #     PinyinTestFiles = [
    #                 '../standard_pinyin_label/King-TTS-042-24k_LeftCH.txt',
    #                 '../standard_pinyin_label/King-TTS-025-24k_LeftCH.txt',
    #                 '../standard_pinyin_label/King-TTS-099_NewFormat_24k.txt',
    #                 '../standard_pinyin_label/King-TTS-110_NewFormat_24k.txt',
                            
    #                         ]
    #     word_total = 0
    #     err_total  = 0
    #     sentence_total = 0
    #     sentence_err   = 0
    #     poly_total = 0
    #     poly_error = 0
        
    #     for sub_PinyinTestFile in PinyinTestFiles:
    #         with open(sub_PinyinTestFile, encoding="utf-8",) as ttf:
    #             lines = ttf.readlines()
    #             for idx in range(0, len(lines), 2):
    #                 # pdb.set_trace()
    #                 utt_id = lines[idx].strip().split()[0]
    #                 chn_char = lines[idx].strip()[len(utt_id):].strip()
    #                 standard_pinyin_list = lines[idx + 1].strip().split()
    #                 standard_pinyin_list_after_check = []
    #                 for sub_standard_py in standard_pinyin_list:
    #                     tmp_pinyin = convert_pinyin_to_ours(sub_standard_py)
    #                     standard_pinyin_list_after_check.append(tmp_pinyin)

    #                 chn_char = chn_char.replace('#1','').replace('#2','').replace('#3','').replace('#4','')
    #                 chn_char_clean = text_clean(chn_char)
    #                 if contains_letter_or_digit(chn_char):
    #                     continue
    #                 if len(chn_char) < 4:
    #                     continue
    #                 start_time = time.time()
    #                 ours_pinyins_list = TTSModelInst.front_model.match_infer(text=chn_char, text_pinyins=None) # jttts front
    #                 # ours_pinyins_list = TTSModelInst.convert_char_to_pinyin([chn_char]) # f5 front

    #                 if front_type==1:
    #                     ours_pinyins_front1 = []
    #                     for sub_py in ours_pinyins_list:
    #                         if contains_letter_or_digit(sub_py):
    #                             if sub_py[-1].isdigit() is False:
    #                                 sub_py = sub_py + '5'
    #                             ours_pinyins_front1.append(sub_py)
    #                     ours_pinyins_list = ours_pinyins_front1

    #                 end_time = time.time()
    #                 time_cost = round((end_time - start_time)*1000, 2)

    #                 if ali_ttsfrd_inst:
    #                     start_time_ali = time.time()
    #                     ali_prosody, ali_pinyin = ali_ttsfrd_inst.infer(text=chn_char)
    #                     end_time_ali = time.time()
    #                     time_cost_ali = round((end_time_ali - start_time_ali)*1000, 2)
    #                     # print(f'text = {chn_char}, length= {len(chn_char)}, ours_time_cost = {time_cost}ms, ali_time_cost={time_cost_ali}ms')
    #                     ali_pinyin_list = ali_pinyin.split()
    #                     ali_pinyin_list_after_check = []
    #                     for sub_ali_py in ali_pinyin_list:
    #                         tmp_pinyin = convert_pinyin_to_ours(sub_ali_py)
    #                         ali_pinyin_list_after_check.append(tmp_pinyin)

    #                     ours_pinyins_list = ali_pinyin_list_after_check
    #                 # continue
    #                 error_flag = 0
    #                 if len(standard_pinyin_list_after_check) != len(ours_pinyins_list):
    #                     print(f'[error] {utt_id} not equal chn_char={chn_char}')
    #                     standard_pinyin_str = ' '.join(standard_pinyin_list_after_check)
    #                     ours_pinyins_str    = ' '.join(ours_pinyins_list)
    #                     print(f'standard_pinyin_list= {standard_pinyin_str}')
    #                     print(f'ours_pinyins_list   = {ours_pinyins_str}')
    #                 elif len(chn_char_clean) != len(standard_pinyin_list_after_check) and len(chn_char_clean) != len(ours_pinyins_list):
    #                     print(f'[error] {utt_id} word and pinyin not equal chn_char={chn_char}')
    #                     standard_pinyin_str = ' '.join(standard_pinyin_list_after_check)
    #                     ours_pinyins_str    = ' '.join(ours_pinyins_list)
    #                     print(f'standard_pinyin_list= {standard_pinyin_str}')
    #                     print(f'ours_pinyins_list   = {ours_pinyins_str}')
    #                     # pdb.set_trace()
    #                 else:
    #                     word_total += len(ours_pinyins_list)
    #                     sentence_total += 1
    #                     for idy  in  range(len(ours_pinyins_list)):
    #                         if chn_char_clean[idy] in poly_dict:
    #                             poly_total += 1
    #                         if standard_pinyin_list_after_check[idy] != ours_pinyins_list[idy]:
    #                             error_flag = 1
    #                             err_total += 1
    #                             if chn_char_clean[idy] in poly_dict:
    #                                 poly_error += 1
    #                             # pdb.set_trace()
                            
    #                 if error_flag == 0:
    #                     print(f'[success]{idx//2}/{len(lines)//2} {utt_id}  ')
    #                 else:
    #                     print(f'[error] {utt_id} chn_char={chn_char}')
    #                     standard_pinyin_str = ' '.join(standard_pinyin_list_after_check)
    #                     ours_pinyins_str    = ' '.join(ours_pinyins_list)
    #                     print(f'standard_pinyin_list= {standard_pinyin_str}')
    #                     print(f'ours_pinyins_list   = {ours_pinyins_str}')
    #                     sentence_err += 1

    #         print(f'word_total={word_total}, err_total={err_total}, error_rate={err_total/word_total}')
    #         print(f'poly_total={poly_total}, poly_error={poly_error}, error_rate={poly_error/poly_total}')
    #         print(f'sentence_total={sentence_total}, sentence_err={sentence_err}, error_rate={sentence_err/sentence_total}')

    # elif only_front and front_type == 2:
    #     out_segments = TTSModelInst.front_model.only_tn_rhy(text=text)
    #     print(f'origin_text: {text}')
    #     print(f'split_tn   : {out_segments}')
    #     out_str = ''
    #     for segment in out_segments:
    #         for seg in segment:
    #             out_str += seg[0]
    #     print(f'after_tn   : {out_str}')

    else:
        text = '中国目前已开始部署近地小行星防御系统，全球科学家也正以行星防御为纽带展开协作。'
        req = {}
        req['debug_mode']  =   1
        audio_data, sample_rate = TTSModelInst.actual_infer(text = text, user_id='F6', req = req)

        current_millis = int(round(time.time() * 1000))
        timestamp = time.strftime("%m-%d_%H-%M-%S", time.localtime())
        timestamp = timestamp + '-' + str(current_millis)[-3:]
        output_file_new  = 'test_f5_wav' + '-' + timestamp + '.wav'

        sf.write(output_file_new, audio_data, sample_rate, "PCM_16",)


