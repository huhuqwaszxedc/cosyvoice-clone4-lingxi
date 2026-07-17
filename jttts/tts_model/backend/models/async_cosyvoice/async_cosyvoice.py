# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import asyncio
import time
from typing import Generator, Union, AsyncGenerator
os.environ["VLLM_USE_V1"] = '1'

import torch
from async_cosyvoice.frontend import CosyVoiceFrontEnd
from async_cosyvoice.model import CosyVoice3Model
# from cosyvoice.utils.file_utils import logging
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
from tqdm import tqdm
from jttts.tts_common.common_utils import  count_elements
from jttts.tts_common.logger import logger 
import pdb
from concurrent.futures import ThreadPoolExecutor
import uuid
import logging
logging.basicConfig(level=logging.INFO, format='[ %(levelname)s %(asctime)s %(filename)s:%(lineno)d ] %(message)s')

class AsyncCosyVoice3:
    def __init__(self, model_dir, load_jit=False, load_trt=False, fp16=False, flow_worker_stream=None, flow_worker_offline=None):
        self.instruct = True if '-Instruct' in model_dir else False
        self.model_dir = model_dir
        self.fp16 = fp16
        # pdb.set_trace()
        if not os.path.exists(model_dir):
            logging.error(f'{model_dir} is not exists! please check!')
            exit()
        hyper_yaml_path = '{}/cosyvoice3.yaml'.format(model_dir)

        override_dict = {k: None for k in ['llm']}
        with open(hyper_yaml_path, 'r') as f:
            configs = load_hyperpyyaml(f, overrides={**override_dict, 'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})

        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v3.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),   # 不会加载这个pt里面的用户信息
                                          configs['allowed_special'])
        self.sample_rate = configs['sample_rate']
        if torch.cuda.is_available() is False and (load_jit is True or load_trt is True or fp16 is True):
            load_jit, load_trt, fp16 = False, False, False
            logging.warning('no cuda device, set load_jit/load_trt/fp16 to False')
        self.model = CosyVoice3Model(
            model_dir,
            configs['flow'],
            configs['hift'],
            fp16,
            flow_worker_stream=flow_worker_stream,
            flow_worker_offline=flow_worker_offline
        )
        self.model.load(
            '{}/flow.pt'.format(model_dir),
            '{}/hift.pt'.format(model_dir),
        )
        # if load_jit:
        #     self.model.load_jit('{}/flow.encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'))
        # if load_trt:
        #     self.model.load_onnx_and_convert_trt('{}/flow.decoder.estimator.{}.mygpu.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
        #                         '{}/flow.decoder.estimator.fp32.onnx'.format(model_dir),
        #                         self.fp16)

        if load_trt:
            self.model.load_trt('{}/flow.decoder.estimator.{}.h100.my.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),)

        self.thread_count = 4
        self.thread_executor = ThreadPoolExecutor(max_workers=self.thread_count)

        del configs

    def list_available_spks(self):
        spks = list(self.frontend.spk2info.keys())
        return spks

    async def add_spk_info(self, spk_id, spk_info):
        self.frontend.add_spk_info(spk_id, spk_info)

    async def inference_sft(self, tts_text, spk_id, stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_sft(i, spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            async for model_output in self.model.async_tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}, time cost: {}'.format(speech_len, (time.time() - start_time) / speech_len, (time.time() - start_time)))
                yield model_output
                start_time = time.time()

    async def inference_cross_lingual(self, tts_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_cross_lingual(i, prompt_speech_16k, self.sample_rate)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            async for model_output in self.model.async_tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    async def inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True):
        prompt_text = self.frontend.text_normalize(prompt_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            if (not isinstance(i, Union[Generator, AsyncGenerator])) and len(i) < 0.5 * len(prompt_text):
                logging.warning('synthesis text {} too short than prompt text {}, this may lead to bad performance'.format(i, prompt_text))
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_speech_16k, self.sample_rate)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            async for model_output in self.model.async_tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    async def inference_instruct2(self, tts_text, instruct_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_instruct2(i, instruct_text, prompt_speech_16k, self.sample_rate)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            async for model_output in self.model.async_tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    async def inference_instruct2_by_spk_id(self, tts_text, instruct_text, spk_id, stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_instruct2_by_spk_id(i, instruct_text, spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            async for model_output in self.model.async_tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    async def inference_zero_shot_by_spk_id(self, tts_text, spk_id, stream=False, speed=1.0, text_frontend=True, req= {}):
        """使用预定义的说话人执行 zero_shot 推理"""

        text_splits = self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)
        chunk_actual_index = 0
        for idx, sub_text in enumerate(text_splits) :
        # if True:
        #     # i = tts_text
            # sub_text = self.frontend.text_normalize(tts_text, split=False, text_frontend=text_frontend)

            english_count, chinese_count, digit_count = count_elements(sub_text)
            if (english_count + chinese_count + digit_count) > 80:
                logger.info(f'The sentense split out length >80, {sub_text}')
                yield {'tts_speech': '', 'tts_status':-2, 'chunk_index': -2, 'speech_token': ''}


            # model_input = self.frontend.frontend_zero_shot_by_spk_id(sub_text, spk_id)

            loop = asyncio.get_event_loop()
            model_input = await loop.run_in_executor(self.thread_executor,
                                                        self.frontend.frontend_zero_shot_by_spk_id,
                                                        sub_text,
                                                        spk_id
                                                        )
            if len(model_input) == 0:
                yield {'tts_speech': '', 'tts_status':-2, 'chunk_index': -2, 'speech_token': ''}

            start_time = time.time()
            last_time = start_time
            
            # logging.info('synthesis text {}'.format(i))
            async for model_output in self.model.async_tts(**model_input, stream=stream, speed=speed):
                if model_output['tts_status'] == 0:
                    # speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                    # logging.info('yield speech index:{}, len {:.2f}, rtf {:.3f},  cost {:.3f}s,  all cost time {:.3f}s'.format(chunk_index, speech_len,  (time.time()-last_time)/speech_len, time.time()-last_time, time.time()-start_time))
                    if idx == len(text_splits)-1 and model_output['chunk_index'] == -1:
                        model_output['chunk_index'] = -1
                    else:
                        model_output['chunk_index'] = chunk_actual_index
                    yield model_output
                    last_time = time.time()
                    chunk_actual_index += 1
                else: # model_output['chunk_index'] == -2:
                    logger.info(f"[error] Voice_Clone TextSplit part:{sub_text}, sequence=-2 and skip tts !", is_encrypt=True)
                    yield {'tts_speech': '', 'tts_status':-2, 'chunk_index': -2, 'speech_token': ''}



    async def inference_llm_token_by_spk_id(self, llm_tokens, spk_id, req= {}):
        """使用llm_token + spk_id 输出音频"""


        sub_text = '今天'
        loop = asyncio.get_running_loop()
        model_input = await loop.run_in_executor(self.thread_executor,
                                                    self.frontend.frontend_zero_shot_by_spk_id,
                                                    sub_text,
                                                    spk_id
                                                    )
        if len(model_input) == 0:
            return {'tts_speech': '', 'tts_status':-2, 'chunk_index': -2, 'speech_token': ''}

        this_tts_speech_token = torch.tensor(llm_tokens).unsqueeze(dim=0)
        this_tts_speech_token_len = torch.tensor([this_tts_speech_token.size(1)], dtype=torch.int32)
        flow_prompt_speech_token = model_input['flow_prompt_speech_token']
        prompt_speech_feat  = model_input['prompt_speech_feat']
        flow_embedding    = model_input['flow_embedding']
        this_uuid = str(uuid.uuid1())
        token_offset = 0
        chunk_index = -1

        this_tts_speech = await loop.run_in_executor(self.thread_executor,
                                                    self.model.token2wav,
                                                    this_tts_speech_token,
                                                    flow_prompt_speech_token,
                                                    prompt_speech_feat,
                                                    flow_embedding,
                                                    this_uuid,
                                                    0,
                                                    False,
                                                    True,
                                                    1.0,
                                                    this_tts_speech_token_len,
                                                    chunk_index
                                                    )

        return {'tts_speech': this_tts_speech.cpu(), 'tts_status': 0, 'chunk_index': chunk_index, 'speech_token': this_tts_speech_token.cpu()}


