#!/usr/bin/python
# -*- encoding: utf-8 -*-


import os

from jttts.config import GlobalConfigInst


current_dir = os.path.dirname(os.path.abspath(__file__))
resource_path = os.path.join(current_dir, 'resource')

# base_path = os.getcwd()
if os.path.exists(GlobalConfigInst.front_init_dir):
    base_path = GlobalConfigInst.front_init_dir
elif os.path.exists(resource_path):
    base_path = resource_path
else:
    base_path = '/mnt/d/work/git_code/pretrained_models/tts_resource/front_model'


phones_dict = None
tones_dict = None
wave_path = os.path.join(os.getcwd(),"init.wav")

PHONE_TYPE = ""
PHONE_NATION = ""
DEBUG = False
DEBUG_TIME = False

enc_suffix = 'enc'

with_erhua = False