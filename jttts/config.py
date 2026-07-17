import sys,os, json

def get_code_version(version_file):
    hash_info = ''
    if os.path.exists(version_file):
        try:
            with open(version_file, 'r', encoding="utf-8") as fjson:
                version_info = json.load(fjson)
            
            hash_info = version_info.get('hash', '')
        except:
            hash_info = ''
    return hash_info    


class GlobalConfig:
    def __init__(self):
        self.workers  = 1
        self.http_port      = 7820
        self.websocket_port = 8820
        self.grpc_port      = 9820

        self.check_port = self.http_port + 1
        self.interface_type = 2 # 1--v1.0, 2--v2.0

        self.container_root_dir = '/tts_deploy_info'
        os.makedirs(self.container_root_dir, exist_ok=True)

        self.pretrained_model_rootdir = os.path.join(self.container_root_dir, 'pretrained_models_clone6')

        if os.path.exists(self.pretrained_model_rootdir) is False:
            self.container_root_dir = '../'
            self.pretrained_model_rootdir = os.path.join(self.container_root_dir, 'pretrained_models_clone6')
            print(f'use local container root dir')

        model_config_json_file = os.path.join(self.pretrained_model_rootdir, 'model_config.json')
        if os.path.exists(model_config_json_file):
            with open(model_config_json_file, 'r', encoding="utf-8") as fjson:
                model_info = json.load(fjson)
                backend_acoustic_model   = model_info["backend_acoustic_model"]
                backend_vocoder_model   = model_info["backend_vocoder_model"]
                front_model     = model_info["front_model"]
                self.backend_acoustic_model = os.path.join(self.pretrained_model_rootdir,  backend_acoustic_model)
                self.backend_vocoder_model  = os.path.join(self.pretrained_model_rootdir,  backend_vocoder_model)
                self.front_init_dir         = os.path.join(self.pretrained_model_rootdir,  front_model)

                # 运行模式，0--正常，1--测试
                if 'run_mode' in model_info:
                    self.run_mode = int(model_info['run_mode'])
                else:
                    self.run_mode = 0


                self.tts_gpu_num = 1
                self.infer_speed_mode = 1


        else:
            print(f'The model_config.json is not exists, please check !')
            exit()

        # speaker info
        self.build_in_speaker_info_dir = './build_in_speaker'
        self.speaker_info_dir     = os.path.join(self.container_root_dir, 'speaker_info')
        os.makedirs(self.speaker_info_dir, exist_ok=True)
        os.makedirs(os.path.join(self.speaker_info_dir, "prompt_wavs"), exist_ok=True)

        # eval speaker info 用于测试
        self.eval_speaker_info_dir     = os.path.join(self.container_root_dir, 'eval_speaker_info')

        # license info
        self.external_license_dir = os.path.join(self.container_root_dir, 'license_info')
        os.makedirs(self.external_license_dir, exist_ok=True)

        # inference log info
        self.external_log_dir = os.path.join(self.container_root_dir, 'log_info')
        os.makedirs(self.external_log_dir, exist_ok=True)


        # prompt quality check
        self.prompt_quality_check = True   # only for asr check
        self.prompt_cer_threshold = 0.08

        self.prompt_denoise     = True
        self.prompt_remove_sil  = True

        if self.prompt_quality_check is False:
            self.prompt_denoise     = False

        # default sample rate
        if self.interface_type == 2:
            self.default_sample_rate = 24000 
            self.default_speed       = float(1.0)
        elif self.interface_type == 1:
            self.default_sample_rate = 32000
            self.default_speed       = int(5)
        else:
            self.default_sample_rate = 24000
            self.default_speed       = float(1.0)

        # default return audio type
        self.default_encoding = 'raw'  # 'raw' or 'wav'

        self.max_tts_concurrent = 12
        self.log_mode = 'Normal' 
        self.save_middle_result = 1  # 0--不保存， 1--保存 ; for debug
        hash_info = get_code_version('./code_version.json')
        if len(hash_info) > 0:
            self.version  = f'v6.1_{hash_info}'
        else:
            self.version  = 'v6.1'
        


GlobalConfigInst = GlobalConfig()
