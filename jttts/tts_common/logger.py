import logging
from logging.handlers import TimedRotatingFileHandler
import os
import time
import socket
from jttts.tts_common.content_encoder import encrypt_and_encode, encryption_label
from datetime import datetime
from jttts.config import GlobalConfigInst

if os.path.exists('/tts_deploy_info/log_info/log-LAPTOP-0963S9S1'):
    os.remove('/tts_deploy_info/log_info/log-LAPTOP-0963S9S1')


class Singleton(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            orig = super(Singleton, cls)
            cls._instance = orig.__new__(cls) #, *args, **kwargs
        return cls._instance



class SafeTimedRotatingFileHandler(TimedRotatingFileHandler):
    """
    支持在日志文件被外部删除后自动重建的 Handler。
    每次写入前检查文件是否存在，若不存在则重新创建。
    """

    def emit(self, record):
        """
        Emit a record.
        If the file has been removed, reopen it before writing.
        """
        try:
            # 检查当前文件是否还存在
            if not os.path.exists(self.baseFilename):
                self.stream.close()
                self.stream = self._open()  # 重新打开文件（会自动创建）
        except Exception:
            # 如果检查或重开失败，至少不要中断原 emit
            pass

        # 调用父类的 emit 进行实际写入
        super().emit(record)



class Logger(logging.Logger, Singleton):
    def __init__(self, log_dir='./', subfix=''):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True

        super(Logger, self).__init__("")

        # # 生成时间戳，格式：YYYYMMDD_HHMM
        # timestamp = datetime.now().strftime("%Y%m%d%H%M")
        # hostname = socket.gethostname()
        # filename = os.path.join(GlobalConfigInst.external_log_dir, f"log_v6.1_{hostname}_{timestamp}")


        # ✅ 优先从环境变量读取统一的日志文件名
        env_filename = os.getenv('LOGGER_FILENAME')
        if env_filename:
            filename = os.path.join(log_dir, env_filename)
            if len(subfix) > 0:
                filename = filename[:-4] + f'{subfix}.log'
        else:
            hostname = socket.gethostname()
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            log_filename = f"log_{GlobalConfigInst.version}_{hostname}_{ts}{subfix}.log"
            filename = os.path.join(log_dir, log_filename)



        self.filename = filename

        # 确保目录存在
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)

        # 使用自定义的安全 Handler
        fh = SafeTimedRotatingFileHandler(
            self.filename,
            when='MIDNIGHT',
            interval=1,
            backupCount=10000,
            encoding='utf-8'
        )
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        self.addHandler(fh)

        self.parent_loginfo = super(Logger, self).info
        self.parent_logerror = super(Logger, self).error

    def info(self, content:str, is_encrypt=False):
        if GlobalConfigInst.log_mode == 'Normal':
            pass
        else:
            if is_encrypt:
                content = encrypt_and_encode(content)
                content = encryption_label + content
        
        self.parent_loginfo(f'{content}')

    def error(self, content:str, is_encrypt=False):
        if GlobalConfigInst.log_mode == 'Normal':
            pass
        else:
            if is_encrypt:
                content = encrypt_and_encode(content)
                content = encryption_label + content
        self.parent_logerror(f'{content}')


logger = Logger(log_dir=GlobalConfigInst.external_log_dir)

# *************************************************************************************************************************

class Singleton2(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            orig = super(Singleton2, cls)
            cls._instance = orig.__new__(cls) #, *args, **kwargs
        return cls._instance


class Logger2(logging.Logger, Singleton2):
    def __init__(self, log_dir='./', subfix=''):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True

        super(Logger2, self).__init__("")

        # ✅ 优先从环境变量读取统一的日志文件名
        env_filename = os.getenv('LOGGER_FILENAME')
        if env_filename:
            filename = os.path.join(log_dir, env_filename)
            if len(subfix) > 0:
                filename = filename[:-4] + f'{subfix}.log'
        else:
            hostname = socket.gethostname()
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            log_filename = f"log_{GlobalConfigInst.version}_{hostname}_{ts}{subfix}.log"
            filename = os.path.join(log_dir, log_filename)



        self.filename = filename

        # 确保目录存在
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)

        # 使用自定义的安全 Handler
        fh = SafeTimedRotatingFileHandler(
            self.filename,
            when='MIDNIGHT',
            interval=1,
            backupCount=10000,
            encoding='utf-8'
        )
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        self.addHandler(fh)

        self.parent_loginfo = super(Logger2, self).info
        self.parent_logerror = super(Logger2, self).error

    def info(self, content:str, is_encrypt=False):
        self.parent_loginfo(f'{content}')

    def error(self, content:str, is_encrypt=False):
        self.parent_logerror(f'{content}')


moreinfo_log_dir = os.path.join(GlobalConfigInst.external_log_dir, 'moreinfo')
os.makedirs(moreinfo_log_dir, exist_ok=True)
logger_debug = Logger2(log_dir=moreinfo_log_dir, subfix='-moreinfo')

