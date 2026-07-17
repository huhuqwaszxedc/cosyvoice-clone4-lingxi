import os
from datetime import datetime
import socket
from jttts.config import GlobalConfigInst
import argparse
import multiprocessing
import random

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-po', '--port_offset', type=int, required=False, default=0, help='port num offset')
    args = parser.parse_args()
    return args


if 'LOGGER_FILENAME' not in os.environ:
    hostname = socket.gethostname()
    ts = datetime.now().strftime("%Y%m%d_%H%M")  # 只在主进程生成一次
    log_filename = f"log_{GlobalConfigInst.version}_{hostname}_{ts}.log"
    os.environ['LOGGER_FILENAME'] = log_filename
else:
    log_filename = os.environ['LOGGER_FILENAME']


import sys
import signal
import asyncio
import logging
from pathlib import Path
import subprocess
import time
import uvicorn
from jttts.tts_common.check_aut import check_license_process
from jttts.tts_common.logger import logger

# import logging

ftp_proxy_value = os.getenv('ftp_proxy', None)
https_proxy_value = os.getenv('https_proxy', None)
http_proxy_value = os.getenv('http_proxy', None)

if ftp_proxy_value:
    del os.environ['ftp_proxy']
if https_proxy_value:
    del os.environ['https_proxy']
if http_proxy_value:
    del os.environ['http_proxy']

# main py file path
py_file = os.path.abspath(os.path.abspath(__file__))
logger.info("py main file is {}".format(py_file))

# para setting
external_license_file = os.path.join(GlobalConfigInst.external_license_dir, 'tts_licensev2')# 宿主机映射到本机的license
internal_file_path = './tts_licensev2' # 本地的license

license_file_path = ''
if os.path.exists(external_license_file):
    license_file_path = external_license_file
elif os.path.exists(internal_file_path):
    license_file_path = internal_file_path
else:
    print(f'[error] {license_file_path} is not exists !, please check')
    logger.info("license_file is not exists!")
    exit()

logger.info("license_file is {}".format(license_file_path))


def run_license_check_process(license_path):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    while True:
        flag = check_license_process(license_path)
        if flag:
            time.sleep(60*60*2)
        else:
            cmd = "ps -aux|grep python|awk '{print $2}'|xargs -i kill -9 {}"
            val = os.system(cmd)
            break

# 添加路径
current_file_dir = os.path.dirname(__file__)
sys.path.append(current_file_dir)
sys.path.append(os.path.join(current_file_dir, 'jttts/tts_model/backend/models'))
sys.path.append(os.path.join(current_file_dir, 'jttts/tts_model/backend/models/async_cosyvoice'))
sys.path.append(os.path.join(current_file_dir, 'jttts/tts_engine'))
sys.path.append(os.path.join(current_file_dir, 'jttts/tts_model/backend/models/third_party/Matcha-TTS'))

from jtservers.core.model import BaseModel_WorkerConfig, BaseModelWorker
from async_cosyvoice.flowworker import CosyFlowMatchingWorker
from async_cosyvoice.async_cosyvoice import AsyncCosyVoice3

# 导入 HTTP App
from jttts.http_server import APP as http_app, lifespan

# 导入 gRPC 服务
import grpc
from concurrent import futures
import tts_interface_pb2_grpc
from jttts.grpc_server import GrpcServiceImpl

# === 导入 websocket_server ===
import jttts.websocket_server as websocket_server


# 全局变量
flow_worker_stream = None
flow_worker_offline = None
flow_worker_stream0 = None
TTS_MODEL = None


async def init_model():
    global flow_worker_stream, flow_worker_offline, flow_worker_stream0, TTS_MODEL
    model_dir = GlobalConfigInst.backend_acoustic_model

    if GlobalConfigInst.infer_speed_mode == 0:
        # 初始化流式/离线 worker
        config0 = BaseModel_WorkerConfig()
        config0.model_name = 'CosyFlowMatchingStream0'
        config0.model_num = 1
        config0.combine_batch_wait_time = [40]
        config0.combine_batch_max_size = [24]
        config0.combine_batch_duration = [43]
        flow_worker_stream0 = CosyFlowMatchingWorker(config0, model_dir)
        flow_worker_stream0.start()

        config1 = BaseModel_WorkerConfig()
        config1.model_name = 'CosyFlowMatchingStream1'
        config1.model_num = 1
        config1.combine_batch_wait_time = [40, 30, 30, 30]
        config1.combine_batch_max_size = [24, 24, 16, 8]
        config1.combine_batch_duration = [83, 123, 163, 203]
        flow_worker_stream = CosyFlowMatchingWorker(config1, model_dir)
        flow_worker_stream.start()

        config2 = BaseModel_WorkerConfig()
        config2.model_name = 'CosyFlowMatchingOffline'
        config2.model_num = 1
        config2.combine_batch_wait_time = [50, 50, 50, 20, 10]
        config2.combine_batch_max_size = [24, 24, 8, 4, 2]
        config2.combine_batch_duration = [128, 256, 512, 1024, 1536]
        flow_worker_offline = CosyFlowMatchingWorker(config2, model_dir)
        flow_worker_offline.start()
        logger.info("Flow workers started (speed mode enabled)")
    else:
        logger.info("Flow speed mode disabled")

    # 初始化 TTS 模型
    class Args:
        def __init__(self):
            self.model_dir = model_dir
            self.load_jit = False
            self.load_trt = True 
            self.fp16 = True

    args = Args()
    TTS_MODEL = AsyncCosyVoice3(
        args.model_dir,
        load_jit=args.load_jit,
        load_trt=args.load_trt,
        fp16=args.fp16,
        flow_worker_stream=[flow_worker_stream0, flow_worker_stream],
        flow_worker_offline=flow_worker_offline
    )

    # 注入模型到 HTTP app
    http_app.TTS = TTS_MODEL

    # Warmup
    from jttts.http_server import Upload_Prompt_http_Warmup, Voice_Clone_http_Warmup
    await Upload_Prompt_http_Warmup(http_app)
    await Voice_Clone_http_Warmup(http_app)
    logger.info("Model warmup completed")


async def start_grpc_server( port, inference_semaphore=''):
    options = [
        ('grpc.max_send_message_length', 100 * 1024 * 1024),
        ('grpc.max_receive_message_length', 100 * 1024 * 1024),
    ]
    server = grpc.aio.server(
        migration_thread_pool=futures.ThreadPoolExecutor(max_workers=32),
        options=options,
        maximum_concurrent_rpcs=100
    )
    servicer = GrpcServiceImpl(inference_semaphore=inference_semaphore)
    servicer.tts_model = TTS_MODEL
    logger.info("gRPC servicer initialized with shared model")

    tts_interface_pb2_grpc.add_cloneServicer_to_server(servicer, server)
    server.add_insecure_port(f'[::]:{port}')
    await server.start()
    logger.info(f"gRPC server listening on port {port}")

    async def shutdown():
        await server.stop(5)

    return shutdown


async def start_http_server(port):
    host = "0.0.0.0"

    config = uvicorn.Config(
        app=http_app,
        host=host,
        port=port,
        workers=1,  # 必须为1，否则无法共享模型
        log_level="info",
        reload=False,
    )
    server = uvicorn.Server(config)

    # 启动服务器（非阻塞）
    await server.serve()
    logger.info(f"HTTP server stopped")

# === 新增：启动 WebSocket 服务 ===
async def start_websocket_server(port):
    host = "0.0.0.0"
    config = uvicorn.Config(
        app=websocket_server.app,
        host=host,
        port=port,
        log_level="info",
        access_log=False, 
        workers=1,  # 必须为1
    )
    server = uvicorn.Server(config)
    # 注入模型（关键！）
    websocket_server.app.state.TTS_MODEL = TTS_MODEL
    
    logger.info(f"WebSocket server starting on ws://{host}:{port}")
    await server.serve()
    logger.info(f"WebSocket server stopped")



async def main(port_offset=0):

    http_port      = GlobalConfigInst.http_port         + port_offset
    websocket_port = GlobalConfigInst.websocket_port    + port_offset
    grpc_port      = GlobalConfigInst.grpc_port         + port_offset

    initial_delay = random.uniform(0.0, 180.0) 
    await asyncio.sleep(initial_delay) # 避免在k8s启动时，多个pod同时启动对磁盘的资源竞争

    # --- Start License Checker Process ---
    
    # Create the license checker process
    license_checker_proc = multiprocessing.Process(
        target=run_license_check_process,
        args=(license_file_path,),
        name="LicenseCheckerProcess"
    )
    license_checker_proc.start()
    logger.info(f"Started license checker process with PID: {license_checker_proc.pid}")


    # ===== 新增：启动 register_service =====
    register_proc = None
    if GlobalConfigInst.prompt_quality_check:
        script_name = './register_service.pyc' if os.path.exists('./register_service.pyc') else './register_service.py'
        try:
            # 使用 subprocess.Popen 启动（非阻塞）
            register_proc = subprocess.Popen(
                [sys.executable, script_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            logger.info(f"Started register_service: PID={register_proc.pid}")
            # 等待服务启动（30秒是原逻辑）
            time.sleep(30)
        except Exception as e:
            logger.error(f"Failed to start register_service: {e}")
            if register_proc:
                register_proc.terminate()
            exit(1)
    # ===================================

    # 初始化模型（在 lifespan 之前手动调用）
    await init_model()

    # >>>>>>>>>>> 新增：创建全局推理并发控制信号量 <<<<<<<<<<<<<
    max_concurrent = GlobalConfigInst.max_tts_concurrent  
    inference_semaphore = asyncio.Semaphore(max_concurrent)
    
    # 注入到 HTTP app
    http_app.state.inference_semaphore = inference_semaphore
    # 注入到 WebSocket app
    websocket_server.app.state.inference_semaphore = inference_semaphore


    # 启动GRPC
    grpc_shutdown = await start_grpc_server(port=grpc_port, inference_semaphore=inference_semaphore)

    # 并发运行 HTTP 和等待终止
    http_task = asyncio.create_task(start_http_server(port=http_port))
    ws_task = asyncio.create_task(start_websocket_server(port=websocket_port))

    try:
        # 等待任一服务异常退出
        done, pending = await asyncio.wait(
            [http_task, ws_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, shutting down...")
    finally:
        # 关闭服务
        if register_proc:
            register_proc.terminate()
            register_proc.wait(timeout=10)

        await grpc_shutdown()

        # 停止 workers
        global flow_worker_stream0, flow_worker_stream, flow_worker_offline
        for worker in [flow_worker_stream0, flow_worker_stream, flow_worker_offline]:
            if worker:
                worker.stop()

        logger.info("All services shut down gracefully")



if __name__ == "__main__":
    args = get_args()
    port_offset = args.port_offset
    asyncio.run(main(port_offset))