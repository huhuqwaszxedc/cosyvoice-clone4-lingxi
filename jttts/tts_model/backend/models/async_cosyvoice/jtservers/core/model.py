# Copyright (c) 2025 JT ChinaMobile. All rights reserved.
# Authors: Ma Yong <mayongyjy@chinamobile.com>.

import queue
import numpy as np
import torch
import torch.multiprocessing as mp
import uuid
import yaml
from collections import OrderedDict

from jtservers.core.work import Worker
from jtservers.core.request import RequestStatus, Request_Single, Request_Batch

import logging as log
log.basicConfig(level=log.INFO,format='[ %(levelname)s %(asctime)s %(filename)s:%(lineno)d ] %(message)s')

class BaseModel_WorkerConfig():
    def __init__(self):
        self.model_name = "BaseModel"
        self.model_num = 2
        self.combine_batch_wait_time = [30, 20, 20]
        self.combine_batch_max_size = [4, 2, 2]
        self.combine_batch_duration = [768, 896, 1024]

class BaseModelWorker():
    def __init__(self, basemodel_workerconfig, config_file):
        super(BaseModelWorker, self).__init__()
        
        self.model_name = basemodel_workerconfig.model_name
        self.model_num = basemodel_workerconfig.model_num
        self.combine_batch_wait_time = basemodel_workerconfig.combine_batch_wait_time
        self.combine_batch_max_size  = basemodel_workerconfig.combine_batch_max_size
        self.combine_batch_duration  = basemodel_workerconfig.combine_batch_duration

        # Ensure that the three lists have the same size
        if not (len(self.combine_batch_wait_time) == len(self.combine_batch_max_size) == len(self.combine_batch_duration)):
            raise ValueError("The sizes of combine_batch_wait_time, combine_batch_max_size, and combine_batch_duration must be equal.")
        
        self.worker = Worker(self.model_init, self.model_forward, self.model_num, self.combine_batch_wait_time, self.combine_batch_max_size, self.combine_batch_duration, self.model_name, config_file)

        self.request_single_uuid_mapto_single_queue = OrderedDict.fromkeys([])

        self.use_cuda_idx = 0

    def model_init(self, process_idx, config_file):
        log.info(f'[BaseModelWorker] model_init process_idx: {process_idx} config_file: {config_file}')
        model_list = []
        return model_list

    def model_forward(self, process_idx, model_list, request_batch : Request_Batch):
        time_from_create_to_forwart = request_batch.get_cost_time()
        batch_input_tensor = request_batch.get_batch_data()
        time_copy_batch_data = request_batch.get_cost_time() - time_from_create_to_forwart
        # log.info(f'[{self.model_name}:{self.use_cuda_idx}] request_batch: {request_batch}. time_from_create_to_forward: {time_from_create_to_forwart} ms time_copy_batch_data: {time_copy_batch_data} ms.')
        output_tensor_list = self.forward(process_idx, model_list, request_batch, batch_input_tensor)
        request_batch.run_status = RequestStatus.FINISHED_FINISHED
        request_batch.set_result(output_tensor_list)
        self.finished(request_batch)
        # log.info(f'[{self.model_name}:{self.use_cuda_idx}] request_batch: {request_batch}. Total cost {request_batch.get_cost_time()} ms.')

    def forward(self, process_idx, model_list, request_batch, batch_input_tensor):
        output_tensor_list = []
        return output_tensor_list

    def add_request(self, request_single:Request_Single):
        if isinstance(request_single, Request_Single):
            request_queue = mp.Queue()
            self.request_single_uuid_mapto_single_queue[request_single.uuid] = request_queue
            self.worker.add_request(request_single, request_queue)
            return True
        else:
            log.error(f'Not Get Request_Single')
            return False

    def wait_complete(self, request_single:Request_Single):
        request_queue = self.request_single_uuid_mapto_single_queue[request_single.uuid]
        value = self.request_single_uuid_mapto_single_queue.pop(request_single.uuid, None)
        result_request = request_queue.get()
        return result_request

    def start(self):
        self.worker.start_all()

    def stop(self):
        self.worker.stop_all()

    def finished(self, request_batch : Request_Batch):
        self.worker.worker_response_queue.put(request_batch)
