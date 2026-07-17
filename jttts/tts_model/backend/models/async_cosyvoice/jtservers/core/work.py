# Copyright (c) 2025 JT ChinaMobile. All rights reserved.
# Authors: Ma Yong <mayongyjy@chinamobile.com>.
import time
from datetime import datetime
import queue
import threading
import torch
import torch.multiprocessing as mp
from collections import OrderedDict

from jtservers.core.request import RequestStatus, Request_Single, Request_Batch

import logging as log
log.basicConfig(level=log.INFO,format='[ %(levelname)s %(asctime)s %(filename)s:%(lineno)d ] %(message)s')

class WorkerProcess(mp.Process):
    def __init__(self, process_name, process_idx, worker_process_queue, model_init, model_forward, config_file = None):
        super(WorkerProcess, self).__init__()
        torch.set_num_threads(1)
        
        self.process_name       = process_name
        self.process_idx        = process_idx

        self.worker_process_queue = worker_process_queue
        
        self.model_init     = model_init
        self.model_forward  = model_forward

        self.config_file = config_file

    def run(self):
        model_list = self.model_init(self.process_idx, self.config_file)
        while True:
            # Get a task from the queue
            request = self.worker_process_queue.get()
            # If a special signal is received, the task is finished
            if request is None:
                break
            # Process the task
            self.model_forward(self.process_idx, model_list, request)
            # del request

class Worker(mp.Process):
    def __init__(self, model_init, model_forward, model_num = 3, combine_batch_wait_time = [30, 20, 20], combine_batch_max_size = [4, 2, 2], combine_batch_duration = [768, 896, 1024], model_name = 'Base_Worker', config_file = None):
        super(Worker, self).__init__()
        self.worker_queue          = mp.Queue()
        self.worker_process_queue  = mp.Queue()
        self.worker_response_queue = mp.Queue()
        self.workers = [WorkerProcess(model_name + '_' + str(idx), idx, self.worker_process_queue, model_init, model_forward, config_file) for idx in range(model_num)]

        self.batch_uuid_base = 'Batch_' + model_name + "_"
        self.batch_idx = 0

        self.lock = threading.Lock()
        self.request_single_uuid_mapto_single_queue = OrderedDict.fromkeys([])

        self.run_callback_queue_thread = threading.Thread(target=self.run_callback_queue)

        self.combine_batch_wait_time = combine_batch_wait_time  # List of wait times for each batch level (ms)
        self.combine_batch_max_size = combine_batch_max_size  # List of max batch sizes for each level
        self.combine_batch_duration = combine_batch_duration  # List of durations for each batch level

        self.batch_level_instances = [None] * len(combine_batch_wait_time)  # Holds instances for different batch levels

    def start_workers_process(self):
        for worker in self.workers:
            worker.start()

    def stop_workers_process(self):
        for worker in self.workers:
            self.worker_process_queue.put(None)
        for worker in self.workers:
            worker.join()

    def start_all(self):
        self.start_workers_process()
        self.start()
        self.run_callback_queue_thread.start()

    def stop_all(self):
        self.worker_queue.put(True)
        self.worker_response_queue.put(None)
        self.stop_workers_process()
        self.run_callback_queue_thread.join()

    def add_request(self, request_single: Request_Single, queue_single):
        self.worker_queue.put(request_single)
        with self.lock:
            self.request_single_uuid_mapto_single_queue[request_single.uuid] = queue_single

    def _use_batch_idx(self):
        batch_idx = self.batch_idx
        self.batch_idx = self.batch_idx + 1
        if self.batch_idx > 9999999:
            self.batch_idx = 0 
        return f'{batch_idx:07d}'

    def _create_request_batch_instance(self, batch_level):
        batch_uuid = self.batch_uuid_base + self._use_batch_idx()
        request_batch_instance = Request_Batch(batch_uuid, self.combine_batch_max_size[batch_level], self.combine_batch_duration[batch_level])
        return request_batch_instance

    def _create_batch_tensor_and_send(self, request_batch_instance: Request_Batch):
        # The operation batch_input_tensor_list.append(torch.zeros(...)) is very time-consuming and needs to be moved to request.py
        # {
            # batch_size = request_batch_instance.batch_size
            # for index, max_batch_shape in enumerate(request_batch_instance.max_input_shape):
            #     request_batch_instance.batch_input_tensor_list.append(torch.zeros((batch_size, *request_batch_instance.max_input_shape[index]), dtype=request_batch_instance.max_input_dype[index]))
        # }

        request_batch_instance.run_status = RequestStatus.RUNNING
        self.worker_process_queue.put(request_batch_instance)
        return True

    def run(self):
        # Initialize batch level instances
        for idx in range(len(self.combine_batch_wait_time)):
            self.batch_level_instances[idx] = self._create_request_batch_instance(idx)

        while True:
            try:
                request_single = self.worker_queue.get_nowait()
            except queue.Empty:
                request_single = None
            
            if isinstance(request_single, bool):
                if request_single:
                    break

            if request_single is not None:
                # Determine batch level based on request_single.duration
                batch_level = self._get_batch_level_for_duration(request_single.duration)

                # Add the request to the appropriate batch level instance
                if self.batch_level_instances[batch_level].add_request(request_single):
                    self._create_batch_tensor_and_send(self.batch_level_instances[batch_level])
                    self.batch_level_instances[batch_level] = self._create_request_batch_instance(batch_level)

            if self.worker_queue.empty():
                # Iterate over all batch levels to check for timeout and full batch conditions
                for batch_level, batch_instance in enumerate(self.batch_level_instances):
                    if batch_instance.get_cost_time() > self.combine_batch_wait_time[batch_level] and batch_instance.batch_size > 0:
                        self._create_batch_tensor_and_send(batch_instance)
                        self.batch_level_instances[batch_level] = self._create_request_batch_instance(batch_level)

            if request_single is None:
                time.sleep(0.01)
                 # Check all batch levels for the case where no request is available
                for batch_level, batch_instance in enumerate(self.batch_level_instances):
                    if batch_instance.batch_size == 0:
                        batch_instance.create_request_batch_instance_time = time.time()

    def run_callback_queue(self):
        while True:
            request_batch_instance = self.worker_response_queue.get()
            if request_batch_instance is None:
                break
            for request_single in request_batch_instance.request_single_list:
                queue = self.request_single_uuid_mapto_single_queue[request_single.uuid]
                queue.put(request_single)
                with self.lock:
                    value = self.request_single_uuid_mapto_single_queue.pop(request_single.uuid, None)

    def _get_batch_level_for_duration(self, duration):
        # Determine the batch level based on request_single.duration
        for idx, level_duration in enumerate(self.combine_batch_duration):
            if duration <= level_duration:
                return idx
        return len(self.combine_batch_duration) - 1  # Default to the last level if duration exceeds all levels
