# Copyright (c) 2025 JT ChinaMobile. All rights reserved.
# Authors: Ma Yong <mayongyjy@chinamobile.com>.

import math
import enum
import time
import torch
from typing import Dict, List, Optional, Union
from collections import OrderedDict

from jtservers.core.colorslog import ColorsLog

import logging as log
log.basicConfig(level=log.INFO,format='[ %(levelname)s %(asctime)s %(filename)s:%(lineno)d ] %(message)s')

class RequestStatus(enum.Enum):
    WAITING = enum.auto()
    RUNNING = enum.auto()
    FINISHED_FINISHED = enum.auto()
    FINISHED_ABORTED = enum.auto()

    @staticmethod
    def is_finished(status) -> bool:
        return status in [
            RequestStatus.FINISHED_FINISHED,
            RequestStatus.FINISHED_ABORTED,
        ]

    @staticmethod
    def get_status_str(status) -> Union[str, None]:
        if status == RequestStatus.WAITING:
            status_str = "waiting"
        elif status == RequestStatus.RUNNING:
            status_str = "running"
        elif RequestStatus.is_finished(status):
            status_str = "finished"
        else:
            status_str = None
        return status_str

    @staticmethod
    def get_finished_reason(status) -> Union[str, None]:
        if status == RequestStatus.FINISHED_FINISHED:
            finish_reason = "stop"
        elif status == RequestStatus.FINISHED_ABORTED:
            finish_reason = "abort"
        else:
            finish_reason = None
        return finish_reason

class Request:
    def __init__(self, uuid):
        self.name = self.__class__.__name__
        self.uuid = self.name + '_' + uuid
        
        self.run_status = RequestStatus.WAITING
        self.run_status_message = ''

        self.duration = 0

    def __str__(self):
        return f'{ColorsLog.YELLOW}address{ColorsLog.RESET}: {id(self)} {ColorsLog.YELLOW}uuid{ColorsLog.RESET}: {self.uuid}, {ColorsLog.YELLOW}run_status{ColorsLog.RESET}: {self.run_status}, {ColorsLog.YELLOW}run_status_message{ColorsLog.RESET}: {self.run_status_message}'

    def set_duration(self, duration):
        self.duration = duration


class Request_Single(Request):
    def __init__(self, uuid):
        super().__init__(uuid)
        # self.name = self.__class__.__name__
        # self.uuid = self.name + '_' + uuid

        self.input_tensor_list = []
        self.output_tensor_list = []

        self.input_shape_list   = []
        self.input_dtype_list   = []

    def set_input_tensor_list(self, input_tensor_list):
        self.input_tensor_list = input_tensor_list
        for index, input_tensor in enumerate(input_tensor_list):
            self.input_shape_list.append(list(input_tensor.shape))
            self.input_dtype_list.append(input_tensor.dtype)

        return True

    def get_batch_result(self):
        return self.output_tensor_list
        
    def __str__(self):
        parent_info = super().__str__()
        return f'--> [ {ColorsLog.YELLOW}name{ColorsLog.RESET}: {self.name} {parent_info} {ColorsLog.YELLOW}input_shape_list{ColorsLog.RESET}: {self.input_shape_list} {ColorsLog.YELLOW}input_dtype_list{ColorsLog.RESET}: {self.input_dtype_list} ]'


class Request_Batch(Request):
    def __init__(self, uuid, max_batch_size = 8, max_duration = 512):
        super().__init__(uuid)
        # self.name = self.__class__.__name__
        # self.uuid = self.name + '_' + uuid

        self.max_batch_size = max_batch_size
        self.max_duration = max_duration
        self.batch_size = 0
        self.request_single_list = []

        self.batch_input_tensor_list = []

        self.max_input_shape = []
        self.max_input_dype = []

        self.max_input_shape_mul = []

        self.create_request_batch_instance_time = time.time()

    def add_request(self, request_subsingle):
        self.request_single_list.append(request_subsingle)
        self.batch_size = len(self.request_single_list)
        
        for index, input_shape in enumerate(request_subsingle.input_shape_list):
            if len(self.max_input_shape_mul) == index:
                self.max_input_shape_mul.append(0)
            if self.max_input_shape_mul[index] < math.prod(input_shape):
                self.max_input_shape_mul[index] = math.prod(input_shape)
                if len(self.max_input_shape) == index:
                    self.max_input_shape.append([])
                    self.max_input_dype.append(torch.float32)
                self.max_input_shape[index] = input_shape
                self.max_input_dype[index] = request_subsingle.input_dtype_list[index]

        request_subsingle.run_status = RequestStatus.RUNNING
        if self.batch_size == self.max_batch_size:
            return True
        return False

    def get_batch_data(self):
        # for index_input, input_tensor in enumerate(self.batch_input_tensor_list):
        #     log.info(f'uuid: {self.uuid} index_input: {index_input} input_tensor: {input_tensor.size()}')
        
        batch_size = self.batch_size
        for index, max_batch_shape in enumerate(self.max_input_shape):
            self.batch_input_tensor_list.append(torch.zeros((batch_size, *self.max_input_shape[index]), dtype=self.max_input_dype[index]))

        for batch_index, request_single in enumerate(self.request_single_list):
            for index, input_tensor in enumerate(request_single.input_tensor_list):
                shape_input_tensor = input_tensor.size()
                index_tensor = []
                index_tensor.append(batch_index)
                for dim in shape_input_tensor:
                    index_tensor.append(slice(0, dim))
                self.batch_input_tensor_list[index][index_tensor] = input_tensor
        return self.batch_input_tensor_list

    def set_result(self, output_tensor_list):
        if self.batch_size != output_tensor_list[0].size(0):
            log.error(f'[Request_Batch:set_result] set_result error.')
        num_output_tensor = len(output_tensor_list)
        for batch_index in range(self.batch_size):
            request_single = self.request_single_list[batch_index]
            for output_tensor_index in range(num_output_tensor):
                output_tensor = output_tensor_list[output_tensor_index][batch_index:batch_index+1, ...]
                output_tensor.share_memory_()
                request_single.output_tensor_list.append(output_tensor)
                del output_tensor

    def get_cost_time(self):
        return (time.time() - self.create_request_batch_instance_time) * 1000

    def __str__(self):
        parent_info = super().__str__()
        return f'--> [ {ColorsLog.YELLOW}name{ColorsLog.RESET}: {self.name} , {parent_info}, {ColorsLog.YELLOW}batch_size{ColorsLog.RESET}: {self.batch_size} | {ColorsLog.YELLOW}self.max_batch_size{ColorsLog.RESET}: {self.max_batch_size} | {ColorsLog.YELLOW}self.max_duration{ColorsLog.RESET}: {self.max_duration}]'

