# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu, Zetao Hu)
#               2025 Alibaba Inc (authors: Xiang Lyu, Yabin Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import argparse
import torch
import torchaudio
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_model_dir', type=str,required=True, default="input_model_dir")
    args = parser.parse_args()

    return args



def get_trt_kwargs():
    min_shape = [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4)]
    opt_shape = [(2, 80, 500), (2, 1, 500), (2, 80, 500), (2, 80, 500)]
    max_shape = [(2, 80, 3000), (2, 1, 3000), (2, 80, 3000), (2, 80, 3000)]
    input_names = ["x", "mask", "mu", "cond"]
    return {'min_shape': min_shape, 'opt_shape': opt_shape, 'max_shape': max_shape, 'input_names': input_names}


def convert_onnx_to_trt(trt_model, trt_kwargs, onnx_model, fp16):
    import tensorrt as trt
    logging.info(f"Converting onnx to trt {trt_model} ...")
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)  # 4GB
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    # load onnx model
    with open(onnx_model, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError('failed to parse {}'.format(onnx_model))
    # set input shapes
    for i in range(len(trt_kwargs['input_names'])):
        profile.set_shape(trt_kwargs['input_names'][i], trt_kwargs['min_shape'][i], trt_kwargs['opt_shape'][i], trt_kwargs['max_shape'][i])
    tensor_dtype = trt.DataType.HALF if fp16 else trt.DataType.FLOAT
    # set input and output data type
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        input_tensor.dtype = tensor_dtype
    for i in range(network.num_outputs):
        output_tensor = network.get_output(i)
        output_tensor.dtype = tensor_dtype
    config.add_optimization_profile(profile)
    engine_bytes = builder.build_serialized_network(network, config)
    # save trt engine
    with open(trt_model, "wb") as f:
        f.write(engine_bytes)
    logging.info("Succesfully convert onnx to trt...")


def main(args):
    input_model_dir = args.input_model_dir
    if os.path.exists(input_model_dir) is False:
        logging.error(f'[error] input_model_dir in not exists! please check! {input_model_dir}')
        exit()
    
    onnx_model_file     = os.path.join(input_model_dir, 'flow.decoder.estimator.fp32.onnx')
    trt_kwargs = get_trt_kwargs()


    trt_model_fp32_file      = os.path.join(input_model_dir, 'flow.decoder.estimator.fp32.h100.my.plan')
    convert_onnx_to_trt(trt_model_fp32_file, trt_kwargs, onnx_model_file, fp16=False)

    trt_model_fp16_file      = os.path.join(input_model_dir, 'flow.decoder.estimator.fp16.h100.my.plan')    
    convert_onnx_to_trt(trt_model_fp16_file, trt_kwargs, onnx_model_file, fp16=True)




    logging.info(f'convert success !!')



if __name__ == '__main__':
    args = get_args()
    main(args)


