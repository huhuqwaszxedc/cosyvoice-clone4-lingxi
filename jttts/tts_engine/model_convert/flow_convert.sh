#!/usr/bin/env bash

# set -e
set -u
source /root/miniconda3/bin/activate
conda activate cv2_ttsfront


if [ $# -ne 1 ]; then
  echo "Usage: $0  <model_dir>"
  echo "model_dir: model_dir abs_path"
  exit 1
fi

model_dir=$1


unset SETUPTOOLS_USE_DISTUTILS
export PYTHONPATH=/root/work/filestorage/usr/yanghuibao/deploy_info/clone4.0_deploy/flow_convert/CosyVoice-main-acc-gws:/root/work/filestorage/usr/yanghuibao/deploy_info/clone4.0_deploy/flow_convert/CosyVoice-main-acc-gws/third_party/Matcha-TTS:/root/work/filestorage/usr/mayong/app/llm/vllm/vllm
# export LD_LIBRARY_PATH=/root/work/filestorage/usr/mayong/tools/tensorrt/TensorRT-10.6.0.26/lib:/usr/local/cuda/compat/lib.real:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64



python export_onnx.py --model_dir $model_dir

echo "convert onnx success"

TRT_DIR=/root/work/filestorage/usr/yanghuibao/deploy_info/clone4.0_deploy/flow_convert/TensorRT-10.6.0.26/
MODEL_DIR=$model_dir

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TRT_DIR/lib

$TRT_DIR/bin/trtexec --onnx=$MODEL_DIR/flow.decoder.estimator.fp32.2.onnx --saveEngine=$MODEL_DIR/flow.decoder.estimator.fp16.h100.jtaudio.plan --fp16 --minShapes=x:2x80x15,mask:2x1x15,mu:2x80x15,cond:2x80x15,t:2,spks:2x80 --optShapes=x:32x80x512,mask:32x1x512,mu:32x80x512,cond:32x80x512,t:32,spks:32x80 --maxShapes=x:128x80x1536,mask:128x1x1536,mu:128x80x1536,cond:128x80x1536,t:96,spks:128x80 --inputIOFormats=fp16:chw,fp16:chw,fp16:chw,fp16:chw,fp16:chw,fp16:chw --outputIOFormats=fp16:chw

echo "trt conver success"

