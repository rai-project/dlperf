#!/bin/sh

set -x

declare -a models=(
  bvlc_alexnet \
  bvlc_googlenet \
  bvlc_reference_caffenet \
  bvlc_reference_rcnn_ilsvrc13 \
  densenet121 \
  emotion_ferplus \
  inception_v1 \
  inception_v2 \
  mnist \
  resnet50 \
  shufflenet \
  squeezenet \
  tiny_yolov2 \
  vgg19 \
  zfnet512
  )


for i in "${models[@]}"
do
    echo $i
    go run  main.go weightsinfo --model_path ~/onnx_models_1.3/$i/model.onnx --output_file=onnx_models_1.3/out
done
