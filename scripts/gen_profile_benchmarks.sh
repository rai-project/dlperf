#!/bin/bash
set -x


go build

declare -a batch_sizes=(
  1 \
  2 \
  4 \
  8 \
  16 \
  32 \
  64 \
  128 \
  256 \
  512 \
  1024
)

mkdir -p gen
for batch_size in "${batch_sizes[@]}"
do
    echo $batch_size
    go run main.go benchgen --model_path ~/data/carml/dlperf/ResNet50-v1/resnet50v1/resnet50v1.onnx -o gen/generated_profile_benchmarks_${batch_size}.hpp --backward=false --forward=true --batch_size=${batch_size}
done
