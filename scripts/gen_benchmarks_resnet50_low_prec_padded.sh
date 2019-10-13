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
    go run main.go benchgen --model_path ~/data/carml/dlperf/ResNet50-v1 -o gen/generated_resnet50_low_prec_padded_benchmarks_${batch_size}.hpp --fused=false --backward=false --forward=true --randomize=true --datatype=float16 --pad_layers --pad_layers_multiple=8 --batch_size=${batch_size}
done
