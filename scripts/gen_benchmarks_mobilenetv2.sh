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
    go run main.go benchgen --model_path ~/data/carml/dlperf/MobileNet-v2/ -o gen/generated_mobilenetv2_benchmarks_${batch_size}.hpp --backward=false --forward=true --randomize=true --batch_size=${batch_size}
done
