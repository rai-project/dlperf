#!/bin/sh

set -x

go build


declare -a batch_sizes=(
  1 \
  2 \
  4 \
  8 \
  16
)

declare -a machines=(
  Tesla_K80 \
  Tesla_V100-SXM2-16GB \
  Quadro_RTX_6000 \
  Tesla_P100-PCIE-16GB \
  Tesla_M60 \
  Tesla_P4
)


for batch_size in "${batch_sizes[@]}"
do
    echo $batch_size
    for machine in "${machines[@]}"
    do
        echo $machine
        ./dlperf benchinfo --model_path ~/data/carml/dlperf/ --benchmark_database results/${machine}/${batch_size}.json --short=false --batch_size=${batch_size} --human=false --strategy=parallel --output_file=./assets/latency/batchsize_${batch_size}/${machine} --total=true --format=csv --trim_layer_name=false
    done
done
