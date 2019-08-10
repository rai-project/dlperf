#!/bin/sh

set -x

go build


declare -a batch_sizes=(
  1 \
  8
)

declare -a machines=(
  k80 \
  v100
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