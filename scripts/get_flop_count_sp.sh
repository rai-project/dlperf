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
        ./dlperf benchinfo --model_path ~/data/carml/dlperf/ --benchmark_database results/${machine}/profile/${batch_size}.json.gz --short=false --batch_size=${batch_size} --human=false --strategy=parallel --metrics --output_file=./assets/flop_count_sp/batchsize_${batch_size}/${machine} --flops_only --flops_aggregate=true --metric_filter=flop_count_sp --total=false --format=csv --trim_layer_name=false
    done
done