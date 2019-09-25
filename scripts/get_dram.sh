#!/bin/sh
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

declare -a machines=(
  Tesla_K80 \
  Tesla_M60 \
  Quadro_RTX_6000 \
  TITAN_V \
  TITAN_Xp \
  Tesla_P100-PCIE-16GB \
  Tesla_V100-SXM2-16GB
)


metrics="dram_read_bytes,dram_write_bytes"


for batch_size in "${batch_sizes[@]}"
do
    echo $batch_size
    for machine in "${machines[@]}"
        do
        echo $machine
        ./dlperf benchinfo --model_path ~/data/carml/dlperf/ResNet50-v1/resnet50v1/resnet50v1.onnx  --benchmark_database results/${machine}/profile/${batch_size}.json.gz --short=false --batch_size=${batch_size} --human=false --strategy=parallel --metrics --output_file=./assets/dram/batchsize_${batch_size}/${machine} --flops_only --flops_aggregate=true --metric_filter=${metrics} --total=false --format=csv --trim_layer_name=false
    done
done
