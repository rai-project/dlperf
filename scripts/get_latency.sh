#!/bin/bash

# set -x

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
  # Tesla_K80 \
  # Tesla_M60 \
  # Quadro_RTX_6000 \
  # TITAN_V \
  # TITAN_Xp \
  Tesla_T4 \
  # Tesla_V100-SXM2-16GB \
  # GRID_K520 \
)



# for batch_size in "${batch_sizes[@]}"
# do
#     echo $batch_size
#     for machine in "${machines[@]}"
#     do
#         echo $machine
#         # ./dlperf benchinfo --model_path ~/data/carml/dlperf/ResNet50-v1/ --benchmark_database results/resnet50/${machine}/${batch_size}.json --short=false --batch_size=${batch_size} --human=false --strategy=parallel --output_file=./assets/resnet50_latency/batchsize_${batch_size}/${machine} --total=true --format=csv --trim_layer_name=false
#         # ./dlperf benchinfo --model_path ~/data/carml/dlperf/ResNet50-v1/ --benchmark_database results/resnet50/${machine}/${batch_size}.json --short=false --batch_size=${batch_size} --human=false --strategy=serial --output_file=./assets/resnet50_serial_latency/batchsize_${batch_size}/${machine} --total=true --format=csv --trim_layer_name=false
#         # ./dlperf benchinfo --model_path ~/data/carml/dlperf/ResNet50-v1/ --benchmark_database results/resnet50/${machine}/${batch_size}.json --short=false --batch_size=${batch_size} --human=false --strategy=parallel --output_file=./assets/fused_latency/batchsize_${batch_size}/${machine} --total=true --format=csv --trim_layer_name=false --fused
#         # ./dlperf benchinfo --model_path ~/data/carml/dlperf/ResNet50-v1/ --benchmark_database results/resnet50/${machine}/${batch_size}.json --short=false --batch_size=${batch_size} --human=false --strategy=serial --output_file=./assets/fused_serial_latency/batchsize_${batch_size}/${machine} --total=true --format=csv --trim_layer_name=false --fused
#         ./dlperf benchinfo --model_path ~/data/carml/dlperf/ --benchmark_database results/${machine}/${batch_size}.json --short=false --batch_size=${batch_size} --human=false --strategy=serial --output_file=./assets/cudnn_advised_serial_latency/batchsize_${batch_size}/${machine} --total=true --format=csv --trim_layer_name=false --choose_cudnn_heuristics
#         ./dlperf benchinfo --model_path ~/data/carml/dlperf/ --benchmark_database results/${machine}/${batch_size}.json --short=false --batch_size=${batch_size} --human=false --strategy=serial --output_file=./assets/serial_latency/batchsize_${batch_size}/${machine} --total=true --format=csv --trim_layer_name=false
#         ./dlperf benchinfo --model_path ~/data/carml/dlperf/ --benchmark_database results/${machine}/${batch_size}.json --short=false --batch_size=${batch_size} --human=false --strategy=parallel --output_file=./assets/cudnn_advised_latency/batchsize_${batch_size}/${machine} --total=true --format=csv --trim_layer_name=false --choose_cudnn_heuristics
#         ./dlperf benchinfo --model_path ~/data/carml/dlperf/ --benchmark_database results/${machine}/${batch_size}.json --short=false --batch_size=${batch_size} --human=false --strategy=parallel --output_file=./assets/latency/batchsize_${batch_size}/${machine} --total=true --format=csv --trim_layer_name=false
#     done
# done



declare -a machines=(
  # Quadro_RTX_6000 \
  # TITAN_V \
  # Tesla_T4 \
  Tesla_V100-SXM2-16GB \
)

for batch_size in "${batch_sizes[@]}"
do
    echo $batch_size
    for machine in "${machines[@]}"
    do
      echo $machine
       ./dlperf benchinfo --model_path ~/data/carml/dlperf/ResNet50-v1/ --benchmark_database results/low_prec/${machine}/${batch_size}.json --short=false --batch_size=${batch_size} --human=false --strategy=parallel --output_file=./assets/latency_float16/batchsize_${batch_size}/${machine} --total=true --format=csv --trim_layer_name=false --datatype=float16
       ./dlperf benchinfo --model_path ~/data/carml/dlperf/ResNet50-v1/ --benchmark_database results/low_prec/${machine}/${batch_size}.json --short=false --batch_size=${batch_size} --human=false --strategy=serial --output_file=./assets/serial_latency_float16/batchsize_${batch_size}/${machine} --total=true --format=csv --trim_layer_name=false --datatype=float16
       ./dlperf benchinfo --model_path ~/data/carml/dlperf/ResNet50-v1/ --benchmark_database results/nhwc_low_prec/${machine}/${batch_size}.json --short=false --batch_size=${batch_size} --human=false --strategy=serial --output_file=./assets/serial_nhwc_latency_float16/batchsize_${batch_size}/${machine} --total=true --format=csv --trim_layer_name=false --datatype=float16
       ./dlperf benchinfo --model_path ~/data/carml/dlperf/ResNet50-v1/ --benchmark_database results/low_prec/${machine}/${batch_size}.json --short=false --batch_size=${batch_size} --human=false --strategy=parallel --output_file=./assets/fused_latency_float16/batchsize_${batch_size}/${machine} --total=true --format=csv --trim_layer_name=false --datatype=float16 --fused
       ./dlperf benchinfo --model_path ~/data/carml/dlperf/ResNet50-v1/ --benchmark_database results/low_prec/${machine}/${batch_size}.json --short=false --batch_size=${batch_size} --human=false --strategy=serial --output_file=./assets/serial_fused_latency_float16/batchsize_${batch_size}/${machine} --total=true --format=csv --trim_layer_name=false --datatype=float16 --fused
    done
done


declare -a machines=(
  Tesla_V100-SXM2-16GB \
)


for batch_size in "${batch_sizes[@]}"
do
    echo $batch_size
    for machine in "${machines[@]}"
    do
      echo $machine
       ./dlperf benchinfo --model_path ~/data/carml/dlperf/ResNet50-v1/ --benchmark_database results/7.6.4.38/low_prec/${machine}/${batch_size}.json --short=false --batch_size=${batch_size} --human=false --strategy=parallel --output_file=./assets/7.6.4.38/latency_float16/batchsize_${batch_size}/${machine} --total=true --format=csv --trim_layer_name=false --datatype=float16
       ./dlperf benchinfo --model_path ~/data/carml/dlperf/ResNet50-v1/ --benchmark_database results/7.6.4.38/low_prec/${machine}/${batch_size}.json --short=false --batch_size=${batch_size} --human=false --strategy=serial --output_file=./assets/7.6.4.38/serial_latency_float16/batchsize_${batch_size}/${machine} --total=true --format=csv --trim_layer_name=false --datatype=float16
       ./dlperf benchinfo --model_path ~/data/carml/dlperf/ResNet50-v1/ --benchmark_database results/7.6.4.38/nhwc_low_prec/${machine}/${batch_size}.json --short=false --batch_size=${batch_size} --human=false --strategy=serial --output_file=./assets/7.6.4.38/serial_nhwc_latency_float16/batchsize_${batch_size}/${machine} --total=true --format=csv --trim_layer_name=false --datatype=float16
       ./dlperf benchinfo --model_path ~/data/carml/dlperf/ResNet50-v1/ --benchmark_database results/7.6.4.38/low_prec/${machine}/${batch_size}.json --short=false --batch_size=${batch_size} --human=false --strategy=parallel --output_file=./assets/7.6.4.38/fused_latency_float16/batchsize_${batch_size}/${machine} --total=true --format=csv --trim_layer_name=false --datatype=float16 --fused
       ./dlperf benchinfo --model_path ~/data/carml/dlperf/ResNet50-v1/ --benchmark_database results/7.6.4.38/low_prec/${machine}/${batch_size}.json --short=false --batch_size=${batch_size} --human=false --strategy=serial --output_file=./assets/7.6.4.38/serial_fused_latency_float16/batchsize_${batch_size}/${machine} --total=true --format=csv --trim_layer_name=false --datatype=float16 --fused
    done
done