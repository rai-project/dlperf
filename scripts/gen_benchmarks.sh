#!/bin/sh
set -x

mkdir -p gen
go run main.go benchgen --model_path ~/data/carml/dlperf -o gen/generated_benchmarks_1.hpp --backward=true --forward=true --batch_size=1
go run main.go benchgen --model_path ~/data/carml/dlperf -o gen/generated_benchmarks_2.hpp --backward=true --forward=true --batch_size=2
go run main.go benchgen --model_path ~/data/carml/dlperf -o gen/generated_benchmarks_4.hpp --backward=true --forward=true --batch_size=4
go run main.go benchgen --model_path ~/data/carml/dlperf -o gen/generated_benchmarks_8.hpp --backward=true --forward=true --batch_size=8
go run main.go benchgen --model_path ~/data/carml/dlperf -o gen/generated_benchmarks_16.hpp --backward=true --forward=true --batch_size=16
