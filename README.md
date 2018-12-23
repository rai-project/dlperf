# DLPerf

[![Build Status](https://travis-ci.org/rai-project/dlperf.svg?branch=master)](https://travis-ci.org/rai-project/dlperf)

## Download Models

Download the models listed models in [model_urls](cmd/model_urls.go) to `model_dir` using

```bash
go run main.go downloadmodels --model_dir ~/onnx_models
```

## Layer Stats

`model_path` and `output_path` can be a folder or a file.

### Infer and view the shape information using

```bash
go run main.go layerstats --model_path ~/onnx_models/emotion_ferplus/model.onnx --format dot
```

### Get layer statistics in json format using

```bash
go run main.go layerstats --model_path ~/onnx_models --output_path assets/layer_stats --format json
```

`dot` from `graphviz` is needed. On macos, install it using `brew install graphviz`.

## Get Flops

Get the flops information for alexnet using

```
go run main.go flopsinfo --model_path ~/onnx_models/bvlc_alexnet/model.onnx
```

Get information per layer using

```
go run main.go flopsinfo --model_path ~/onnx_models/bvlc_alexnet/model.onnx --full
```

## Find Patterns

`model_path` and `output_path` can be a folder or a file.

Find the patterns of length 4

```bash
go run main.go patterns --model_path ~/onnx_models/ --length 4
```

## Generate Benchmarks

Generate the benchmark files of a model or across models at `model_path`.

```
go run main.go benchgen --model_path ~/onnx_models/bvlc_alexnet/model.onnx
```

## Query benchmark database

Query benchmark database at `benchmark_database` to to get information on the model at `model_path`

```
go run main.go benchinfo --model_path ~/onnx_models/vgg19/vgg19.onnx
```

or

```
go run main.go benchinfo --model_path ~/onnx_models/resnet100/resnet100.onnx --benchmark_database ../microbench/results/cudnn/ip-172-31-26-89.json
```
