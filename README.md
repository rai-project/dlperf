# DLPerf
[![Build Status](https://travis-ci.org/rai-project/dlperf.svg?branch=master)](https://travis-ci.org/rai-project/dlperf)


## Layer Stats

Get layer statistics in json format using

```bash
go run main.go layerstats --model_path ~/onnx_models --output assets/layer_stats --format json
```

infer and view the shape information

```bash
go run main.go layerstats --model_path ~/onnx_models/emotion_ferplus/model.onnx --format dot
```

## Find Patterns

Find the patterns of length 4

```bash
go run main.go patterns --model_path ~/onnx_models/ --length 4
```

## Get Flops

To get the flops information for alexnet use

```
go run main.go flopsinfo --model_path ~/onnx_models/bvlc_alexnet/model.onnx
```

to get information per layer use

```
go run main.go flopsinfo --model_path ~/onnx_models/bvlc_alexnet/model.onnx --full
```
