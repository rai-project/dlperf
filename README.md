# DLPerf
[![Build Status](https://travis-ci.org/rai-project/dlperf.svg?branch=master)](https://travis-ci.org/rai-project/dlperf)


## Layer Stats

Get layer statistics in json format using

```bash
go run main.go layerstats --model_path ~/onnx_models --output assets/layer_stats --format json
```


## Find Patterns

Find the patterns of length 4

```bash
go run main.go patterns --model_path ~/onnx_models/ --length=4
```
