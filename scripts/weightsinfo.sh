#!/bin/sh

go run  main.go weightsinfo --model_path ~/onnx_models/mnist/model.onnx --output_file=out
go run  main.go weightsinfo --model_path ~/onnx_models/bvlc_alexnet/model.onnx --output_file=out
