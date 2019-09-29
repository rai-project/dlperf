# DLPerf  [![View notebooks](https://www.wolframcloud.com/obj/user-3ed91f5b-da42-447a-8a46-df433a144012/dev/WDV/badge2.png)](https://www.wolframcloud.com/obj/user-3ed91f5b-da42-447a-8a46-df433a144012/dev/WDV/wdv_api?user=rai-project&repo=dlperf&branch=master)


[![Build Status](https://dev.azure.com/dakkak/rai/_apis/build/status/rai-project.dlperf?branchName=master)](https://dev.azure.com/dakkak/rai/_build/latest?definitionId=26&branchName=master)
[![Build Status](https://travis-ci.org/rai-project/dlperf.svg?branch=master)](https://travis-ci.org/rai-project/dlperf)

## Show Registered Models

Show the models listed models in [model_urls](cmd/model_urls.go)

```bash
go run main.go list
```

## Download Models

Download the models listed models in [model_urls](cmd/model_urls.go) to `model_dir` using

```bash
go run main.go downloadmodels -o ~/data/carml/dlperf
```

## Show Downloaded Models

Show downloaded model paths

```bash
go run main.go paths --model_path ~/onnx_models
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

## Get Weights Histogram

Get information per layer using

```
go run main.go weightsinfo --model_path ~/onnx_models/bvlc_alexnet/model.onnx --output_file=out
```

## Store the recalled benchmark in json

Store the recalled benchmarks in json using

```
go run main.go benchinfo --model_path ~/data/carml/ --benchmark_database results/v100/8.json --short=false --batch_size=8 --human=true -o assets/benchinfo/v100 -f json
```

## Find Patterns

`model_path` and `output_path` can be a folder or a file.

Find the patterns of length 4

```bash
go run main.go patterns --model_path ~/onnx_models/ --length 4
```

## Generate Benchmarks

Generate the benchmark files of a model or across models at `model_path`.
Use `--forward` and `--backward` to control whether to generate benchmarks for forward and backward pass.

```
go run main.go benchgen --model_path ~/data/carml/dlperf/Emotion-FerPlus/emotion_ferplus/model.onnx --forward=true --backward=false -o test_generated_benchmarks.hpp
```

or

```
 ./scripts/gen_benchmarks.sh
```

## Query benchmark database

Query benchmark database at `benchmark_database` to to get information on the model at `model_path`

```
go run main.go benchinfo --model_path ~/data/carml/dlperf/BVLC_AlexNet/bvlc_alexnet/model.onnx --benchmark_database results/Tesla_V100-SXM2-16GB/1.json --output_file=testout --short=false --batch_size=1 --human=false --strategy=parallel --total=true --format=json --trim_layer_name=false
```

Options:

- trim_layer_name: limit the layer name length
- strategy: "parallel" executes the layers on the short path; "serial" serializes the layers
- total: output total number of all layers

## Draw a graph of the layers using benchmark data

You can draw a graph with the runtime data using the following command

```
go run main.go benchinfo --model_path ~/data/carml/dlperf/ArcFace/resnet100/resnet100.onnx --benchmark_database results/v100/8.json --short=false --batch_size=8 --human=true --strategy=parallel --show --highlight_fast_path
```

## Query metrics

You can query both kernels and metrics that are dummed by cudnn|scope using the following command

```
go run main.go benchinfo --model_path ~/data/carml/dlperf/ResNet50-v1/resnet50v1/resnet50v1.onnx --benchmark_database results/v100/profile/8.json.gz --short=false --batch_size=8 --human=true --strategy=parallel --metrics --format=json
```

### Show only metric summary

```
go run main.go benchinfo --model_path ~/data/carml/dlperf/ResNet50-v1/resnet50v1/resnet50v1.onnx --benchmark_database results/v100/profile/8.json.gz --short=false --batch_size=8 --human=false --strategy=parallel --metrics --output_file=tmp.tbl --flops_only --flops_aggregate=true
```

### Filter only certain metrics

```
go run main.go benchinfo --model_path ~/data/carml/dlperf/ResNet50-v1/resnet50v1/resnet50v1.onnx --benchmark_database results/v100/profile/8.json.gz --short=false --batch_size=8 --human=false --strategy=parallel --metrics --output_file=tmp.tbl --flops_only --flops_aggregate=true --metric_filter=flop_count_sp --trim_layer_name=false
```

## Show Layer to kernel name mapping

```
go run main.go benchinfo --model_path ~/data/carml/dlperf/ResNet50-v1/resnet50v1/resnet50v1.onnx --benchmark_database results/v100/profile/8.json.gz --short=false --batch_size=8 --human=false --strategy=parallel --metrics --output_file=tmp.tbl --trim_layer_name=false --total=false --format=csv --kernels_only=true
```

## References

- https://github.com/onnx/models
- https://github.com/rai-project/onnx_examples (private)
-
