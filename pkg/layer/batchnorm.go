package layer

import (
	"github.com/mitchellh/hashstructure"
	dlperf "github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/dlperf/pkg/benchmark"
)

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#BatchNormalization
//easyjson:json
type BatchNorm struct {
	*Base   `json:",inline,flatten,omitempty"`
	Spatial int64 `json:"spatial,omitempty"`
}

func (BatchNorm) OperatorType() string {
	return "BatchNorm"
}

func (BatchNorm) Description() string {
	return ``
}

func (c *BatchNorm) InferShape(inputLayers dlperf.Layers) {
	inputShapes := getOutputShapes(inputLayers)
	c.SetInputShapes(inputShapes)
	c.SetOutputShapes(inputShapes)
}

func (c BatchNorm) FwdBenchmarkName(iopts ...dlperf.FwdBenchmarkArgsOptionFunc) string {
	opts := dlperf.CreateFwdBenchmarkArgsOption(iopts...)
	if opts.IsTraining {
		return "LAYER_CUDNN_BATCHNORM_FWD_TRAINING"
	}
	return "LAYER_CUDNN_BATCHNORM_FWD_INFERENCE"
}

func (c BatchNorm) BwdBenchmarkName(opts ...dlperf.BwdBenchmarkArgsOptionFunc) string {
	return "LAYER_CUDNN_BATCHNORM_BWD"
}

func (c BatchNorm) FwdCUDNNName() string {
	return ""
}

func (c BatchNorm) BwdCUDNNName() string {
	return ""
}

func (c BatchNorm) FwdTiming(system string /* hardware/software struct */) string {
	return ""
}

func (c BatchNorm) BwdTiming(system string /* hardware/software struct */) string {
	return ""
}

func (c BatchNorm) FwdBenchmarkAlgorithms(opts ...dlperf.FwdBenchmarkArgsOptionFunc) []string {
	return c.BenchmarkAlgorithms()
}

func (c BatchNorm) BwdBenchmarkAlgorithms(opts ...dlperf.BwdBenchmarkArgsOptionFunc) []string {
	return c.BenchmarkAlgorithms()
}

func (c BatchNorm) BenchmarkAlgorithms() []string {
	switch c.Spatial {
	case 1:
		return []string{
			"CUDNN_BATCHNORM_SPATIAL",
			"CUDNN_BATCHNORM_SPATIAL_PERSISTENT",
		}
	default:
		return []string{
			"CUDNN_BATCHNORM_PER_ACTIVATION",
		}
	}
}

type batchnormBenchmarkArgs struct {
	BaseBenchmarkArgs
	BaseBenchmarkInputArgs
	IsTraining bool
}

func (c BatchNorm) FwdBenchmarkArgs(iopts ...dlperf.FwdBenchmarkArgsOptionFunc) interface{} {
	opts := dlperf.CreateFwdBenchmarkArgsOption(iopts...)
	res := batchnormBenchmarkArgs{
		BaseBenchmarkInputArgs: mkBaseBenchmarkInputArgs(&c),
		BaseBenchmarkArgs:      mkBaseBenchmarkFWDArgs(&c, iopts...),
		IsTraining:             opts.IsTraining,
	}

	hash, err := hashstructure.Hash(
		res,
		&hashstructure.HashOptions{
			TagName: "hash",
		},
	)
	if err != nil {
		log.Fatal(err)
	}
	res.UniqueBenchmarkID = hash

	return res
}

func (c BatchNorm) BwdBenchmarkArgs(iopts ...dlperf.BwdBenchmarkArgsOptionFunc) interface{} {
	res := batchnormBenchmarkArgs{
		BaseBenchmarkInputArgs: mkBaseBenchmarkInputArgs(&c),
		BaseBenchmarkArgs:      mkBaseBenchmarkBWDArgs(&c, iopts...),
	}

	hash, err := hashstructure.Hash(
		res,
		&hashstructure.HashOptions{
			TagName: "hash",
		},
	)
	if err != nil {
		log.Fatal(err)
	}
	res.UniqueBenchmarkID = hash

	return res
}

func (c BatchNorm) FwdBenchmarkFilter(datatype, algorithm string, opts ...dlperf.FwdBenchmarkArgsOptionFunc) benchmark.Benchmark {
	if algorithm == "" {
		algorithm = c.FwdBenchmarkAlgorithms(opts...)[0]
	}
	return benchmark.Benchmark{
		Name:       mkFwdBenchmarkFilterName(&c, datatype, algorithm),
		Attributes: benchmarkAttributes(c.FwdBenchmarkArgs()),
	}
}

func (c BatchNorm) BwdBenchmarkFilter(datatype, algorithm string, opts ...dlperf.BwdBenchmarkArgsOptionFunc) benchmark.Benchmark {
	if algorithm == "" {
		algorithm = c.BwdBenchmarkAlgorithms(opts...)[0]
	}
	return benchmark.Benchmark{
		Name:       mkBwdBenchmarkFilterName(&c, datatype, algorithm),
		Attributes: benchmarkAttributes(c.BwdBenchmarkArgs(opts...)),
	}
}

func (c BatchNorm) FwdBenchmarkGenerator(opts ...dlperf.FwdBenchmarkArgsOptionFunc) string {
	templString := _escFSMustString(false, "/scope/batchnorm.tmpl")
	return templateExecFWD(&c, templateBasePrefix+templString+templateBaseSuffix, opts...)
}

func (c BatchNorm) BwdBenchmarkGenerator(opts ...dlperf.BwdBenchmarkArgsOptionFunc) string {
	templString := _escFSMustString(false, "/scope/batchnorm.tmpl")
	return templateExecBWD(&c, templateBasePrefix+templString+templateBaseSuffix)
}

func (c BatchNorm) FwdBenchmarkGeneratorArgNames() []string {
	return benchmarkArgNames(batchnormBenchmarkArgs{})
}

func (c BatchNorm) BwdBenchmarkGeneratorArgNames() []string {
	return benchmarkArgNames(batchnormBenchmarkArgs{})
}

func (c BatchNorm) Shape() dlperf.ShapeInformation {
	return c.Information().Shape()
}

func (c BatchNorm) Information() dlperf.LayerInformation {
	info := &Information{
		Base: c.Base,
		shape: dlperf.ShapeInformation{
			InputShapes:  c.InputShapes(),
			OutputShapes: c.OutputShapes(),
		},
	}

	if isAnyEmpty(c.OutputShapes()) {
		log.WithField("layer", c.OperatorType()).Info("len(OutputShapes) is 0")
		return info
	}

	checkNumber(c.InputShapes, []int{5}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputShapes, []int{1, 2, 3, 4, 5}, c.OperatorType(), "number of outputs")

	inputShapes := c.InputShapes()[0]

	numOps := int64(1)
	for _, s := range inputShapes {
		numOps *= s
	}

	// this is for inference
	info.flops = dlperf.FlopsInformation{
		Additions: numOps,
		Divisions: numOps,
	}

	return info
}

func init() {
	dlperf.Register(&BatchNorm{})
}
