package layer

import (
	"github.com/mitchellh/hashstructure"
	"github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/dlperf/pkg/benchmark"
)

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#BatchNormalization

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

func (c BatchNorm) FwdBenchmarkName() string {
	return "LAYER_CUDNN_BATCHNORM_FWD_INFERENCE"
}

func (c BatchNorm) FwdCUDNNName() string {
	return ""
}

func (c BatchNorm) FwdTiming(system string /* hardware/software struct */) string {
	return ""
}

func (c BatchNorm) FwdBenchmarkAlgorithms() []string {
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
	baseBenchmarkArgs
	BaseBenchmarkInputArgs
}

func (c BatchNorm) FwdBenchmarkGeneratorArgNames() []string {
	return benchmarkArgNames(reluBenchmarkArgs{})
}

func (c BatchNorm) FwdBenchmarkArgs() interface{} {

	res := batchnormBenchmarkArgs{
		BaseBenchmarkInputArgs: mkBaseBenchmarkInputArgs(&c),
		baseBenchmarkArgs:      mkBaseBenchmarkArgs(&c),
	}

	hash, err := hashstructure.Hash(
		res,
		&hashstructure.HashOptions{
			TagName: "args",
		},
	)
	if err != nil {
		panic(err)
	}
	res.UniqueBenchmarkID = hash

	return res
}

func (c BatchNorm) FwdBenchmarkFilter(datatype, algorithm string) benchmark.Benchmark {
	if algorithm == "" {
		algorithm = c.FwdBenchmarkAlgorithms()[0]
	}
	return benchmark.Benchmark{
		Name:       mkBenchmarkFilterName(&c, datatype, algorithm),
		Attributes: benchmarkAttributes(c.FwdBenchmarkArgs()),
	}
}

func (c BatchNorm) FwdBenchmarkGenerator() string {
	const templString = `
[[ range $datatype := .DataTypes ]]
template <cudnnBatchNormMode_t batchnorm_mode>
static void [[ $.BenchmarkName ]]_[[ $datatype.Name | upper ]]__[[$.UniqueBenchmarkID]](benchmark::State& state) {
  [[ $.BenchmarkName ]]_Impl<[[ $datatype.CType ]], batchnorm_mode>(state);
  BENCHMARK_[[ $.BenchmarkName ]]_ADD_COUNTERS__[[$.UniqueBenchmarkID]](state);
}
[[ end ]]
`

	return templateExec(&c, templateBasePrefix+templString+templateBaseSuffix)
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
