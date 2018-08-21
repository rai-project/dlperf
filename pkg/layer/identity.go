package layer

import (
	"github.com/mitchellh/hashstructure"
	"github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/dlperf/pkg/benchmark"
)

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Identity

type Identity struct {
	*Base `json:",inline,flatten,omitempty"`
}

func (Identity) OperatorType() string {
	return "Identity"
}

func (Identity) Description() string {
	return ``
}

func (c *Identity) InferShape(inputLayers dlperf.Layers) {
	inputShapes := getOutputShapes(inputLayers)
	c.SetInputShapes(inputShapes)
	c.SetOutputShapes(inputShapes)
}

func (c Identity) FwdBenchmarkName() string {
	return "LAYER_CUDNN_ACTIVATION_FWD"
}

func (c Identity) FwdCUDNNName() string {
	return ""
}

func (c Identity) FwdTiming(system string /* hardware/software struct */) string {
	return ""
}

func (c Identity) FwdBenchmarkAlgorithms() []string {
	return []string{
		"CUDNN_ACTIVATION_IDENTITY",
	}
}

type identityBenchmarkArgs struct {
	baseBenchmarkArgs
	BaseBenchmarkInputArgs
}

func (c Identity) FwdBenchmarkGeneratorArgNames() []string {
	return benchmarkArgNames(identityBenchmarkArgs{})
}

func (c Identity) FwdBenchmarkArgs() interface{} {
	res := identityBenchmarkArgs{
		BaseBenchmarkInputArgs: mkBaseBenchmarkInputArgs(&c),
		baseBenchmarkArgs:      mkBaseBenchmarkArgs(&c),
	}

	hash, err := hashstructure.Hash(res, nil)
	if err != nil {
		panic(err)
	}
	res.UniqueBenchmarkID = hash

	return res
}

func (c Identity) FwdBenchmarkFilter(datatype, algorithm string) benchmark.Benchmark {
	if algorithm == "" {
		algorithm = c.FwdBenchmarkAlgorithms()[0]
	}
	return benchmark.Benchmark{
		Name:       mkBenchmarkFilterName(&c, datatype, algorithm),
		Attributes: benchmarkAttributes(c.FwdBenchmarkArgs()),
	}
}

func (c Identity) FwdBenchmarkGenerator() string {
	const templString = `
[[ range $datatype := .DataTypes ]]
template <cudnnActivationMode_t activation_mode>
static void [[ $.BenchmarkName ]]_[[ $datatype.Name | upper ]]__[[$.UniqueBenchmarkID]](benchmark::State& state) {
  [[ $.BenchmarkName ]]_Impl<[[ $datatype.CType ]], activation_mode>(state);
  BENCHMARK_[[ $.BenchmarkName ]]_ADD_COUNTERS__[[$.UniqueBenchmarkID]](state);
}
[[ end ]]
`

	return templateExec(&c, templateBasePrefix+templString+templateBaseSuffix)
}

func (c Identity) Shape() dlperf.ShapeInformation {
	return c.Information().Shape()
}

func (c Identity) Information() dlperf.LayerInformation {
	info := &Information{
		Base: c.Base,
		shape: dlperf.ShapeInformation{
			InputShapes:  c.inputShapes,
			OutputShapes: c.outputShapes,
		},
	}

	if isAnyEmpty(c.outputShapes) {
		log.WithField("layer", c.OperatorType()).Info("len(OutputShapes) is 0")
		return info
	}

	info.flops = dlperf.FlopsInformation{}

	return info
}

func init() {
	dlperf.Register(&Identity{})
}
