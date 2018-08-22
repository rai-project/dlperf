package layer

import (
	"strings"

	"github.com/mitchellh/hashstructure"
	"github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/dlperf/pkg/benchmark"
)

type Relu struct {
	*Base `json:",inline,flatten,omitempty"`
}

func (Relu) OperatorType() string {
	return "Relu"
}

func (Relu) Description() string {
	return ``
}

func (c *Relu) InferShape(inputLayers dlperf.Layers) {
	inputShapes := getOutputShapes(inputLayers)
	c.SetInputShapes(inputShapes)
	c.SetOutputShapes(inputShapes)
}

func (c Relu) FwdBenchmarkName() string {
	return "LAYER_CUDNN_ACTIVATION_FWD"
}

func (c Relu) FwdCUDNNName() string {
	return ""
}

func (c Relu) FwdTiming(system string /* hardware/software struct */) string {
	return ""
}

func (c Relu) FwdBenchmarkAlgorithms() []string {
	switch strings.ToLower(c.onnxOperatorType) {
	case "sigmoid":
		return []string{
			"CUDNN_ACTIVATION_SIGMOID",
		}
	case "tanh":
		return []string{
			"CUDNN_ACTIVATION_TANH",
		}
	case "relu":
		return []string{
			"CUDNN_ACTIVATION_RELU",
			"CUDNN_ACTIVATION_CLIPPED_RELU",
		}
	case "elu":
		return []string{
			"CUDNN_ACTIVATION_ELU",
		}
	case "prelu":
		return []string{
			"ACTIVATION_PRELU",
		}
	case "leakyrelu":
		return []string{
			"ACTIVATION_LEAKY_RELU",
		}
	case "identity":
		return []string{
			"CUDNN_ACTIVATION_IDENTITY",
		}
	}

	panic("invalid relu operator = " + c.onnxOperatorType)

	return nil
}

type reluBenchmarkArgs struct {
	BaseBenchmarkArgs
	BaseBenchmarkInputArgs
}

func (c Relu) FwdBenchmarkGeneratorArgNames() []string {
	return benchmarkArgNames(reluBenchmarkArgs{})
}

func (c Relu) FwdBenchmarkArgs() interface{} {
	res := reluBenchmarkArgs{
		BaseBenchmarkInputArgs: mkBaseBenchmarkInputArgs(&c),
		BaseBenchmarkArgs:      mkBaseBenchmarkArgs(&c),
	}

	// substitution because cudnn does not support certain algorithms
	for ii, alg := range res.Algorithms {
		switch strings.ToUpper(alg) {
		case "ACTIVATION_PRELU", "ACTIVATION_LEAKY_RELU":
			res.Algorithms[ii] = "CUDNN_ACTIVATION_RELU"
		}
	}

	hash, err := hashstructure.Hash(
		res,
		&hashstructure.HashOptions{
			TagName: "hash",
		},
	)
	if err != nil {
		panic(err)
	}
	res.UniqueBenchmarkID = hash

	return res
}

func (c Relu) FwdBenchmarkFilter(datatype, algorithm string) benchmark.Benchmark {
	if algorithm == "" {
		algorithm = c.FwdBenchmarkAlgorithms()[0]
	}
	return benchmark.Benchmark{
		Name:       mkBenchmarkFilterName(&c, datatype, algorithm),
		Attributes: benchmarkAttributes(c.FwdBenchmarkArgs()),
	}
}

func (c Relu) FwdBenchmarkGenerator() string {
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

func (c Relu) Shape() dlperf.ShapeInformation {
	return c.Information().Shape()
}

func (c Relu) Information() dlperf.LayerInformation {
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

	checkNumber(c.InputShapes, []int{1, 2}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputShapes, []int{1}, c.OperatorType(), "number of outputs")

	inputShapes := c.InputShapes()[0]

	numOps := int64(1)
	for _, s := range inputShapes {
		numOps *= s
	}

	info.flops = dlperf.FlopsInformation{
		Comparisons: numOps,
	}

	return info
}

func init() {
	dlperf.Register(&Relu{})
}
