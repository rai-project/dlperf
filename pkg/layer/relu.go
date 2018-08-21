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
	baseBenchmarkArgs
	Input0 int64 `args:"input[0]"`
	Input1 int64 `args:"input[1]"`
	Input2 int64 `args:"input[2]"`
	Input3 int64 `args:"input[3]"`
	Input4 int64 `args:"input[4]"`
	Input5 int64 `args:"input[5]"`
	Input6 int64 `args:"input[6]"`
	Input7 int64 `args:"input[7]"`
}

func (c Relu) FwdBenchmarkGeneratorArgNames() []string {
	return benchmarkArgNames(reluBenchmarkArgs{})
}

func (c Relu) FwdBenchmarkArgs() interface{} {
	inShape := c.InputShapes()[0]
	get := func(idx int) int64 {
		if len(inShape) <= idx {
			return -1
		}
		return inShape[idx]
	}

	res := reluBenchmarkArgs{
		Input0:            get(0),
		Input1:            get(1),
		Input2:            get(2),
		Input3:            get(3),
		Input4:            get(4),
		Input5:            get(5),
		Input6:            get(6),
		Input7:            get(7),
		baseBenchmarkArgs: mkBaseBenchmarkArgs(&c),
	}

	hash, err := hashstructure.Hash(res, nil)
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
    CUDNN_RELU_FWD_Impl<[[ $datatype.CType ]], activation_mode>(state);
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
			InputShapes:  c.inputShapes,
			OutputShapes: c.outputShapes,
		},
	}

	if isAnyEmpty(c.outputShapes) {
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
