package layer

import (
	"github.com/mitchellh/hashstructure"
	"github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/dlperf/pkg/benchmark"
)

type Softmax struct {
	*Base `json:",inline,flatten,omitempty"`
}

func (Softmax) OperatorType() string {
	return "Softmax"
}

func (Softmax) Description() string {
	return ``
}

func (c *Softmax) InferShape(inputLayers dlperf.Layers) {
	inputShapes := getOutputShapes(inputLayers)
	c.SetInputShapes(inputShapes)
	c.SetOutputShapes(inputShapes)
}

func (c Softmax) FwdBenchmarkName() string {
	return "LAYER_CUDNN_SOFTMAX_FWD"
}

func (c Softmax) FwdCUDNNName() string {
	return ""
}

func (c Softmax) FwdTiming(system string /* hardware/software struct */) string {
	return ""
}

func (c Softmax) FwdBenchmarkAlgorithms() []string {
	return []string{
		"CUDNN_SOFTMAX_MODE_INSTANCE",
		"CUDNN_SOFTMAX_MODE_CHANNEL",
	}
}

func (c Softmax) Shape() dlperf.ShapeInformation {
	return c.Information().Shape()
}

type softmaxBenchmarkArgs struct {
	baseBenchmarkArgs
	BaseBenchmarkInputArgs
}

func (c Softmax) FwdBenchmarkGeneratorArgNames() []string {
	return benchmarkArgNames(softmaxBenchmarkArgs{})
}

func (c Softmax) FwdBenchmarkArgs() interface{} {

	res := softmaxBenchmarkArgs{
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

func (c Softmax) FwdBenchmarkFilter(datatype, algorithm string) benchmark.Benchmark {
	if algorithm == "" {
		algorithm = c.FwdBenchmarkAlgorithms()[0]
	}
	return benchmark.Benchmark{
		Name:       mkBenchmarkFilterName(&c, datatype, algorithm),
		Attributes: benchmarkAttributes(c.FwdBenchmarkArgs()),
	}
}

func (c Softmax) FwdBenchmarkGenerator() string {
	const templString = `
[[ range $datatype := .DataTypes ]]
template <cudnnSoftmaxAlgorithm_t softmax_algorithm, cudnnSoftmaxMode_t softmax_mode>
static void [[ $.BenchmarkName ]]_[[ $datatype.Name | upper ]]__[[$.UniqueBenchmarkID]](benchmark::State& state) {
  [[ $.BenchmarkName ]]_Impl<[[ $datatype.CType ]], softmax_algorithm, softmax_mode>(state);
  BENCHMARK_[[ $.BenchmarkName ]]_ADD_COUNTERS__[[$.UniqueBenchmarkID]](state);
}
[[ end ]]
#define BENCHMARK_[[ .BenchmarkName ]]0(b, SOFTMAX_MODE) \
[[ range $algorithm := .Algorithms ]] BENCHMARK_TEMPLATE(b, [[ $algorithm ]], SOFTMAX_MODE)->BENCHMARK_[[ $.BenchmarkName ]]_INPUT_ARG_NAMES()->UseManualTime(); \
[[ end ]]

#define BENCHMARK_[[ .BenchmarkName ]](b)                                                                                             \
  BENCHMARK_[[ .BenchmarkName ]]0(b, CUDNN_SOFTMAX_MODE_INSTANCE);                                                                    \
  BENCHMARK_[[ .BenchmarkName ]]0(b, CUDNN_SOFTMAX_MODE_CHANNEL)

[[ range $datatype := .DataTypes ]]BENCHMARK_[[ $.BenchmarkName ]]([[ $.BenchmarkName ]]_[[ $datatype.Name | upper ]]__[[$.UniqueBenchmarkID]]);
[[ end ]]
#undef BENCHMARK_[[ .BenchmarkName ]]_INPUT_ARGS
$undef BENCHMARK_[[ .BenchmarkName ]]_INPUT_ARG_NAMES
#undef BENCHMARK_[[ .BenchmarkName ]]0
#undef BENCHMARK_[[ .BenchmarkName ]]
}
`

	return templateExec(&c, templateBasePrefix+templString)
}

func (c Softmax) Information() dlperf.LayerInformation {
	info := &Information{
		Base: c.Base,
		shape: dlperf.ShapeInformation{
			InputShapes:  c.inputShapes,
			OutputShapes: c.outputShapes,
		},
	}

	if isAnyEmpty(c.inputShapes) {
		log.WithField("layer", c.OperatorType()).Info("len(InputShapes) is 0")
		return info
	}

	checkNumber(c.InputShapes, []int{1}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputShapes, []int{1}, c.OperatorType(), "number of outputs")

	inputShapes := c.InputShapes()[0]

	numOps := int64(1)
	for _, s := range inputShapes {
		numOps *= s
	}

	info.flops = dlperf.FlopsInformation{
		Exponentiations: numOps,
		Additions:       numOps,
		Divisions:       numOps,
	}

	return info
}

func init() {
	dlperf.Register(&Softmax{})
}
