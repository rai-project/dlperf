package layer

import (
	"github.com/mitchellh/hashstructure"
	dlperf "github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/dlperf/pkg/benchmark"
)

//easyjson:json
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

func (c Softmax) FwdBenchmarkName(opts ...dlperf.FwdBenchmarkArgsOptionFunc) string {
	return "LAYER_CUDNN_SOFTMAX_FWD"
}

func (c Softmax) BwdBenchmarkName(opts ...dlperf.BwdBenchmarkArgsOptionFunc) string {
	return "LAYER_CUDNN_SOFTMAX_BWD"
}

func (c Softmax) FwdCUDNNName() string {
	return ""
}

func (c Softmax) BwdCUDNNName() string {
	return ""
}

func (c Softmax) FwdTiming(system string /* hardware/software struct */) string {
	return ""
}

func (c Softmax) BwdTiming(system string /* hardware/software struct */) string {
	return ""
}

func (c Softmax) FwdBenchmarkAlgorithms(...dlperf.FwdBenchmarkArgsOptionFunc) []string {
	return c.BenchmarkAlgorithms()
}

func (c Softmax) BwdBenchmarkAlgorithms(...dlperf.BwdBenchmarkArgsOptionFunc) []string {
	return c.BenchmarkAlgorithms()
}

func (c Softmax) BenchmarkAlgorithms() []string {
	return []string{
		"CUDNN_SOFTMAX_MODE_INSTANCE",
		"CUDNN_SOFTMAX_MODE_CHANNEL",
	}
}

func (c Softmax) Shape() dlperf.ShapeInformation {
	return c.Information().Shape()
}

type softmaxBenchmarkArgs struct {
	BaseBenchmarkArgs
	BaseBenchmarkInputArgs
}

func (c Softmax) FwdBenchmarkArgs(opts ...dlperf.FwdBenchmarkArgsOptionFunc) interface{} {

	res := softmaxBenchmarkArgs{
		BaseBenchmarkInputArgs: mkBaseBenchmarkInputArgs(&c),
		BaseBenchmarkArgs:      mkBaseBenchmarkFWDArgs(&c, opts...),
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

func (c Softmax) BwdBenchmarkArgs(opts ...dlperf.BwdBenchmarkArgsOptionFunc) interface{} {

	res := softmaxBenchmarkArgs{
		BaseBenchmarkInputArgs: mkBaseBenchmarkInputArgs(&c),
		BaseBenchmarkArgs:      mkBaseBenchmarkBWDArgs(&c, opts...),
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

func (c Softmax) FwdBenchmarkFilter(datatype, algorithm string, opts ...dlperf.FwdBenchmarkArgsOptionFunc) benchmark.Benchmark {
	var alg string
	if algorithm == "" {
		alg = "CUDNN_SOFTMAX_FAST"
	} else {
		alg = "CUDNN_SOFTMAX_FAST, " + algorithm
	}
	return benchmark.Benchmark{
		Name:       mkFwdBenchmarkFilterName(&c, datatype, alg),
		Attributes: benchmarkAttributes(c.FwdBenchmarkArgs(opts...)),
	}
}

func (c Softmax) BwdBenchmarkFilter(datatype, algorithm string, opts ...dlperf.BwdBenchmarkArgsOptionFunc) benchmark.Benchmark {
	var alg string
	if algorithm == "" {
		alg = "CUDNN_SOFTMAX_FAST"
	} else {
		alg = "CUDNN_SOFTMAX_FAST, " + algorithm
	}
	return benchmark.Benchmark{
		Name:       mkBwdBenchmarkFilterName(&c, datatype, alg),
		Attributes: benchmarkAttributes(c.BwdBenchmarkArgs(opts...)),
	}
}

func (c Softmax) FwdBenchmarkGenerator() string {
	templString := _escFSMustString(false, "/scope/softmax.tmpl")
	return templateExecFWD(&c, templateBasePrefix+templString)
}

func (c Softmax) BwdBenchmarkGenerator() string {
	templString := _escFSMustString(false, "/scope/softmax.tmpl")
	return templateExecBWD(&c, templateBasePrefix+templString)
}

func (c Softmax) FwdBenchmarkGeneratorArgNames() []string {
	return benchmarkArgNames(softmaxBenchmarkArgs{})
}

func (c Softmax) BwdBenchmarkGeneratorArgNames() []string {
	return benchmarkArgNames(softmaxBenchmarkArgs{})
}

func (c Softmax) Information() dlperf.LayerInformation {
	info := &Information{
		Base: c.Base,
		shape: dlperf.ShapeInformation{
			InputShapes:  c.InputShapes(),
			OutputShapes: c.OutputShapes(),
		},
	}

	if isAnyEmpty(c.InputShapes()) {
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
