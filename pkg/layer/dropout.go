package layer

import (
	"github.com/mitchellh/hashstructure"
	dlperf "github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/dlperf/pkg/benchmark"
)

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Dropout

//easyjson:json
type Dropout struct {
	*Base `json:",inline,flatten,omitempty"`
}

func (Dropout) OperatorType() string {
	return "Dropout"
}

func (Dropout) Description() string {
	return ``
}

func (c *Dropout) InferShape(inputLayers dlperf.Layers) {
	inputShapes := getOutputShapes(inputLayers)
	c.SetInputShapes(inputShapes)
	c.SetOutputShapes(inputShapes)
}

func (c Dropout) FwdBenchmarkName(opts ...dlperf.FwdBenchmarkArgsOptionFunc) string {
	return "LAYER_CUDNN_DROPOUT_FWD"
}

func (c Dropout) BwdBenchmarkName(opts ...dlperf.BwdBenchmarkArgsOptionFunc) string {
	return "LAYER_CUDNN_DROPOUT_BWD"
}

func (c Dropout) FwdCUDNNName() string {
	return ""
}

func (c Dropout) BwdCUDNNName() string {
	return ""
}

func (c Dropout) FwdTiming(system string /* hardware/software struct */) string {
	return ""
}

func (c Dropout) BwdTiming(system string /* hardware/software struct */) string {
	return ""
}

func (c Dropout) FwdBenchmarkAlgorithms(...dlperf.FwdBenchmarkArgsOptionFunc) []string {
	return []string{
		"",
	}
}

func (c Dropout) BwdBenchmarkAlgorithms(...dlperf.BwdBenchmarkArgsOptionFunc) []string {
	return []string{
		"",
	}
}

type dropoutBenchmarkArgs struct {
	BaseBenchmarkArgs
	BaseBenchmarkInputArgs
	BatchSize int64
}

func (c Dropout) FwdBenchmarkArgs(opts ...dlperf.FwdBenchmarkArgsOptionFunc) interface{} {
	res := dropoutBenchmarkArgs{
		BaseBenchmarkInputArgs: mkBaseBenchmarkInputArgs(&c),
		BaseBenchmarkArgs:      mkBaseBenchmarkFWDArgs(&c, opts...),
		BatchSize: dlperf.GetBatchSize(),
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

func (c Dropout) BwdBenchmarkArgs(opts ...dlperf.BwdBenchmarkArgsOptionFunc) interface{} {
	res := dropoutBenchmarkArgs{
		BaseBenchmarkInputArgs: mkBaseBenchmarkInputArgs(&c),
		BaseBenchmarkArgs:      mkBaseBenchmarkBWDArgs(&c, opts...),
		BatchSize: dlperf.GetBatchSize(),
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

func (c Dropout) FwdBenchmarkFilter(datatype, algorithm string, opts ...dlperf.FwdBenchmarkArgsOptionFunc) benchmark.Benchmark {
	return benchmark.Benchmark{
		Name:       mkFwdBenchmarkFilterName(&c, datatype, algorithm),
		Attributes: benchmarkAttributes(c.FwdBenchmarkArgs()),
	}
}

func (c Dropout) BwdBenchmarkFilter(datatype, algorithm string, opts ...dlperf.BwdBenchmarkArgsOptionFunc) benchmark.Benchmark {
	return benchmark.Benchmark{
		Name:       mkBwdBenchmarkFilterName(&c, datatype, algorithm),
		Attributes: benchmarkAttributes(c.BwdBenchmarkArgs()),
	}
}

func (c Dropout) FwdBenchmarkGenerator(opts ...dlperf.FwdBenchmarkArgsOptionFunc) string {
	templString := _escFSMustString(false, "/scope/dropout.tmpl")
	return templateExecFWD(&c, templateBasePrefix+templString)
}

func (c Dropout) BwdBenchmarkGenerator(opts ...dlperf.BwdBenchmarkArgsOptionFunc) string {
	templString := _escFSMustString(false, "/scope/dropout.tmpl")
	return templateExecBWD(&c, templateBasePrefix+templString)
}

func (c Dropout) FwdBenchmarkGeneratorArgNames() []string {
	return benchmarkArgNames(dropoutBenchmarkArgs{})
}

func (c Dropout) BwdBenchmarkGeneratorArgNames() []string {
	return benchmarkArgNames(dropoutBenchmarkArgs{})
}

func (c Dropout) Shape() dlperf.ShapeInformation {
	return c.Information().Shape()
}

func (c Dropout) Information() dlperf.LayerInformation {
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

	checkNumber(c.InputShapes, []int{1}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputShapes, []int{1, 2}, c.OperatorType(), "number of outputs")

	inputShapes := c.InputShapes()[0] // (N x C x ...)

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
	dlperf.Register(&Dropout{})
}
