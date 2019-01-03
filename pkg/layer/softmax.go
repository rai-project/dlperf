package layer

import (
	"github.com/mitchellh/hashstructure"
	"github.com/rai-project/dlperf/pkg"
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
	BaseBenchmarkArgs
	BaseBenchmarkInputArgs
}

func (c Softmax) FwdBenchmarkGeneratorArgNames() []string {
	return benchmarkArgNames(softmaxBenchmarkArgs{})
}

func (c Softmax) FwdBenchmarkArgs() interface{} {

	res := softmaxBenchmarkArgs{
		BaseBenchmarkInputArgs: mkBaseBenchmarkInputArgs(&c),
		BaseBenchmarkArgs:      mkBaseBenchmarkArgs(&c),
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

func (c Softmax) FwdBenchmarkFilter(datatype, algorithm string) benchmark.Benchmark {
	if algorithm == "" {
		algorithm = c.FwdBenchmarkAlgorithms()[0]
	}
	return benchmark.Benchmark{
		Name:       mkBenchmarkFilterName(&c, datatype, "CUDNN_SOFTMAX_FAST, "+algorithm),
		Attributes: benchmarkAttributes(c.FwdBenchmarkArgs()),
	}
}

func (c Softmax) FwdBenchmarkGenerator() string {
	templString := _escFSMustString(false, "/scope/softmax.tmpl")

	return templateExec(&c, templateBasePrefix+templString)
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
