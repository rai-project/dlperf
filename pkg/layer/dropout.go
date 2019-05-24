package layer

import (
	"github.com/mitchellh/hashstructure"
	"github.com/rai-project/dlperf/pkg"
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

func (c Dropout) FwdBenchmarkName() string {
	return "LAYER_CUDNN_DROPOUT_FWD"
}

func (c Dropout) FwdCUDNNName() string {
	return ""
}

func (c Dropout) FwdTiming(system string /* hardware/software struct */) string {
	return ""
}

func (c Dropout) FwdBenchmarkAlgorithms() []string {
	return []string{
		"",
	}
}

type dropoutBenchmarkArgs struct {
	BaseBenchmarkArgs
	BaseBenchmarkInputArgs
}

func (c Dropout) FwdBenchmarkGeneratorArgNames() []string {
	return benchmarkArgNames(reluBenchmarkArgs{})
}

func (c Dropout) FwdBenchmarkArgs() interface{} {

	res := dropoutBenchmarkArgs{
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
		panic(err)
	}
	res.UniqueBenchmarkID = hash

	return res
}

func (c Dropout) FwdBenchmarkFilter(datatype, algorithm string) benchmark.Benchmark {
	if algorithm == "" {
		algorithm = c.FwdBenchmarkAlgorithms()[0]
	}
	return benchmark.Benchmark{
		Name:       mkBenchmarkFilterName(&c, datatype, algorithm),
		Attributes: benchmarkAttributes(c.FwdBenchmarkArgs()),
	}
}

func (c Dropout) FwdBenchmarkGenerator(standAloneGenerate bool) string {
	//templString := _escFSMustString(dlperf.IsDebug, "/scope/dropout.tmpl")
	//return templateExec(&c, templateBasePrefix+templString)
	return ""
}

func (c Dropout) FwdBenchmarkGeneratorPrefix(standAloneGenerate bool) string {
	panic("error")
}

func (c Dropout) FwdBenchmarkGeneratorSuffix(standAloneGenerate bool) string {
	panic("error")
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
