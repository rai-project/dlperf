package layer

import (
	"strings"

	"github.com/mitchellh/hashstructure"
	dlperf "github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/dlperf/pkg/benchmark"
)

//easyjson:json
type ElementWise struct {
	*Base     `json:",inline,flatten,omitempty"`
	Broadcast int64 `json:"broadcast,omitempty"`
	Axis      int64 `json:"axis,omitempty"`
}

func (ElementWise) OperatorType() string {
	return "ElementWise"
}

func (ElementWise) Description() string {
	return ``
}

// multidirectionalBroadcastShapeInference
func (c *ElementWise) InferShape(inputLayers dlperf.Layers) {
	inputShapes := getOutputShapes(inputLayers)
	// outputShapes := multidirectionalBroadcastShapeInference(inputShapes) TODO: NOT correct for mul
	outputShapes := []dlperf.Shape{inputShapes[0]}
	c.SetOutputShapes(outputShapes)
}

func (c ElementWise) FwdBenchmarkName(iopts ...dlperf.FwdBenchmarkArgsOptionFunc) string {
	return "LAYER_CUDNN_OP_TENSOR_FWD"
}

func (c ElementWise) BwdBenchmarkName(iopts ...dlperf.BwdBenchmarkArgsOptionFunc) string {
	return "LAYER_CUDNN_OP_TENSOR_BWD"
}

func (c ElementWise) FwdCUDNNName() string {
	return ""
}

func (c ElementWise) BwdCUDNNName() string {
	return ""
}

func (c ElementWise) FwdTiming(system string /* hardware/software struct */) string {
	return ""
}

func (c ElementWise) BwdTiming(system string /* hardware/software struct */) string {
	return ""
}

func (c ElementWise) FwdBenchmarkAlgorithms(...dlperf.FwdBenchmarkArgsOptionFunc) []string {
	switch strings.ToLower(c.OnnxOperatorType()) {
	case "add", "sum", "sub":
		return []string{"CUDNN_OP_TENSOR_ADD"}
	case "mul", "div": // NOT SURE ABOUT DIV
		return []string{"CUDNN_OP_TENSOR_MUL"}
	case "min":
		return []string{"CUDNN_OP_TENSOR_MIN"}
	case "max":
		return []string{"CUDNN_OP_TENSOR_MAX"}
	case "sqrt":
		return []string{"CUDNN_OP_TENSOR_SQRT"}
	case "not":
		return []string{"CUDNN_OP_TENSOR_NOT"}
	default:
		panic("operator not supported")
	}
}

func (c ElementWise) BwdBenchmarkAlgorithms(...dlperf.BwdBenchmarkArgsOptionFunc) []string {
	return c.FwdBenchmarkAlgorithms()
}

//easyjson:json
type elementWiseBenchmarkArgs struct {
	BaseBenchmarkArgs      `json:",inline,flatten,omitempty"`
	BaseBenchmarkInputArgs `json:",inline,flatten,omitempty"`
	BatchSize              int64  `json:"batch_size,omitempty"`
	Operator               string `hash:"operator", json:"operator,omitempty"`
}

func (c ElementWise) FwdBenchmarkArgs(opts ...dlperf.FwdBenchmarkArgsOptionFunc) interface{} {
	res := elementWiseBenchmarkArgs{
		BaseBenchmarkInputArgs: mkBaseBenchmarkInputArgs(&c),
		BaseBenchmarkArgs:      mkBaseBenchmarkFWDArgs(&c, opts...),
		BatchSize:              dlperf.GetBatchSize(),
		Operator:               c.FwdBenchmarkAlgorithms()[0],
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

func (c ElementWise) BwdBenchmarkArgs(opts ...dlperf.BwdBenchmarkArgsOptionFunc) interface{} {
	res := elementWiseBenchmarkArgs{
		BaseBenchmarkInputArgs: mkBaseBenchmarkInputArgs(&c),
		BaseBenchmarkArgs:      mkBaseBenchmarkBWDArgs(&c, opts...),
		BatchSize:              dlperf.GetBatchSize(),
		Operator:               c.BwdBenchmarkAlgorithms()[0],
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

func (c ElementWise) FwdBenchmarkFilter(datatype, algorithm string, opts ...dlperf.FwdBenchmarkArgsOptionFunc) benchmark.Benchmark {
	return benchmark.Benchmark{
		Name:       mkFwdBenchmarkFilterName(&c, datatype, algorithm, opts...),
		Attributes: benchmarkAttributes(c.FwdBenchmarkArgs()),
	}
}

func (c ElementWise) BwdBenchmarkFilter(datatype, algorithm string, opts ...dlperf.BwdBenchmarkArgsOptionFunc) benchmark.Benchmark {
	return benchmark.Benchmark{
		Name:       mkBwdBenchmarkFilterName(&c, datatype, algorithm, opts...),
		Attributes: benchmarkAttributes(c.BwdBenchmarkArgs()),
	}
}

func (c ElementWise) FwdBenchmarkGenerator(opts ...dlperf.FwdBenchmarkArgsOptionFunc) string {
	templString := _escFSMustString(false, "/scope/elementwise.tmpl")
	return templateExecFWD(&c, templateBasePrefix+templString+templateBaseSuffix)
}

func (c ElementWise) BwdBenchmarkGenerator(opts ...dlperf.BwdBenchmarkArgsOptionFunc) string {
	templString := _escFSMustString(false, "/scope/elementwise.tmpl")
	return templateExecBWD(&c, templateBasePrefix+templString+templateBaseSuffix)
}

func (c ElementWise) FwdBenchmarkGeneratorArgNames() []string {
	return benchmarkArgNames(elementWiseBenchmarkArgs{})
}

func (c ElementWise) BwdBenchmarkGeneratorArgNames() []string {
	return benchmarkArgNames(elementWiseBenchmarkArgs{})
}

func (c ElementWise) Shape() dlperf.ShapeInformation {
	return c.Information().Shape()
}

func (c ElementWise) Information() dlperf.LayerInformation {
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

	checkNumber(c.InputShapes, nil, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputShapes, []int{1}, c.OperatorType(), "number of outputs")

	numInputs := int64(len(c.InputShapes()))
	inputShapes := c.InputShapes()[0]

	var shape []int64
	for _, s := range inputShapes {
		shape = append(shape, s)
	}

	numOps := int64(1)
	for _, s := range shape {
		numOps *= s
	}

	flops := dlperf.FlopsInformation{}
	switch strings.ToLower(c.OnnxOperatorType()) {
	case "add", "sum", "sub":
		flops.Additions = numOps * (numInputs - 1)
	case "mul", "div":
		flops.MultiplyAdds = numOps
	case "max,", "min":
		flops.Comparisons = numOps * (numInputs - 1)
	case "sqrt":
		flops.General = numOps // NOT SURE, TODO
	case "not":
		flops.General = numOps // NOT SURE, TODO
	default:
		log.WithField("layer", c.OperatorType()).WithField("operator", c.OperatorType()).Error("invalid layer operation")
	}

	info.flops = flops

	return info
}

func init() {
	dlperf.Register(&ElementWise{})
}
