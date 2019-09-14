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

func (c ElementWise) getOperatorName() string {
	switch strings.ToLower(c.OnnxOperatorType()) {
	case "add", "sum", "sub":
		return "ADD"
	case "mul", "div": // NOT SURE ABOUT DIV
		return "MUL"
	case "min":
		return "MIN"
	case "max":
		return "MAX"
	case "sqrt":
		return "SQRT"
	case "not":
		return "NOT"
	default:
		panic("operator not supported")
	}
}

// multidirectionalBroadcastShapeInference
func (c *ElementWise) InferShape(inputLayers dlperf.Layers) {
	inputShapes := getOutputShapes(inputLayers)
	outputShapes := make([]dlperf.Shape, 1)
	// outputShapes := multidirectionalBroadcastShapeInference(inputShapes) TODO: NOT correct for mul
	outputShapes[0] = inputShapes[0]
	if len(inputShapes) == 2 && (len(inputShapes[1]) > len(inputShapes[0])) {
		outputShapes[0] = inputShapes[1]
	}
	c.SetOutputShapes(outputShapes)
}

func (c ElementWise) FwdBenchmarkName(opts ...dlperf.FwdBenchmarkArgsOptionFunc) string {
	return "LAYER_CUDNN_OP_TENSOR_" + c.getOperatorName() + "_FWD"
}

func (c ElementWise) BwdBenchmarkName(opts ...dlperf.BwdBenchmarkArgsOptionFunc) string {
	return "LAYER_CUDNN_OP_TENSOR_" + c.getOperatorName() + "_BWD"
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
	return []string{}
}

func (c ElementWise) BwdBenchmarkAlgorithms(...dlperf.BwdBenchmarkArgsOptionFunc) []string {
	return []string{}
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
		Operator:               c.getOperatorName(),
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
		Operator:               c.getOperatorName(),
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
