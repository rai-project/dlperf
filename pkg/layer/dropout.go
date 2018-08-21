package layer

import (
	"fmt"

	"github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/dlperf/pkg/benchmark"
)

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Dropout

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

func (c Dropout) FwdBenchmarkArgs() []string {
	return []string{""}
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

func (c Dropout) FwdBenchmarkFilter(datatype, algorithm string) benchmark.Benchmark {
	if algorithm == "" {
		algorithm = c.FwdBenchmarkAlgorithms()[0]
	}
	attrs := map[string]interface{}{}
	for ii, dim := range c.InputShapes()[0] {
		attrs[fmt.Sprintf("input[%d]", ii)] = dim
	}
	return benchmark.Benchmark{
		Name:       mkBenchmarkFilterName(&c, datatype, algorithm),
		Attributes: attrs,
	}
}

func (c Dropout) Shape() dlperf.ShapeInformation {
	return c.Information().Shape()
}

func (c Dropout) Information() dlperf.LayerInformation {
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
