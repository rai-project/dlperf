package layer

import (
	"fmt"

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
	return "LAYER_CUDNN_ACTIVATION_FWD"
}

func (c Softmax) FwdBenchmarkArgs() []string {
	return []string{""}
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

func (c Softmax) FwdBenchmarkFilter(datatype, algorithm string) benchmark.Benchmark {
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
