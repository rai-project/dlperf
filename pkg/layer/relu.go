package layer

import (
	"fmt"

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

func (c *Relu) InferShape(inputLayers []dlperf.Layer) {
	inputShapes := getOutputShapes(inputLayers)
	c.SetInputShapes(inputShapes)
	c.SetOutputShapes(inputShapes)
}

func (c Relu) FwdBenchmarkName() string {
	return "LAYER_CUDNN_ACTIVATION_FWD"
}

func (c Relu) FwdBenchmarkArgs() []string {
	return []string{""}
}

func (c Relu) FwdCUDNNName() string {
	return ""
}

func (c Relu) FwdTiming(system string /* hardware/software struct */) string {
	return ""
}

func (c Relu) FwdBenchmarkAlgorithms() []string {
	return []string{
		"CUDNN_ACTIVATION_SIGMOID",
		"CUDNN_ACTIVATION_RELU",
		"CUDNN_ACTIVATION_TANH",
		"CUDNN_ACTIVATION_CLIPPED_RELU",
		"CUDNN_ACTIVATION_ELU",
		"CUDNN_ACTIVATION_IDENTITY",
	}
}

func (c Relu) FwdBenchmarkFilter(datatype, algorithm string) benchmark.Benchmark {
	if algorithm == "" {
		algorithm = c.FwdBenchmarkAlgorithms()[1]
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

	inputShapes := c.InputShapes()[0] // (N x C x H x W)

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
