package layer

import (
	"fmt"

	"github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/dlperf/pkg/benchmark"
)

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Identity

type Identity struct {
	*Base `json:",inline,flatten,omitempty"`
}

func (Identity) OperatorType() string {
	return "Identity"
}

func (Identity) Description() string {
	return ``
}

func (c *Identity) InferShape(inputLayers []dlperf.Layer) {
	inputShapes := getOutputShapes(inputLayers)
	c.SetInputShapes(inputShapes)
	c.SetOutputShapes(inputShapes)
}

func (c Identity) FwdBenchmarkName() string {
	return "LAYER_CUDNN_ACTIVATION_FWD"
}

func (c Identity) FwdBenchmarkArgs() []string {
	return []string{""}
}

func (c Identity) FwdCUDNNName() string {
	return ""
}

func (c Identity) FwdTiming(system string /* hardware/software struct */) string {
	return ""
}

func (c Identity) FwdBenchmarkAlgorithms() []string {
	return []string{
		"CUDNN_ACTIVATION_IDENTITY",
	}
}

func (c Identity) FwdBenchmarkFilter(datatype, algorithm string) benchmark.Benchmark {
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

func (c Identity) Shape() dlperf.ShapeInformation {
	return c.Information().Shape()
}

func (c Identity) Information() dlperf.LayerInformation {
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

	info.flops = dlperf.FlopsInformation{}

	return info
}

func init() {
	dlperf.Register(&Identity{})
}
