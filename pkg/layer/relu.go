package layer

import (
	"fmt"
	"strings"

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
	switch strings.ToLower(c.onnxOperatorType) {
	case "sigmoid":
		return []string{
			"CUDNN_ACTIVATION_SIGMOID",
		}
	case "tanh":
		return []string{
			"CUDNN_ACTIVATION_TANH",
		}
	case "relu":
		return []string{
			"CUDNN_ACTIVATION_RELU",
			"CUDNN_ACTIVATION_CLIPPED_RELU",
		}
	case "elu":
		return []string{
			"CUDNN_ACTIVATION_ELU",
		}
	case "identity":
		return []string{
			"CUDNN_ACTIVATION_IDENTITY",
		}
	}

	return nil
}

func (c Relu) FwdBenchmarkFilter(datatype, algorithm string) benchmark.Benchmark {
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

	inputShapes := c.InputShapes()[0]

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
