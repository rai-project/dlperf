package layer

import (
	"fmt"

	"github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/dlperf/pkg/benchmark"
)

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#BatchNormalization

type BatchNorm struct {
	*Base   `json:",inline,flatten,omitempty"`
	Spatial int64 `json:"patial,omitempty"`
}

func (BatchNorm) OperatorType() string {
	return "BatchNorm"
}

func (BatchNorm) Description() string {
	return ``
}

func (c *BatchNorm) InferShape(inputLayers []dlperf.Layer) {
	inputShapes := getOutputShapes(inputLayers)
	c.SetInputShapes(inputShapes)
	c.SetOutputShapes(inputShapes)
}

func (c BatchNorm) FwdBenchmarkName() string {
	return "LAYER_CUDNN_BATCHNORM_FWD_INFERENCE"
}

func (c BatchNorm) FwdBenchmarkArgs() []string {
	return []string{""}
}

func (c BatchNorm) FwdCUDNNName() string {
	return ""
}

func (c BatchNorm) FwdTiming(system string /* hardware/software struct */) string {
	return ""
}

func (c BatchNorm) FwdBenchmarkAlgorithms() []string {
	if c.Spatial == int64(1) {
		return []string{
			"CUDNN_BATCHNORM_SPATIAL",
			"CUDNN_BATCHNORM_SPATIAL_PERSISTENT",
		}
	} else {
		return []string{
			"CUDNN_BATCHNORM_PER_ACTIVATION",
		}
	}
}

func (c BatchNorm) FwdBenchmarkFilter(datatype, algorithm string) benchmark.Benchmark {
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

func (c BatchNorm) Shape() dlperf.ShapeInformation {
	return c.Information().Shape()
}

func (c BatchNorm) Information() dlperf.LayerInformation {
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

	checkNumber(c.InputShapes, []int{5}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputShapes, []int{1, 2, 3, 4, 5}, c.OperatorType(), "number of outputs")

	inputShapes := c.inputShapes[0]

	numOps := int64(1)
	for _, s := range inputShapes {
		numOps *= s
	}

	// this is for inference
	info.flops = dlperf.FlopsInformation{
		Additions: numOps,
		Divisions: numOps,
	}

	return info
}

func init() {
	dlperf.Register(&BatchNorm{})
}
