package layer

import (
	"github.com/rai-project/dlperf/pkg"
)

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#BatchNormalization
// https://arxiv.org/pdf/1502.03167.pdf

type BatchNorm struct {
	*Base `json:",inline,flatten,omitempty"`
}

func (BatchNorm) Description() string {
	return ``
}

func (BatchNorm) OperatorType() string {
	return "BatchNorm"
}

func (c *BatchNorm) InferShape(inputLayers []dlperf.Layer) {
	inputShapes := getOutputShapes(inputLayers)
	c.SetInputShapes(inputShapes)
	c.SetOutputShapes(inputShapes)
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
