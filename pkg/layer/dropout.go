package layer

import (
	"github.com/rai-project/dlperf/pkg"
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

func (c *Dropout) InferShape(inputLayers []dlperf.Layer) {
	inputShapes := getOutputShapes(inputLayers)
	c.SetInputShapes(inputShapes)
	c.SetOutputShapes(inputShapes)
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
