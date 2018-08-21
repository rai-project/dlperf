package layer

import (
	"strings"

	"github.com/rai-project/dlperf/pkg"
)

type ElementWise struct {
	*Base `json:",inline,flatten,omitempty"`
}

func (ElementWise) OperatorType() string {
	return "ElementWise"
}

func (ElementWise) Description() string {
	return ``
}

func (c *ElementWise) InferShape(inputLayers dlperf.Layers) {
	inputShapes := getOutputShapes(inputLayers)
	outputShapes := []dlperf.Shape{inputShapes[0]}
	c.SetOutputShapes(outputShapes)
}

func (c ElementWise) Information() dlperf.LayerInformation {
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
	switch strings.ToLower(c.onnxOperatorType) {
	case "add", "sum", "sub":
		flops.Additions = numOps * (numInputs - 1)
	case "mul", "div":
		flops.MultiplyAdds = numOps
	case "max,", "min":
		flops.Comparisons = numOps * (numInputs - 1)
	default:
		log.WithField("layer", c.OperatorType()).WithField("operator", c.OperatorType()).Error("invalid layer operation")
	}

	info.flops = flops

	return info
}

func init() {
	dlperf.Register(&ElementWise{})
}
