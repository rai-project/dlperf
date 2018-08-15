package layer

import (
	"strings"

	"github.com/rai-project/dlperf/pkg"
)

type ElementWise struct {
	Base `json:",inline,flatten,omitempty"`
}

func (ElementWise) Description() string {
	return ``
}

func (c *ElementWise) InferShape(inputLayers []dlperf.Layer) {
	//c.inputdimensions =  dlperf.ShapeInformation{}
}

func (c ElementWise) Information() dlperf.LayerInformation {
	info := &Information{
		Base: c.Base,
	}

	if isAnyEmpty(c.outputsDimensions) {
		log.WithField("layer", c.OperatorType()).Info("len(OutputsDimensions) is 0")
		return info
	}

	checkNumber(c.InputsDimensions, []int{2}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputsDimensions, []int{1}, c.OperatorType(), "number of outputs")

	inputDimensions := c.InputsDimensions()[0]   // (N x C x H x W)
	outputDimensions := c.OutputsDimensions()[0] // (N x C x H x W)

	var shape []int64
	for _, s := range inputDimensions {
		shape = append(shape, s)
	}

	numOps := int64(1)
	for _, s := range shape {
		numOps *= s
	}

	flops := dlperf.FlopsInformation{}
	switch strings.ToLower(c.operatorType) {
	case "add", "sum", "sub":
		flops.Additions = numOps
	case "mul", "div":
		flops.MultiplyAdds = numOps
	case "max,", "min":
		flops.Comparisons = numOps
	default:
		log.WithField("layer", c.OperatorType()).WithField("operator", c.OperatorType()).Error("invalid layer operation")
	}

	info.flops = flops

	info.shape = dlperf.ShapeInformation{
		InputDimensions:  inputDimensions,
		OutputDimensions: outputDimensions,
	}

	return info
}

func init() {
	dlperf.Register(&ElementWise{})
}
