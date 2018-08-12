package layer

import (
	"strings"

	"github.com/rai-project/dlperf"
)

type ElementWise Base

func (ElementWise) Description() string {
	return ``
}

func (c *ElementWise) Information() dlperf.LayerInformation {
	info := &Information{
		name:              c.Name,
		operatorType:      c.OperatorType(),
		inputs:            c.Inputs(),
		outputs:           c.Outputs(),
		inputsDimensions:  c.InputsDimensions,
		outputsDimensions: c.OutputsDimensions,
	}

	if len(c.OutputsDimensions) == 0 {
		log.WithField("layer", c.OperatorType()).Info("len(OutputsDimensions) is 0")
		return info
	}

	checkNumber(c.InputsDimensions, []int{2}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputsDimensions, []int{1}, c.OperatorType(), "number of outputs")

	inputDimensions := c.InputsDimensions[0]   // (N x C x H x W)
	outputDimensions := c.OutputsDimensions[0] // (N x C x H x W)

	var shape []int64
	for _, s := range inputDimensions {
		shape = append(shape, s)
	}

	numOps := int64(1)
	for _, s := range shape {
		numOps *= s
	}

	info.flops = dlperf.FlopsInformation{}
	switch strings.ToLower(c.operatorType) {
	case "add", "sum", "sub":
		flops.Additions = numOps
	case "mul", "div":
		flops.MultiplyAdds = numOps
	case "max,", "min":
		flops.Comparisons = numOps
	default:
		log.WithField("layer", c.OperatorType()).WithField("operator", c.Operator).Error("invalid layer operation")
	}

	return info
}

func init() {
	dlperf.Register(&ElementWise{})
}
