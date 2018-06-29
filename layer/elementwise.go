package layer

import (
	"strings"

	"github.com/rai-project/dlperf"
)

type ElementWise struct {
	Base     `json:",inline,flatten,omitempty"`
	Operator string `json:"operation,omitempty"`
}

func (c *ElementWise) OperatorType() string {
	return c.Operator
}

func (ElementWise) Description() string {
	return ``
}

func (c *ElementWise) LayerInformation() dlperf.LayerInfo {
	checkNumber(c.InputsDimensions, []int{1, 2}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputsDimensions, []int{1}, c.OperatorType(), "number of outputs")

	inputDimensions := c.InputsDimensions[0]   // (N x C x H x W)
	outputDimensions := c.OutputsDimensions[0] // (N x C x H x W)

	nIn := inputDimensions[0]
	cIn := inputDimensions[1]
	hIn := inputDimensions[2]
	wIn := inputDimensions[3]

	numOps := wIn * hIn * cIn * nIn

	flops := dlperf.FlopsInformation{}
	switch strings.ToLower(c.Operator) {
	case "add", "sum", "sub":
		flops.Additions = numOps
	case "mul", "div":
		flops.MultiplyAdds = numOps
	case "max,", "min":
		flops.Comparisons = numOps
	default:
		log.WithField("layer", c.OperatorType()).WithField("operator", c.Operator).Error("invalid layer operation")
	}

	return &Information{
		name:             c.name,
		operatorType:     c.OperatorType(),
		flops:            flops,
		inputDimensions:  inputDimensions,
		outputDimensions: outputDimensions,
	}
}

func init() {
	dlperf.Register(&ElementWise{})
}
