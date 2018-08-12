package layer

import (
	"strings"

	"github.com/rai-project/dlperf"
)

type ElementWise struct {
	Base              `json:",inline,flatten,omitempty"`
	inputs            []string  `json:",inputs,omitempty"`
	outputs           []string  `json:",outputs,omitempty"`
	Operator          string    `json:"operation,omitempty"`
	InputsDimensions  [][]int64 `json:"inputs_dimensions,omitempty"`
	OutputsDimensions [][]int64 `json:"outputs_dimensions,omitempty"`
}

func (c *ElementWise) OperatorType() string {
	return c.Operator
}

func (ElementWise) Description() string {
	return ``
}

func (c *ElementWise) LayerInformation() dlperf.LayerInfo {
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
