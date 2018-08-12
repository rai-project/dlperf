package layer

import (
	"github.com/rai-project/dlperf"
)

type SoftMax struct {
  Base              `json:",inline,flatten,omitempty"`
  inputs            []string  `json:",inputs,omitempty"`
	outputs           []string  `json:",outputs,omitempty"`s
	InputsDimensions  [][]int64 `json:"inputs_dimensions,omitempty"`
	OutputsDimensions [][]int64 `json:"outputs_dimensions,omitempty"`
}

func (SoftMax) OperatorType() string {
	return "SoftMax"
}

func (SoftMax) Description() string {
	return ``
}

func (c *SoftMax) Information() dlperf.LayerInformation {
	checkNumber(c.InputsDimensions, []int{1}, c.OperatorType(), "number of inputs")
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

	flops := dlperf.FlopsInformation{
		Exponentiations: numOps,
		Additions:       numOps,
		Divisions:       numOps,
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
	dlperf.Register(&SoftMax{})
}
