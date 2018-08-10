package layer

import (
	"github.com/rai-project/dlperf"
)

type SoftMax struct {
	Base              `json:",inline,flatten,omitempty"`
	InputsDimensions  [][]int64 `json:"inputs_dimensions,omitempty"`
	OutputsDimensions [][]int64 `json:"outputs_dimensions,omitempty"`
}

func (SoftMax) OperatorType() string {
	return "SoftMax"
}

func (SoftMax) Description() string {
	return ``
}

func (c *SoftMax) LayerInformation() dlperf.LayerInfo {
	checkNumber(c.InputsDimensions, []int{1}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputsDimensions, []int{1}, c.OperatorType(), "number of outputs")

	inputDimensions := c.InputsDimensions[0]   // (N x C x H x W)
	outputDimensions := c.OutputsDimensions[0] // (N x C x H x W)

	nIn := inputDimensions[0]
	cIn := inputDimensions[1]
	hIn := inputDimensions[2]
	wIn := inputDimensions[3]

	numOps := wIn * hIn * cIn * nIn
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
