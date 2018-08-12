package layer

import (
	"github.com/rai-project/dlperf"
)

type ReLU struct {
	Base              `json:",inline,flatten""`
	inputs            []string  `json:",inputs,omitempty"`
	outputs           []string  `json:",outputs,omitempty"`
	InputsDimensions  [][]int64 `json:"inputs_dimensions,omitempty"`
	OutputsDimensions [][]int64 `json:"outputs_dimensions,omitempty"`
}

func (ReLU) OperatorType() string {
	return "ReLU"
}

func (ReLU) Aliases() []string {
	return []string{"relu"}
}

func (ReLU) Description() string {
	return ``
}

func (c *ReLU) LayerInformation() dlperf.LayerInfo {

	checkNumber(c.InputsDimensions, []int{1, 2}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputsDimensions, []int{1}, c.OperatorType(), "number of outputs")

	inputDimensions := c.InputsDimensions[0]   // (N x C x H x W)
	outputDimensions := c.OutputsDimensions[0] // (N x C x H x W)

	nIn := inputDimensions[0]
	cIn := inputDimensions[1]
	hIn := inputDimensions[2]
	wIn := inputDimensions[3]

	flops := dlperf.FlopsInformation{
		Comparisons: wIn * hIn * cIn * nIn,
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
	dlperf.Register(&ReLU{})
}
