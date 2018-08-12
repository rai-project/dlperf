package layer

import (
	"github.com/rai-project/dlperf"
)

type Scale struct {
	Base              `json:",inline,flatten,omitempty"`
	inputs            []string  `json:",inputs,omitempty"`
	outputs           []string  `json:",outputs,omitempty"`
	InputsDimensions  [][]int64 `json:"inputs_dimensions,omitempty"`
	OutputsDimensions [][]int64 `json:"outputs_dimensions,omitempty"`
}

func (Scale) OperatorType() string {
	return "Scale"
}

func (Scale) Description() string {
	return ``
}

func (c *Scale) Information() dlperf.LayerInformation {
	checkNumber(c.InputsDimensions, []int{1}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputsDimensions, []int{1}, c.OperatorType(), "number of outputs")

	inputDimensions := c.InputsDimensions[0]   // (N x C x H x W)
	outputDimensions := c.OutputsDimensions[0] // (N x C x H x W)

	nIn := inputDimensions[0]
	cIn := inputDimensions[1]
	hIn := inputDimensions[2]
	wIn := inputDimensions[3]

	flops := dlperf.FlopsInformation{
		MultiplyAdds: wIn * hIn * cIn * nIn,
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
	dlperf.Register(&Scale{})
}
