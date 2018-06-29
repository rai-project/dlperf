package layer

import (
	"github.com/rai-project/dlperf"
)

type Concat struct {
	Base `json:",inline,flatten,omitempty"`
}

func (Concat) OperatorType() string {
	return "Concat"
}

func (Concat) Description() string {
	return ``
}

func (c *Concat) LayerInformation() dlperf.LayerInfo {
	checkNumber(c.OutputsDimensions, []int{1}, c.OperatorType(), "number of outputs")

	inputDimensions := c.InputsDimensions[0]   // (N x C x H x W)
	outputDimensions := c.OutputsDimensions[0] // (N x C x H x W)

	flops := dlperf.FlopsInformation{}

	return &Information{
		name:             c.name,
		operatorType:     c.OperatorType(),
		flops:            flops,
		inputDimensions:  inputDimensions,
		outputDimensions: outputDimensions,
	}
}

func init() {
	dlperf.Register(&Concat{})
}
