package layer

import (
	"github.com/rai-project/dlperf"
)

type Concat Base

func (Concat) OperatorType() string {
	return "Concat"
}

func (Concat) Description() string {
	return ``
}

func (c *Concat) Information() dlperf.LayerInformation {
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

	checkNumber(c.OutputsDimensions, []int{1}, c.OperatorType(), "number of outputs")

	inputDimensions := c.InputsDimensions[0]   // (N x C x H x W)
	outputDimensions := c.OutputsDimensions[0] // (N x C x H x W)

	info.flops = dlperf.FlopsInformation{}

	return info
}

func init() {
	dlperf.Register(&Concat{})
}
