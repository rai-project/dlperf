package layer

import (
	"github.com/rai-project/dlperf"
)

type MatMul Base

func (MatMul) Description() string {
	return ``
}

func (c *MatMul) Information() dlperf.LayerInformation {
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

	checkNumber(c.InputsDimensions, []int{1, 2}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputsDimensions, []int{1}, c.OperatorType(), "number of outputs")

	inputADimensions := c.InputsDimensions[0]  // (N x C x H x W)
	inputBDimensions := c.InputsDimensions[1]  // (N x C x H x W)
	outputDimensions := c.OutputsDimensions[0] // (N x C x H x W)

	var numOps int64
	dimLen := len(inputADimensions)
	if dimLen == 2 {
		numOps = inputADimensions[0] * inputADimensions[1] * inputBDimensions[1]
	} else if dimLen == 3 {
		numOps = inputADimensions[0] * inputADimensions[1] * inputADimensions[2] * inputBDimensions[2]

	} else {
		numOps = inputADimensions[0] * inputADimensions[1] * inputADimensions[2] * inputADimensions[3] * inputBDimensions[3]
	}

	info.flops = dlperf.FlopsInformation{
		MultiplyAdds: numOps,
	}

	return info
}

func init() {
	dlperf.Register(&MatMul{})
}
