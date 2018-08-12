package layer

import (
	"github.com/rai-project/dlperf"
)

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Dropout

type Dropout Base

func (Dropout) OperatorType() string {
	return "Dropout"
}

func (Dropout) Description() string {
	return ``
}

func (c *Dropout) Information() dlperf.LayerInformation {
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

	checkNumber(c.InputsDimensions, []int{1}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputsDimensions, []int{1, 2}, c.OperatorType(), "number of outputs")

	inputDimensions := c.InputsDimensions[0]   // (N x C x H x W)
	outputDimensions := c.OutputsDimensions[0] // (N x C x H x W)

	nIn := inputDimensions[0]
	cIn := inputDimensions[1]
	hIn := inputDimensions[2]
	wIn := inputDimensions[3]

	info.flops = dlperf.FlopsInformation{
		Comparisons: wIn * hIn * cIn * nIn,
	}

	return info
}

func init() {
	dlperf.Register(&Dropout{})
}
