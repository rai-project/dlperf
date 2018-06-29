package layer

import (
	"github.com/rai-project/dlperf"
)

type MatMul struct {
	Base `json:",inline,flatten,omitempty"`
}

func (MatMul) OperatorType() string {
	return "MatMul"
}

func (MatMul) Description() string {
	return ``
}

func (c *MatMul) LayerInformation() dlperf.LayerInfo {
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

	flops := dlperf.FlopsInformation{
		MultiplyAdds: numOps,
	}

	return &Information{
		name:             c.name,
		operatorType:     c.OperatorType(),
		flops:            flops,
		inputDimensions:  inputADimensions,
		outputDimensions: outputDimensions,
	}
}

func init() {
	dlperf.Register(&MatMul{})
}
