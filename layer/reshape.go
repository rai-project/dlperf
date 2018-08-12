package layer

import (
	"github.com/rai-project/dlperf"
)

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reshape

type Reshape Base

func (Reshape) Description() string {
	return ``
}

func (c *Reshape) Information() dlperf.LayerInformation {
	checkNumber(c.InputsDimensions, []int{1, 2}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputsDimensions, []int{1}, c.OperatorType(), "number of outputs")

	inputDimensions := c.InputsDimensions[0]   // (N x C x H x W)
	outputDimensions := c.OutputsDimensions[0] // (N x C x H x W)

	info.flops = dlperf.FlopsInformation{}

	return info
}

func init() {
	dlperf.Register(&Reshape{})
}
