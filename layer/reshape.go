package layer

import (
	"github.com/rai-project/dlperf"
)

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reshape

type Reshape struct {
	Base `json:",inline,flatten,omitempty"`
}

func (Reshape) OperatorType() string {
	return "Reshape"
}

func (Reshape) Description() string {
	return ``
}

func (c *Reshape) LayerInformation() dlperf.LayerInfo {
	checkNumber(c.InputsDimensions, []int{1, 2}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputsDimensions, []int{1}, c.OperatorType(), "number of outputs")

	inputDimensions := c.InputsDimensions[0]   // (N x C x H x W)
	outputDimensions := c.OutputsDimensions[0] // (N x C x H x W)

	// this is for inference
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
	dlperf.Register(&Reshape{})
}
