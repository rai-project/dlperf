package layer

import (
	"github.com/rai-project/dlperf"
)

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#BatchNormalization
// https://arxiv.org/pdf/1502.03167.pdf

type BatchNorm struct {
	Base `json:",inline,flatten,omitempty"`
}

func (BatchNorm) OperatorType() string {
	return "BatchNorm"
}

func (BatchNorm) Description() string {
	return ``
}

func (c *BatchNorm) LayerInformation() dlperf.LayerInfo {
	checkNumber(c.InputsDimensions, []int{1, 2, 3, 4, 5}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputsDimensions, []int{1, 2, 3, 4, 5}, c.OperatorType(), "number of outputs")

	inputDimensions := c.InputsDimensions[0]   // (N x C x H x W)
	outputDimensions := c.OutputsDimensions[0] // (N x C x H x W)

	nIn := inputDimensions[0]
	cIn := inputDimensions[1]
	hIn := inputDimensions[2]
	wIn := inputDimensions[3]

	numOps := wIn * hIn * cIn * nIn

	// this is for inference
	flops := dlperf.FlopsInformation{
		Additions: numOps,
		Divisions: numOps,
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
	dlperf.Register(&BatchNorm{})
}
