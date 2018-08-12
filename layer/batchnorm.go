package layer

import (
	"github.com/rai-project/dlperf"
)

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#BatchNormalization
// https://arxiv.org/pdf/1502.03167.pdf

type BatchNorm struct {
	Base `json:",inline,flatten,omitempty"`
}

func (BatchNorm) Description() string {
	return ``
}

func (c BatchNorm) Information() dlperf.LayerInformation {
	info := &Information{
		Base: c.Base,
	}

	if len(c.OutputsDimensions()) == 0 {
		log.WithField("layer", c.OperatorType()).Info("len(OutputsDimensions) is 0")
		return info
	}

	checkNumber(c.InputsDimensions, []int{1, 2, 3, 4, 5}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputsDimensions, []int{1, 2, 3, 4, 5}, c.OperatorType(), "number of outputs")

	inputDimensions := c.inputsDimensions[0]   // (N x C x H x W)
	outputDimensions := c.outputsDimensions[0] // (N x C x H x W)

	nIn := inputDimensions[0]
	cIn := inputDimensions[1]
	hIn := inputDimensions[2]
	wIn := inputDimensions[3]

	numOps := wIn * hIn * cIn * nIn

	// this is for inference
	info.flops = dlperf.FlopsInformation{
		Additions: numOps,
		Divisions: numOps,
	}

	info.shape = dlperf.ShapeInformation{
		InputDimensions:  inputDimensions,
		OutputDimensions: outputDimensions,
	}

	return info
}

func init() {
	dlperf.Register(&BatchNorm{})
}
