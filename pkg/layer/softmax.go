package layer

import (
	"github.com/rai-project/dlperf"
)

type SoftMax struct {
	Base `json:",inline,flatten,omitempty"`
}

func (SoftMax) Description() string {
	return ``
}

func (c SoftMax) Information() dlperf.LayerInformation {
	info := &Information{
		Base: c.Base,
	}

	if isAnyEmpty(c.outputsDimensions) {
		log.WithField("layer", c.OperatorType()).Info("len(OutputsDimensions) is 0")
		return info
	}

	if isAnyEmpty(c.inputsDimensions) {
		log.WithField("layer", c.OperatorType()).Info("len(InputDimensions) is 0")
		return info
	}

	checkNumber(c.InputsDimensions, []int{1}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputsDimensions, []int{1}, c.OperatorType(), "number of outputs")

	inputDimensions := c.InputsDimensions()[0]   // (N x C x H x W)
	outputDimensions := c.OutputsDimensions()[0] // (N x C x H x W)

	var shape []int64
	for _, s := range inputDimensions {
		shape = append(shape, s)
	}

	numOps := int64(1)
	for _, s := range shape {
		numOps *= s
	}

	info.flops = dlperf.FlopsInformation{
		Exponentiations: numOps,
		Additions:       numOps,
		Divisions:       numOps,
	}

	info.shape = dlperf.ShapeInformation{
		InputDimensions:  inputDimensions,
		OutputDimensions: outputDimensions,
	}

	return info
}

func init() {
	dlperf.Register(&SoftMax{})
}
