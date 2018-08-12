package layer

import (
	"github.com/rai-project/dlperf"
)

type Concat struct {
	Base `json:",inline,flatten,omitempty"`
}

func (Concat) Description() string {
	return ``
}

func (c Concat) Information() dlperf.LayerInformation {
	info := &Information{
		Base: c.Base,
	}

	if len(c.OutputsDimensions()) == 0 {
		log.WithField("layer", c.OperatorType()).Info("len(OutputsDimensions) is 0")
		return info
	}

	checkNumber(c.OutputsDimensions, []int{1}, c.OperatorType(), "number of outputs")

	inputDimensions := c.InputsDimensions()[0]   // (N x C x H x W)
	outputDimensions := c.OutputsDimensions()[0] // (N x C x H x W)

	info.flops = dlperf.FlopsInformation{}

	info.shape = dlperf.ShapeInformation{
		InputDimensions:  inputDimensions,
		OutputDimensions: outputDimensions,
	}

	return info
}

func init() {
	dlperf.Register(&Concat{})
}
