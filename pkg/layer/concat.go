package layer

import (
	"github.com/rai-project/dlperf/pkg"
)

type Concat struct {
	Base `json:",inline,flatten,omitempty"`
}

func (Concat) Description() string {
	return ``
}

func (c *Concat) InferShape(inputLayers ...dlperf.Layer) {
	//c.inputdimensions =  dlperf.ShapeInformation{}
}

func (c Concat) Information() dlperf.LayerInformation {
	info := &Information{
		Base: c.Base,
	}

	if isAnyEmpty(c.outputShapes) {
		log.WithField("layer", c.OperatorType()).Info("len(OutputShapes) is 0")
		return info
	}

	checkNumber(c.OutputShapes, []int{1}, c.OperatorType(), "number of outputs")

	inputDimensions := c.InputShapes()[0]   // (N x C x H x W)
	outputDimensions := c.OutputShapes()[0] // (N x C x H x W)

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
