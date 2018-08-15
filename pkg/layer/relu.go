package layer

import (
	"github.com/rai-project/dlperf/pkg"
)

type ReLU struct {
	Base `json:",inline,flatten,omitempty"`
}

func (ReLU) Description() string {
	return ``
}

func (c *ReLU) InferShape(inputLayers ...dlperf.Layer) {
	//c.inputdimensions =  dlperf.ShapeInformation{}
}

func (c ReLU) Information() dlperf.LayerInformation {
	info := &Information{
		Base: c.Base,
	}

	if isAnyEmpty(c.inputShapes) {
		log.WithField("layer", c.OperatorType()).Info("len(InputShapes) is 0")
		return info
	}

	if isAnyEmpty(c.outputShapes) {
		log.WithField("layer", c.OperatorType()).Info("len(OutputShapes) is 0")
		return info
	}

	checkNumber(c.InputShapes, []int{1, 2}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputShapes, []int{1}, c.OperatorType(), "number of outputs")

	inputDimensions := c.InputShapes()[0]   // (N x C x H x W)
	outputDimensions := c.OutputShapes()[0] // (N x C x H x W)

	numOps := int64(1)
	for _, s := range inputDimensions {
		numOps *= s
	}

	info.flops = dlperf.FlopsInformation{
		Comparisons: numOps,
	}

	info.shape = dlperf.ShapeInformation{
		InputDimensions:  inputDimensions,
		OutputDimensions: outputDimensions,
	}

	return info
}

func init() {
	dlperf.Register(&ReLU{})
}
