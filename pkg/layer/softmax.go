package layer

import (
	"github.com/rai-project/dlperf/pkg"
)

type SoftMax struct {
	Base `json:",inline,flatten,omitempty"`
}

func (SoftMax) Description() string {
	return ``
}

func (c *SoftMax) InferShape(inputLayers ...dlperf.Layer) {
	//c.inputdimensions =  dlperf.ShapeInformation{}
}

func (c SoftMax) Information() dlperf.LayerInformation {
	info := &Information{
		Base: c.Base,
	}

	if isAnyEmpty(c.outputShapes) {
		log.WithField("layer", c.OperatorType()).Info("len(OutputShapes) is 0")
		return info
	}

	if isAnyEmpty(c.inputShapes) {
		log.WithField("layer", c.OperatorType()).Info("len(InputDimensions) is 0")
		return info
	}

	checkNumber(c.InputShapes, []int{1}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputShapes, []int{1}, c.OperatorType(), "number of outputs")

	inputDimensions := c.InputShapes()[0]   // (N x C x H x W)
	outputDimensions := c.OutputShapes()[0] // (N x C x H x W)

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
