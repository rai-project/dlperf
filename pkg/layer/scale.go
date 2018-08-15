package layer

import (
	"github.com/rai-project/dlperf/pkg"
)

type Scale struct {
	Base `json:",inline,flatten,omitempty"`
}

func (Scale) Description() string {
	return ``
}

func (c *Scale) InferShape(inputLayers ...dlperf.Layer) {
	//c.inputdimensions =  dlperf.ShapeInformation{}
}

func (c Scale) Information() dlperf.LayerInformation {
	info := &Information{
		Base: c.Base,
	}

	if isAnyEmpty(c.outputShapes) {
		log.WithField("layer", c.OperatorType()).Info("len(OutputShapes) is 0")
		return info
	}

	checkNumber(c.InputShapes, []int{1}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputShapes, []int{1}, c.OperatorType(), "number of outputs")

	inputDimensions := c.InputShapes()[0]   // (N x C x H x W)
	outputDimensions := c.OutputShapes()[0] // (N x C x H x W)

	nIn := inputDimensions[0]
	cIn := inputDimensions[1]
	hIn := inputDimensions[2]
	wIn := inputDimensions[3]

	info.flops = dlperf.FlopsInformation{
		MultiplyAdds: wIn * hIn * cIn * nIn,
	}

	info.shape = dlperf.ShapeInformation{
		InputDimensions:  inputDimensions,
		OutputDimensions: outputDimensions,
	}

	return info
}

func init() {
	dlperf.Register(&Scale{})
}
