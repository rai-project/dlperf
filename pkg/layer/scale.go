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

func (c *Scale) InferShape(inputLayers []dlperf.Layer) {
	//c.inputdimensions =  dlperf.ShapeInformation{}
}

func (c Scale) Information() dlperf.LayerInformation {
	info := &Information{
		Base: c.Base,
		shape: dlperf.ShapeInformation{
			InputShapes:  c.inputShapes,
			OutputShapes: c.outputShapes,
		},
	}

	if isAnyEmpty(c.outputShapes) {
		log.WithField("layer", c.OperatorType()).Info("len(OutputShapes) is 0")
		return info
	}

	checkNumber(c.InputShapes, []int{1}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputShapes, []int{1}, c.OperatorType(), "number of outputs")

	inputShapes := c.InputShapes()[0] // (N x C x H x W)

	nIn := inputShapes[0]
	cIn := inputShapes[1]
	hIn := inputShapes[2]
	wIn := inputShapes[3]

	info.flops = dlperf.FlopsInformation{
		MultiplyAdds: wIn * hIn * cIn * nIn,
	}

	return info
}

func init() {
	dlperf.Register(&Scale{})
}
