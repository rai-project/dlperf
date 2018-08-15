package layer

import (
	"github.com/rai-project/dlperf/pkg"
)

type ConstantInput struct {
	Base `json:",inline,flatten,omitempty"`
}

func (ConstantInput) OperatorType() string {
	return "ConstantInput"
}

func (ConstantInput) Description() string {
	return ``
}

func (c *ConstantInput) InferShape(inputLayers ...dlperf.Layer) {
	//c.inputdimensions =  dlperf.ShapeInformation{}
}

func (c ConstantInput) Information() dlperf.LayerInformation {
	info := &Information{
		Base: c.Base,
	}

	if isAnyEmpty(c.outputsDimensions) {
		log.WithField("layer", c.OperatorType()).Info("len(OutputsDimensions) is 0")
		return info
	}

	info.flops = dlperf.FlopsInformation{}

	info.shape = dlperf.ShapeInformation{}

	return info
}

func init() {
	dlperf.Register(&ConstantInput{})
}
