package layer

import (
	"github.com/rai-project/dlperf/pkg"
)

type ConstantInput struct {
	*Base `json:",inline,flatten,omitempty"`
}

func (ConstantInput) OperatorType() string {
	return "ConstantInput"
}

func (ConstantInput) Description() string {
	return ``
}

func (c *ConstantInput) InferShape(inputLayers []dlperf.Layer) {}

func (c ConstantInput) Information() dlperf.LayerInformation {
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

	info.flops = dlperf.FlopsInformation{}

	return info
}

func init() {
	dlperf.Register(&ConstantInput{})
}
