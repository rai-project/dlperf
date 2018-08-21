package layer

import (
	"github.com/rai-project/dlperf/pkg"
)

type Constant struct {
	*Base `json:",inline,flatten,omitempty"`
}

func (Constant) OperatorType() string {
	return "Constant"
}

func (Constant) Description() string {
	return ``
}

func (c *Constant) InferShape(inputLayers dlperf.Layers) {
	inputShapes := getOutputShapes(inputLayers)
	c.SetInputShapes(inputShapes)
	c.SetOutputShapes(inputShapes)
}

func (c Constant) Information() dlperf.LayerInformation {
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
	dlperf.Register(&Constant{})
}
