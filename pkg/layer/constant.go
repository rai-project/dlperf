package layer

import (
	"github.com/rai-project/dlperf/pkg"
)

type Constant struct {
	Base `json:",inline,flatten,omitempty"`
}

func (Constant) OperatorType() string {
	return "Constant"
}

func (Constant) Description() string {
	return ``
}

func (c *Constant) InferShape(inputLayers []dlperf.Layer) {
	//c.inputdimensions =  dlperf.ShapeInformation{}
}

func (c Constant) Information() dlperf.LayerInformation {
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
	dlperf.Register(&Constant{})
}
