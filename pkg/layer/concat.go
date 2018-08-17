package layer

import (
	"github.com/rai-project/dlperf/pkg"
)

type Concat struct {
	*Base `json:",inline,flatten,omitempty"`
	Axis  int `json:"axis,omitempty"`
}

func (Concat) Description() string {
	return ``
}

func (c *Concat) InferShape(inputLayers []dlperf.Layer) {
	inputShapes := getOutputShapes(inputLayers)
	yShape := c.inputShapes[0]

	for _, input := range inputShapes[1:] {
		yShape[c.Axis] += input[c.Axis]
	}

	c.SetInputShapes(inputShapes)
	c.SetOutputShapes([]dlperf.Shape{yShape})
}

func (c Concat) Information() dlperf.LayerInformation {
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

	checkNumber(c.OutputShapes, []int{1}, c.OperatorType(), "number of outputs")

	info.flops = dlperf.FlopsInformation{}

	return info
}

func init() {
	dlperf.Register(&Concat{})
}
