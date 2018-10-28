package layer

import (
	"github.com/rai-project/dlperf/pkg"
)

//easyjson:json
type Concat struct {
	*Base `json:",inline,flatten,omitempty"`
	Axis  int64 `json:"axis,omitempty"`
}

func (Concat) Description() string {
	return ``
}

func (Concat) OperatorType() string {
	return "Concat"
}

func (c *Concat) InferShape(inputLayers dlperf.Layers) {
	inputShapes := getOutputShapes(inputLayers)
	yShape := c.InputShapes()[0]

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
			InputShapes:  c.InputShapes(),
			OutputShapes: c.OutputShapes(),
		},
	}

	if isAnyEmpty(c.OutputShapes()) {
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
