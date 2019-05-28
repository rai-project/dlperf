package layer

import dlperf "github.com/rai-project/dlperf/pkg"

//easyjson:json
type TopK struct {
	*Base `json:",inline,flatten,omitempty"`
	K     int64 `json:"k,omitempty"`
	Axis  int64 `json:"axis,omitempty"`
}

func (TopK) OperatorType() string {
	return "TopK"
}

func (TopK) Description() string {
	return ``
}

func (c *TopK) InferShape(inputLayers dlperf.Layers) {
	inputShapes := getOutputShapes(inputLayers)
	inputShape := inputShapes[0]
	rank := len(inputShape)

	axis := c.Axis
	if axis < 0 {
		axis += int64(rank)
	}

	inputShape[axis] = c.K

	c.SetOutputShapes([]dlperf.Shape{
		dlperf.Shape(inputShape),
	})
}

func (c TopK) Information() dlperf.LayerInformation {
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

	checkNumber(c.InputShapes, []int{1, 2}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputShapes, []int{1}, c.OperatorType(), "number of outputs")

	inputShapes := c.InputShapes()[0] // (N x C x H x W)

	numOps := int64(1)
	for _, s := range inputShapes {
		numOps *= s
	}

	info.flops = dlperf.FlopsInformation{
		Comparisons: numOps,
	}

	return info
}

func init() {
	dlperf.Register(&TopK{})
}
