package layer

import dlperf "github.com/rai-project/dlperf/pkg"

//easyjson:json
type Reduce struct {
	*Base    `json:",inline,flatten,omitempty"`
	Axes     []int64 `json:"axes,omitempty"`
	KeepDims bool    `json:"keepdims,omitempty"`
}

func (Reduce) OperatorType() string {
	return "Reduce"
}

func (Reduce) Description() string {
	return ``
}

func (c *Reduce) InferShape(inputLayers dlperf.Layers) {
	inputShapes := getOutputShapes(inputLayers)
	c.SetInputShapes(inputShapes)

	inputShape := inputShapes[0]

	axes := make([]int64, len(c.Axes))
	for ii, a := range c.Axes {
		axes[ii] = a
		if a < 0 {
			axes[ii] = a + inputShape[ii]
		}
	}

	res := []int64{}

	for ii := range inputShape {
		if len(axes) != 0 && containsInt64(axes, int64(ii)) {
			res = append(res, inputShape[ii])
		} else if c.KeepDims {
			res = append(res, int64(1))
		}
	}

	c.SetOutputShapes([]dlperf.Shape{
		dlperf.Shape(res),
	})
}

func containsInt64(lst []int64, elem int64) bool {
	for _, a := range lst {
		if a == elem {
			return true
		}
	}
	return false
}

func (c Reduce) Information() dlperf.LayerInformation {
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
	dlperf.Register(&Reduce{})
}
