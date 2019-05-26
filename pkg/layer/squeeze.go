package layer

import dlperf "github.com/rai-project/dlperf/pkg"

//easyjson:json
type Squeeze struct {
	*Base `json:",inline,flatten,omitempty"`
	Axes  []int64 `json:"axes,omitempty"`
}

func (Squeeze) OperatorType() string {
	return "Squeeze"
}

func (Squeeze) Description() string {
	return ``
}

func (c *Squeeze) InferShape(inputLayers dlperf.Layers) {
	inputShapes := getOutputShapes(inputLayers)
	ii := int64(0)
	jj := int64(0)
	res := []int64{}
	for ; ii < int64(len(inputShapes[0])); ii++ {
		if jj < int64(len(c.Axes)) && c.Axes[jj] == ii {
			jj++
		} else {
			res = append(res, inputShapes[0][ii])
		}
	}
	c.SetOutputShapes([]dlperf.Shape{
		dlperf.Shape(res),
	})
}

func (c Squeeze) Information() dlperf.LayerInformation {
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
	dlperf.Register(&Squeeze{})
}
