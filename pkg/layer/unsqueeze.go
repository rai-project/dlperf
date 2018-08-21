package layer

import (
	"github.com/rai-project/dlperf/pkg"
)

type Unsqueeze struct {
	*Base `json:",inline,flatten,omitempty"`
	Axes  []int64 `json:"axes,omitempty"`
}

func (Unsqueeze) OperatorType() string {
	return "Unsqueeze"
}

func (Unsqueeze) Description() string {
	return ``
}

func (c *Unsqueeze) InferShape(inputLayers dlperf.Layers) {
	inputShapes := getOutputShapes(inputLayers)
	for ii, inputShape := range inputShapes {
		for _, ax := range c.Axes {
			if int64(len(inputShape)) >= ax {
				inputShapes[ii] = append(inputShape[:ax], append([]int64{1}, inputShape[ax:]...)...)
			} else {
				if ax != int64(len(inputShape))+1 {
					panic("expecting next axis to be inputShape + 1")
				}
				inputShapes[ii] = append(inputShape, int64(1))
			}
		}
	}
	c.SetOutputShapes(inputShapes)
}

func (c Unsqueeze) Information() dlperf.LayerInformation {
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
	dlperf.Register(&Unsqueeze{})
}
