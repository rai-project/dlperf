package layer

import (
	"github.com/rai-project/dlperf/pkg"
)

//easyjson:json
type GlobalPooling struct {
	*Base `json:",inline,flatten,omitempty"`
}

func (GlobalPooling) OperatorType() string {
	return "GlobalPooling"
}

func (GlobalPooling) Description() string {
	return ``
}

func (c *GlobalPooling) InferShape(inputLayers dlperf.Layers) {
	c.SetInputShapes(getOutputShapes(inputLayers))

	xShape := c.InputShapes()[0]
	xn := xShape[0]
	xc := xShape[1]

	yShape := dlperf.Shape{xn, xc, 1, 1}
	c.SetOutputShapes([]dlperf.Shape{yShape})
}

func (c GlobalPooling) Information() dlperf.LayerInformation {
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

	inputShapes := c.InputShapes()[0] // (N x C x ...)

	nIn := inputShapes[0]
	cIn := inputShapes[1]
	hIn := inputShapes[2]
	wIn := inputShapes[3]

	flops := dlperf.FlopsInformation{}
	switch c.OnnxOperatorType() {
	case "globalmaxpool":
		flops.Comparisons = wIn * hIn * cIn * nIn
	case "globalaveragepool":
		flops.Additions = wIn * hIn * cIn * nIn
	}

	info.flops = flops

	return info
}

func init() {
	dlperf.Register(&GlobalPooling{})
}
