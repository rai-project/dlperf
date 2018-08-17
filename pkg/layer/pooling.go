package layer

import (
	"math"

	"github.com/rai-project/dlperf/pkg"
)

type Pooling struct {
	*Base       `json:",inline,flatten,omitempty"`
	KernelShape dlperf.Shape `json:"kernel_shape,omitempty"`
	Pads        dlperf.Shape `json:"pads,omitempty"`
	Strides     dlperf.Shape `json:"strides,omitempty"`
}

func (Pooling) Description() string {
	return ``
}

func (c *Pooling) InferShape(inputLayers []dlperf.Layer) {
	inputShapes := getOutputShapes(inputLayers)
	xShape := c.inputShapes[0]

	yShape := dlperf.Shape{xShape[0], xShape[1]}
	for ii, xs := range xShape[2:] {
		ys := int64(math.Floor(float64(xs+c.Pads[ii]+c.Pads[ii+1]-c.KernelShape[ii])/float64(c.Strides[ii]))) + 1
		yShape = append(yShape, ys)

	}

	c.SetInputShapes(inputShapes)
	c.SetOutputShapes([]dlperf.Shape{yShape})
}

func (c Pooling) Information() dlperf.LayerInformation {
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

	checkNumber(c.InputShapes, []int{1}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputShapes, []int{1}, c.OperatorType(), "number of outputs")

	outputShape := c.OutputShapes()[0] // (N x C x ...)

	nOut := outputShape[0]
	cOut := outputShape[1]
	hOut := outputShape[2]
	wOut := outputShape[3]

	flops := dlperf.FlopsInformation{}
	switch c.operatorType {
	case "maxpool":
		flops.Comparisons = hOut * wOut * nOut * cOut * c.KernelShape[0] * c.KernelShape[1]
	case "averagepool":
		flops.Additions = hOut * wOut * nOut * cOut * c.KernelShape[0] * c.KernelShape[1]
	}

	info.flops = flops

	return info
}

func init() {
	dlperf.Register(&Pooling{})
}
