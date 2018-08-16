package layer

import (
	"github.com/rai-project/dlperf/pkg"
)

type Pooling struct {
	Base        `json:",inline,flatten,omitempty"`
	KernelShape []int64 `json:"kernel_shape,omitempty"`
}

func (Pooling) Description() string {
	return ``
}

func (c *Pooling) InferShape(inputLayers ...dlperf.Layer) {
	//c.inputdimensions =  dlperf.ShapeInformation{}
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

	inputShapes := c.InputShapes()[0]   // (N x C x H x W)
	outputShapes := c.OutputShapes()[0] // (N x C x H x W)

	nIn := inputShapes[0]
	cIn := inputShapes[1]
	hIn := inputShapes[2]
	wIn := inputShapes[3]

	cOut := outputShapes[1]
	hOut := outputShapes[2]
	wOut := outputShapes[3]

	var kernelH, kernelW int64
	if c.KernelShape != nil {
		kernelH = c.KernelShape[0]
		kernelW = c.KernelShape[1]
	}

	flops := dlperf.FlopsInformation{}
	switch c.operatorType {
	case "maxpool":
		flops.Comparisons = hOut * wOut * cIn * cOut * kernelH * kernelW
	case "globalmaxpool":
		flops.Comparisons = wIn * hIn * cIn * nIn
	case "averagepool":
		flops.Additions = hOut * wOut * cIn * cOut * kernelH * kernelW
	case "globalaveragepool":
		flops.Additions = wIn * hIn * cIn * nIn
	}

	info.flops = flops

	return info
}

func init() {
	dlperf.Register(&Pooling{})
}
