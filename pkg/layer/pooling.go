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
	}

	if isAnyEmpty(c.outputShapes) {
		log.WithField("layer", c.OperatorType()).Info("len(OutputShapes) is 0")
		return info
	}

	checkNumber(c.InputShapes, []int{1}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputShapes, []int{1}, c.OperatorType(), "number of outputs")

	inputDimensions := c.InputShapes()[0]   // (N x C x H x W)
	outputDimensions := c.OutputShapes()[0] // (N x C x H x W)

	nIn := inputDimensions[0]
	cIn := inputDimensions[1]
	hIn := inputDimensions[2]
	wIn := inputDimensions[3]

	cOut := outputDimensions[1]
	hOut := outputDimensions[2]
	wOut := outputDimensions[3]

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

	info.shape = dlperf.ShapeInformation{
		InputDimensions:  inputDimensions,
		OutputDimensions: outputDimensions,
	}

	return info
}

func init() {
	dlperf.Register(&Pooling{})
}
