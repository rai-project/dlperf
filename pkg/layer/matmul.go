package layer

import (
	"github.com/rai-project/dlperf/pkg"
)

type MatMul struct {
	*Base `json:",inline,flatten,omitempty"`
}

func (MatMul) OperatorType() string {
	return "MatMul"
}

func (MatMul) Description() string {
	return ``
}

func (c *MatMul) InferShape(inputLayers dlperf.Layers) {
	c.SetInputShapes(getOutputShapes(inputLayers))

	aShape := c.inputShapes[0]
	bShape := c.inputShapes[1]
	var cShape dlperf.Shape

	if len(aShape) > 2 {
		cShape = aShape
		if len(bShape) >= 2 {
			cShape[len(cShape)-1] = bShape[len(bShape)-1]
		} else {
			cShape = aShape[:len(bShape)-1]
		}
	} else if len(aShape) == 2 {
		cShape = bShape
		if len(bShape) >= 2 {
			cShape[len(cShape)-2] = aShape[len(aShape)-2]
		} else {
			cShape = aShape[:1]
		}
	} else {
		cShape = bShape
		if len(bShape) >= 2 {
			cShape = bShape[1:]
		} else {
			cShape = dlperf.Shape{1}
		}
	}

	c.SetOutputShapes([]dlperf.Shape{cShape})
}

func (c MatMul) Information() dlperf.LayerInformation {
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

	inputADimensions := c.inputShapes[0] // (N x C x H x W)
	inputBDimensions := c.inputShapes[1] // (N x C x H x W)

	var numOps int64
	dimLen := len(inputADimensions)
	if dimLen == 2 {
		numOps = inputADimensions[0] * inputADimensions[1] * inputBDimensions[1]
	} else if dimLen == 3 {
		numOps = inputADimensions[0] * inputADimensions[1] * inputADimensions[2] * inputBDimensions[2]

	} else {
		numOps = inputADimensions[0] * inputADimensions[1] * inputADimensions[2] * inputADimensions[3] * inputBDimensions[3]
	}

	info.flops = dlperf.FlopsInformation{
		MultiplyAdds: numOps,
	}

	return info
}

func init() {
	dlperf.Register(&MatMul{})
}
