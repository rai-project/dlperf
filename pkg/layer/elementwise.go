package layer

import (
	"strings"

	"github.com/rai-project/dlperf/pkg"
)

type ElementWise struct {
	*Base     `json:",inline,flatten,omitempty"`
	Broadcast int64 `json:"broadcast,omitempty"`
	Axis      int64 `json:"axis,omitempty"`
}

func (ElementWise) OperatorType() string {
	return "ElementWise"
}

func (ElementWise) Description() string {
	return ``
}

func (c *ElementWise) InferShape(inputLayers dlperf.Layers) {
	inputShapes := getOutputShapes(inputLayers)
	resultShapeSize := 0
	for _, input := range inputShapes {
		if len(input) > resultShapeSize {
			resultShapeSize = len(input)
		}
	}
	resultShape := dlperf.Shape{}
	for ii := 0; ii < resultShapeSize; ii++ {
		dimValue := int64(1)
		symbolicDim := int64(0)
		numSymbolicDims := int64(0)
		for _, shape := range inputShapes {
			if ii < resultShapeSize-len(shape) {
				continue
			}
			l := ii - resultShapeSize + len(shape)
			dimIJ := int64(0)
			if l < len(shape) {
				dimIJ = shape[l]
			}
			if dimIJ != 0 {
				if dimIJ != 1 {
					if dimValue != dimIJ && dimValue != 1 {
						panic("Incompatible dimensions")
					} else {
						dimValue = dimIJ
					}
				}
			} else {
				if numSymbolicDims == 0 {
					symbolicDim = dimIJ
				}
				numSymbolicDims++
			}
		}
		if dimValue != 0 || numSymbolicDims == 0 {
			resultShape = append(resultShape, int64(dimValue))
		} else if numSymbolicDims == 1 {
			resultShape = append(resultShape, int64(symbolicDim))
		} else {
			resultShape = append(resultShape, 0)
		}
	}

	outputShapes := []dlperf.Shape{resultShape}

	c.SetOutputShapes(outputShapes)
}

func (c ElementWise) Information() dlperf.LayerInformation {
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

	checkNumber(c.InputShapes, nil, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputShapes, []int{1}, c.OperatorType(), "number of outputs")

	numInputs := int64(len(c.InputShapes()))
	inputShapes := c.InputShapes()[0]

	var shape []int64
	for _, s := range inputShapes {
		shape = append(shape, s)
	}

	numOps := int64(1)
	for _, s := range shape {
		numOps *= s
	}

	flops := dlperf.FlopsInformation{}
	switch strings.ToLower(c.OnnxOperatorType()) {
	case "add", "sum", "sub":
		flops.Additions = numOps * (numInputs - 1)
	case "mul", "div":
		flops.MultiplyAdds = numOps
	case "max,", "min":
		flops.Comparisons = numOps * (numInputs - 1)
	default:
		log.WithField("layer", c.OperatorType()).WithField("operator", c.OperatorType()).Error("invalid layer operation")
	}

	info.flops = flops

	return info
}

func init() {
	dlperf.Register(&ElementWise{})
}
