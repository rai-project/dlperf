package layer

import (
	dlperf "github.com/rai-project/dlperf/pkg"
)

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reshape

//easyjson:json
type Reshape struct {
	*Base `json:",inline,flatten,omitempty"`
}

func (Reshape) OperatorType() string {
	return "Reshape"
}

func (Reshape) Description() string {
	return ``
}

func (c *Reshape) InferShape(inputLayers dlperf.Layers) {
	inputShapes := getOutputShapes(inputLayers)
	c.SetInputShapes(inputShapes)
	if len(inputShapes) == 1 {
		c.SetOutputShapes(inputShapes)
		return
	}
	if len(inputShapes) != 2 {
		panic("expecting only 2 inputs for reshape layer")
	}

	inputTensor := inputShapes[0]
	resizeTensor := inputShapes[1]

	negativeOneDim := -1
	resShape := make([]int64, len(resizeTensor))
	for ii, val := range resizeTensor {
		if val == 0 {
			resShape[ii] = inputTensor[ii]
		}
		if val > 0 {
			resShape[ii] = val
		}
		if val == -1 {
			if negativeOneDim != -1 {
				panic("Target shape may not have multiple -1 dimensions")
			}
			negativeOneDim = ii
		}
	}

	if negativeOneDim != -1 && negativeOneDim == len(inputTensor)-1 {
		panic("-1 dimension only supported at last position for now")
	}
	if negativeOneDim != -1 {
		accum := int64(1)
		for _, val := range inputTensor[negativeOneDim:] {
			accum *= val
		}
		resShape[negativeOneDim] = accum
	}

	c.SetOutputShapes([]dlperf.Shape{resShape})
}

func (c Reshape) Information() dlperf.LayerInformation {
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

	info.flops = dlperf.FlopsInformation{}

	return info
}

func init() {
	dlperf.Register(&Reshape{})
}
