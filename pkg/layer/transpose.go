package layer

import (
	"github.com/rai-project/dlperf/pkg"
)

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Transpose

type Transpose struct {
	*Base       `json:",inline,flatten,omitempty"`
	Permutation []int64 `json:"perm,omitempty"`
}

func (Transpose) OperatorType() string {
	return "Transpose"
}

func (Transpose) Description() string {
	return ``
}

func (c *Transpose) InferShape(inputLayers dlperf.Layers) {
	inputShapes := getOutputShapes(inputLayers)
	// if c.OperatorType() == "Transpose" {
	// 	pp.Println(inputLayers[1].Name())
	// 	pp.Println(inputLayers[1].OutputShapes())
	// }

	outputShapes := make([]dlperf.Shape, len(inputShapes))
	for ii, inputShape := range inputShapes {
		outputShapes[ii] = inputShape
		for jj, perm := range c.Permutation {
			outputShapes[ii][jj] = inputShape[perm]
		}
	}
	c.SetOutputShapes(outputShapes)
}

func (c Transpose) Information() dlperf.LayerInformation {
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

	info.flops = dlperf.FlopsInformation{}

	return info
}

func init() {
	dlperf.Register(&Transpose{})
}
