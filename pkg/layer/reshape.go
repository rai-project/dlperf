package layer

import (
	"github.com/rai-project/dlperf/pkg"
)

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reshape

type Reshape struct {
	*Base `json:",inline,flatten,omitempty"`
}

func (Reshape) Description() string {
	return ``
}

func (c *Reshape) InferShape(inputLayers []dlperf.Layer) {
	inputShapes := getOutputShapes(inputLayers)
	// if c.OperatorType() == "reshape" {
	// 	pp.Println(inputLayers[1].Name())
	// 	pp.Println(inputLayers[1].OutputShapes())
	// }
	c.SetInputShapes(inputShapes)
	c.SetOutputShapes([]dlperf.Shape{inputShapes[1]})
}

func (c Reshape) Information() dlperf.LayerInformation {
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
	dlperf.Register(&Reshape{})
}
