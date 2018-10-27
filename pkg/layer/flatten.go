package layer

import (
	"github.com/rai-project/dlperf/pkg"
)

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Flatten

type Flatten struct {
	*Base `json:",inline,flatten,omitempty"`
	Axis int64
}

func (Flatten) OperatorType() string {
	return "Flatten"
}

func (Flatten) Description() string {
	return ``
}

func (c *Flatten) InferShape(inputLayers dlperf.Layers) {
	inputShapes := getOutputShapes(inputLayers)
	accum := int64(1)
	for  ii := int64(0); ii < c.Axis; ii+=1 {
		accum *= inputShapes[0][ii]
	}
	shape := []int64{accum}
	for  ii := c.Axis; ii < int64(len(inputShapes[0])); ii+=1 {
		shape = append(shape, inputShapes[0][ii])
	}

	c.SetInputShapes(inputShapes)
	c.SetOutputShapes([]dlperf.Shape{dlperf.Shape(shape)})
}

func (c Flatten) Information() dlperf.LayerInformation {
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
	dlperf.Register(&Flatten{})
}
