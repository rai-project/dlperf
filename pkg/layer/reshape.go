package layer

import dlperf "github.com/rai-project/dlperf/pkg"

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
	c.SetOutputShapes([]dlperf.Shape{inputShapes[1]})
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
