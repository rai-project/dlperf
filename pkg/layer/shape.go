package layer

import dlperf "github.com/rai-project/dlperf/pkg"

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Shape

//easyjson:json
type Shape struct {
	*Base `json:",inline,flatten,omitempty"`
}

func (Shape) OperatorType() string {
	return "Shape"
}

func (Shape) Description() string {
	return ``
}

func (c *Shape) InferShape(inputLayers dlperf.Layers) {
	inputShapes := getOutputShapes(inputLayers)
	c.SetInputShapes(inputShapes)

	c.SetOutputShapes([]dlperf.Shape{
		[]int64{int64(len(inputShapes))},
	})
}

func (c Shape) Information() dlperf.LayerInformation {
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
	dlperf.Register(&Shape{})
}
