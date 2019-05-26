package layer

import dlperf "github.com/rai-project/dlperf/pkg"

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Cast

//easyjson:json
type Cast struct {
	*Base `json:",inline,flatten,omitempty"`
}

func (Cast) OperatorType() string {
	return "Cast"
}

func (Cast) Description() string {
	return ``
}

func (c *Cast) InferShape(inputLayers dlperf.Layers) {
	inputShapes := getOutputShapes(inputLayers)
	c.SetInputShapes(inputShapes)
	c.SetOutputShapes(inputShapes)
}

func (c Cast) Information() dlperf.LayerInformation {
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

	info.flops = dlperf.FlopsInformation{}

	return info
}

func init() {
	dlperf.Register(&Cast{})
}
