package layer

import dlperf "github.com/rai-project/dlperf/pkg"

//easyjson:json
type ConstantOfShape struct {
	*Base `json:",inline,flatten,omitempty"`
}

func (ConstantOfShape) OperatorType() string {
	return "ConstantOfShape"
}

func (ConstantOfShape) Description() string {
	return ``
}

func (c *ConstantOfShape) InferShape(inputLayers dlperf.Layers) {
	inputShapes := getOutputShapes(inputLayers)
	c.SetInputShapes(inputShapes)
	c.SetOutputShapes(inputShapes)
}

func (c ConstantOfShape) Information() dlperf.LayerInformation {
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
	dlperf.Register(&ConstantOfShape{})
}
