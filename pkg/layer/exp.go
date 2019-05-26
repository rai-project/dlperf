package layer

import (
	"strings"

	dlperf "github.com/rai-project/dlperf/pkg"
)

//easyjson:json
type Exp struct {
	*Base     `json:",inline,flatten,omitempty"`
	Broadcast int64 `json:"broadcast,omitempty"`
	Axis      int64 `json:"axis,omitempty"`
}

func (Exp) OperatorType() string {
	return "Exp"
}

func (Exp) Description() string {
	return ``
}

// multidirectionalBroadcastShapeInference
func (c *Exp) InferShape(inputLayers dlperf.Layers) {
	inputShapes := getOutputShapes(inputLayers)
	// outputShapes := multidirectionalBroadcastShapeInference(inputShapes) TODO: NOT correct for mul
	outputShapes := []dlperf.Shape{inputShapes[0]}
	c.SetOutputShapes(outputShapes)
}

func (c Exp) Information() dlperf.LayerInformation {
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
	case "exp":
		flops.MultiplyAdds = numOps
	default:
		log.WithField("layer", c.OperatorType()).WithField("operator", c.OperatorType()).Error("invalid layer operation")
	}

	info.flops = flops

	return info
}

func init() {
	dlperf.Register(&Exp{})
}
