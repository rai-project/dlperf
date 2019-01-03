package layer

import (
	"strings"

	dlperf "github.com/rai-project/dlperf/pkg"
)

//easyjson:json
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

// multidirectionalBroadcastShapeInference
func (c *ElementWise) InferShape(inputLayers dlperf.Layers) {
	inputShapes := getOutputShapes(inputLayers)
	outputShapes := multidirectionalBroadcastShapeInference(inputShapes)
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
