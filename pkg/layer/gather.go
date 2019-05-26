package layer

import dlperf "github.com/rai-project/dlperf/pkg"

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gather

//easyjson:json
type Gather struct {
	*Base `json:",inline,flatten,omitempty"`
	Axis  int64 `json:"axis,omitempty"`
}

func (Gather) OperatorType() string {
	return "Gather"
}

func (Gather) Description() string {
	return ``
}

func (c *Gather) InferShape(inputLayers dlperf.Layers) {
	inputShapes := getOutputShapes(inputLayers)
	c.SetInputShapes(inputShapes)

	dims := inputShapes
	dataShape := dims[0]
	indicesShape := dims[1]

	axis := c.Axis

	r := int64(len(dataShape))
	q := int64(len(indicesShape))

	if axis < 0 {
		axis += q
	}
	outRank := q + r - 1

	res := []int64{}
	for ii := int64(0); ii < outRank; ii++ {
		if ii < axis {
			res = append(res, dataShape[ii])
			continue
		}
		if ii >= axis && ii < axis+q {
			res = append(res, indicesShape[ii-axis])
			continue
		}
		res = append(res, dataShape[ii-q+1])
	}

	c.SetOutputShapes([]dlperf.Shape{dlperf.Shape(res)})

}

func (c Gather) Information() dlperf.LayerInformation {
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
	dlperf.Register(&Gather{})
}
