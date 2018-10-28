package layer

import (
	"github.com/rai-project/dlperf/pkg"
)

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gemm

//easyjson:json
type Gemm struct {
	*Base  `json:",inline,flatten,omitempty"`
	Alpha  float64 `json:"alpha,omitempty"`
	Beta   float64 `json:"beta,omitempty"`
	TransA int64   `json:"transa,omitempty"`
	TransB int64   `json:"transb,omitempty"`
}

func (Gemm) OperatorType() string {
	return "Gemm"
}

func (Gemm) Description() string {
	return ``
}

func (c *Gemm) InferShape(inputLayers dlperf.Layers) {
	c.SetInputShapes(getOutputShapes(inputLayers))

	aShape := c.InputShapes()[0]
	var am int64
	if c.TransA == 0 {
		am = aShape[0]
	} else {
		am = aShape[1]
	}

	bShape := c.InputShapes()[1]
	var bn int64
	if c.TransB == 0 {
		bn = bShape[1]
	} else {
		bn = bShape[0]
	}

	yShape := dlperf.Shape{am, bn}
	c.SetOutputShapes([]dlperf.Shape{yShape})
}

func (c Gemm) Information() dlperf.LayerInformation {
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

	checkNumber(c.InputShapes, []int{3}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputShapes, []int{1}, c.OperatorType(), "number of outputs")

	aShape := c.InputShapes()[0]
	var am, ak int64
	if c.TransA == 0 {
		am = aShape[0]
		ak = aShape[1]
	} else {
		am = aShape[1]
		ak = aShape[0]
	}

	bShape := c.InputShapes()[1]
	var bn int64
	if c.TransB == 0 {
		bn = bShape[1]
	} else {
		bn = bShape[0]
	}

	numOuts := am * bn

	info.flops = dlperf.FlopsInformation{
		MultiplyAdds: numOuts * ak,
		Additions:    numOuts,
	}

	return info
}

func init() {
	dlperf.Register(&Gemm{})
}
