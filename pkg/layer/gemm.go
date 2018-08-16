package layer

import (
	"github.com/rai-project/dlperf/pkg"
)

type Gemm struct {
	Base `json:",inline,flatten,omitempty"`
}

func (Gemm) Description() string {
	return ``
}

func (c *Gemm) InferShape(inputLayers []dlperf.Layer) {
	//c.inputdimensions =  dlperf.ShapeInformation{}
}

func (c Gemm) Information() dlperf.LayerInformation {
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

	checkNumber(c.InputShapes, []int{3}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputShapes, []int{1}, c.OperatorType(), "number of outputs")

	inputADimensions := c.InputShapes()[0] // (M, K) or (K, M)
	outputShapes := c.OutputShapes()[0]    // (M, N)

	K := inputADimensions[1]
	if K == outputShapes[0] {
		K = inputADimensions[0]
	}

	numOuts := outputShapes[0] * outputShapes[1]

	info.flops = dlperf.FlopsInformation{
		MultiplyAdds: numOuts * K,
		Additions:    numOuts,
	}

	return info
}

func init() {
	dlperf.Register(&Gemm{})
}
