package layer

import (
	"github.com/rai-project/dlperf"
)

type Gemm struct {
	Base `json:",inline,flatten,omitempty"`
}

func (Gemm) Description() string {
	return ``
}

func (c Gemm) Information() dlperf.LayerInformation {
	info := &Information{
		Base: c.Base,
	}

	if isAnyEmpty(c.inputsDimensions) {
		log.WithField("layer", c.OperatorType()).Info("len(InputsDimensions) is 0")
		return info
	}

	if isAnyEmpty(c.outputsDimensions) {
		log.WithField("layer", c.OperatorType()).Info("len(OutputsDimensions) is 0")
		return info
	}

	checkNumber(c.InputsDimensions, []int{3}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputsDimensions, []int{1}, c.OperatorType(), "number of outputs")

	inputADimensions := c.InputsDimensions()[0]  // (M, K) or (K, M)
	outputDimensions := c.OutputsDimensions()[0] // (M, N)

	K := inputADimensions[1]
	if K == outputDimensions[0] {
		K = inputADimensions[0]
	}

	numOuts := outputDimensions[0] * outputDimensions[1]

	info.flops = dlperf.FlopsInformation{
		MultiplyAdds: numOuts * K,
		Additions:    numOuts,
	}

	info.shape = dlperf.ShapeInformation{
		InputDimensions:  inputADimensions,
		OutputDimensions: outputDimensions,
	}

	return info
}

func init() {
	dlperf.Register(&Gemm{})
}
