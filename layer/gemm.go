package layer

import (
	"github.com/rai-project/dlperf"
)

type Gemm struct {
	Base `json:",inline,flatten,omitempty"`
}

func (Gemm) OperatorType() string {
	return "Gemm"
}

func (Gemm) Description() string {
	return ``
}

func (c *Gemm) LayerInformation() dlperf.LayerInfo {
	checkNumber(c.InputsDimensions, []int{3}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputsDimensions, []int{1}, c.OperatorType(), "number of outputs")

	inputADimensions := c.InputsDimensions[0]  // (M, K) or (K, M)
	outputDimensions := c.OutputsDimensions[0] // (M, N)

	K := inputADimensions[1]
	if K == outputDimensions[0] {
		K = inputADimensions[0]
	}

	numOuts := outputDimensions[0] * outputDimensions[1]

	flops := dlperf.FlopsInformation{
		MultiplyAdds: numOuts * K,
		Additions:    numOuts,
	}

	return &Information{
		name:             c.name,
		operatorType:     c.OperatorType(),
		flops:            flops,
		inputDimensions:  inputADimensions,
		outputDimensions: outputDimensions,
	}
}

func init() {
	dlperf.Register(&Gemm{})
}
