package layer

import (
	"github.com/rai-project/dlperf"
)

type Gemm Base

func (Gemm) Description() string {
	return ``
}

func (c *Gemm) Information() dlperf.LayerInformation {
	info := &Information{
		name:              c.Name,
		operatorType:      c.OperatorType(),
		inputs:            c.Inputs(),
		outputs:           c.Outputs(),
		inputsDimensions:  c.InputsDimensions,
		outputsDimensions: c.OutputsDimensions,
	}

	if len(c.OutputsDimensions) == 0 {
		log.WithField("layer", c.OperatorType()).Info("len(OutputsDimensions) is 0")
		return info
	}

	checkNumber(c.InputsDimensions, []int{3}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputsDimensions, []int{1}, c.OperatorType(), "number of outputs")

	inputADimensions := c.InputsDimensions[0]  // (M, K) or (K, M)
	outputDimensions := c.OutputsDimensions[0] // (M, N)

	K := inputADimensions[1]
	if K == outputDimensions[0] {
		K = inputADimensions[0]
	}

	numOuts := outputDimensions[0] * outputDimensions[1]

	info.flops = dlperf.FlopsInformation{
		MultiplyAdds: numOuts * K,
		Additions:    numOuts,
	}

	return info
}

func init() {
	dlperf.Register(&Gemm{})
}
