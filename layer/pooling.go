package layer

import (
	"strings"

	"github.com/rai-project/dlperf"
)

type Pooling struct {
	Base              `json:",inline,flatten,omitempty"`
	inputs            []string  `json:",inputs,omitempty"`
	outputs           []string  `json:",outputs,omitempty"`
	Operator          string    `json:"operator,omitempty"`
	KernelShape       []int64   `json:"kernel_shape,omitempty"`
	InputsDimensions  [][]int64 `json:"inputs_dimensions,omitempty"`
	OutputsDimensions [][]int64 `json:"outputs_dimensions,omitempty"`
}

func (c *Pooling) OperatorType() string {
	return c.Operator
}

func (Pooling) Description() string {
	return ``
}

func (c *Pooling) LayerInformation() dlperf.LayerInfo {
	checkNumber(c.InputsDimensions, []int{1}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputsDimensions, []int{1}, c.OperatorType(), "number of outputs")

	inputDimensions := c.InputsDimensions[0]   // (N x C x H x W)
	outputDimensions := c.OutputsDimensions[0] // (N x C x H x W)

	nIn := inputDimensions[0]
	cIn := inputDimensions[1]
	hIn := inputDimensions[2]
	wIn := inputDimensions[3]

	cOut := outputDimensions[1]
	hOut := outputDimensions[2]
	wOut := outputDimensions[3]

	var kernelH, kernelW int64
	if c.KernelShape != nil {
		kernelH = c.KernelShape[0]
		kernelW = c.KernelShape[1]
	}

	flops := dlperf.FlopsInformation{}
	switch strings.ToLower(c.Operator) {
	case "maxpool":
		flops.Comparisons = hOut * wOut * cIn * cOut * kernelH * kernelW
	case "globalmaxpool":
		flops.Comparisons = wIn * hIn * cIn * nIn
	case "averagepool":
		flops.Additions = hOut * wOut * cIn * cOut * kernelH * kernelW
	case "globalaveragepool":
		flops.Additions = wIn * hIn * cIn * nIn
	}

	return &Information{
		name:             c.name,
		operatorType:     c.OperatorType(),
		flops:            flops,
		inputDimensions:  inputDimensions,
		outputDimensions: outputDimensions,
	}
}

func init() {
	dlperf.Register(&Pooling{})
}
