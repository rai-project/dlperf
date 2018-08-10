package layer

import (
	"github.com/rai-project/dlperf"
)

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#LRN
// https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

type LRN struct {
	Base              `json:",inline,flatten,omitempty"`
	Size              int64     `json:"size,omitempty"`
	InputsDimensions  [][]int64 `json:"inputs_dimensions,omitempty"`
	OutputsDimensions [][]int64 `json:"outputs_dimensions,omitempty"`
}

func (LRN) OperatorType() string {
	return "LRN"
}

func (LRN) Description() string {
	return ``
}

func (c *LRN) LayerInformation() dlperf.LayerInfo {
	checkNumber(c.InputsDimensions, []int{1}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputsDimensions, []int{1}, c.OperatorType(), "number of outputs")

	inputDimensions := c.InputsDimensions[0]   // (N x C x H x W)
	outputDimensions := c.OutputsDimensions[0] // (N x C x H x W)

	nIn := inputDimensions[0]
	cIn := inputDimensions[1]
	hIn := inputDimensions[2]
	wIn := inputDimensions[3]

	// Each input value is divided by (1+(α/n)∑xi^2)^β
	numInputs := wIn * hIn * cIn * nIn
	size := int64(c.Size)

	// TODO
	flops := dlperf.FlopsInformation{
		MultiplyAdds:    numInputs * size, // (∑xi^2)
		Additions:       numInputs,        // (1+...)
		Exponentiations: numInputs,        // (...)^β
		Divisions:       numInputs * 2,    // (α/n)*... + divide by sum
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
	dlperf.Register(&LRN{})
}
