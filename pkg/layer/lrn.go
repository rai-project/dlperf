package layer

import (
	"github.com/rai-project/dlperf/pkg"
)

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#LRN
// https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

type LRN struct {
	Base `json:",inline,flatten,omitempty"`
	Size int64 `json:"size,omitempty"`
}

func (LRN) Description() string {
	return ``
}

func (c *LRN) InferShape(inputLayers ...dlperf.Layer) {
	//c.inputdimensions =  dlperf.ShapeInformation{}
}

func (c LRN) Information() dlperf.LayerInformation {
	info := &Information{
		Base: c.Base,
	}

	if isAnyEmpty(c.outputShapes) {
		log.WithField("layer", c.OperatorType()).Info("len(OutputShapes) is 0")
		return info
	}

	checkNumber(c.InputShapes, []int{1}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputShapes, []int{1}, c.OperatorType(), "number of outputs")

	inputDimensions := c.InputShapes()[0]   // (N x C x H x W)
	outputDimensions := c.OutputShapes()[0] // (N x C x H x W)

	nIn := inputDimensions[0]
	cIn := inputDimensions[1]
	hIn := inputDimensions[2]
	wIn := inputDimensions[3]

	// Each input value is divided by (1+(α/n)∑xi^2)^β
	numInputs := wIn * hIn * cIn * nIn
	size := int64(c.Size)

	// TODO
	info.flops = dlperf.FlopsInformation{
		MultiplyAdds:    numInputs * size, // (∑xi^2)
		Additions:       numInputs,        // (1+...)
		Exponentiations: numInputs,        // (...)^β
		Divisions:       numInputs * 2,    // (α/n)*... + divide by sum
	}

	info.shape = dlperf.ShapeInformation{
		InputDimensions:  inputDimensions,
		OutputDimensions: outputDimensions,
	}

	return info
}

func init() {
	dlperf.Register(&LRN{})
}
