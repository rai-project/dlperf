package layer

import (
	"github.com/rai-project/dlperf"
)

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv
// NCHW tensor layout for passing inputs and outputs

type Conv struct {
	Base              `json:",inline,flatten",omitempty"`
	InputsDimensions  [][]int64 `json:"inputs_dimensions,omitempty"`
	OutputsDimensions [][]int64 `json:"outputs_dimensions,omitempty"`
	AutoPad           string    `json:"auto_pad,omitempty"`
	Dilations         []int64   `json:"dilation,omitempty"`
	Group             int64     `json:"group,omitempty"`
	KernelShape       []int64   `json:"kernel_shape,omitempty"`
	Pads              []int64   `json:"pad_h,omitempty"`
	Strides           []int64   `json:"stride_h,omitempty"`
}

func (Conv) OperatorType() string {
	return "Conv"
}

func (Conv) Description() string {
	return ``
}

func (c *Conv) LayerInformation() dlperf.LayerInfo {
	inputCnt := len(c.InputsDimensions)
	if inputCnt != 2 && inputCnt != 3 {
		log.WithField("layer", c.OperatorType()).WithField("number of inputs ", inputCnt).Error("Conv must have 2 or 3 inputs")
		return nil
	}

	outputCnt := len(c.OutputsDimensions)
	if outputCnt != 1 {
		log.WithField("layer", c.OperatorType()).WithField("number of outputs ", outputCnt).Error("Conv must have 1 output")
		return nil
	}

	inputDimensions := c.InputsDimensions[0]   // (N x C x H x W)
	weightDimensions := c.InputsDimensions[1]  // (C x M x kH x kW)
	outputDimensions := c.OutputsDimensions[0] // (N x C x H x W)

	nIn := inputDimensions[0]
	cIn := inputDimensions[1]

	cOut := outputDimensions[1]
	hOut := outputDimensions[2]
	wOut := outputDimensions[3]

	if weightDimensions[2] != c.KernelShape[0] || weightDimensions[3] != c.KernelShape[1] {
		log.WithField("layer", c.OperatorType()).WithField("weight dimensions", weightDimensions).Error("weight dimensions do not match kernel_shape")

		return nil
	}

	kernelH := c.KernelShape[0]
	kernelW := c.KernelShape[1]

	flops := dlperf.FlopsInformation{
		MultiplyAdds: int64(kernelH*kernelW*hOut*wOut*cIn*cOut*nIn) / c.Group,
	}

	info := &Information{
		name:             c.name,
		operatorType:     c.OperatorType(),
		flops:            flops,
		inputDimensions:  inputDimensions,
		outputDimensions: outputDimensions,
	}

	return info
}

func init() {
	dlperf.Register(&Conv{})
}
