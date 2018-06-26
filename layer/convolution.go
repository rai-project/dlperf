package layer

import (
	"math"

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

func (Conv) Type() string {
	return "Conv"
}

func (Conv) Description() string {
	return ``
}

func (c *Conv) LayerInformation() dlperf.LayerInfo {
	inputCnt := len(c.InputsDimensions)
	if inputCnt != 2 && inputCnt != 3 {
		log.WithField("layer", c.Type()).WithField("number of inputs ", inputCnt).Error("Conv must have 2 or 3 inputs")
		return nil
	}

	outputCnt := len(c.OutputsDimensions)
	if outputCnt != 1 {
		log.WithField("layer", c.Type()).WithField("number of outputs ", outputCnt).Error("Conv must have 1 output")
		return nil
	}

	inputDimensions := c.InputsDimensions[0]  // (N x C x H x W)
	weightDimensions := c.InputsDimensions[1] // (M x C x kH x kW)
	if weightDimensions[2] != c.KernelShape[0] || weightDimensions[3] != c.KernelShape[1] {
		log.WithField("layer", c.Type()).WithField("weight dimensions", weightDimensions).Error("weight dimensions do not match kernel_shape")
		return nil
	}
	outputDimensions := c.OutputsDimensions[0]

	nIn := inputDimensions[0]
	cIn := inputDimensions[1]
	hIn := inputDimensions[2]
	wIn := inputDimensions[3]

	kernelH := c.Dilations[0]*(c.KernelShape[0]-1) + 1
	kernelW := c.Dilations[1]*(c.KernelShape[1]-1) + 1

	padH := c.Pads[0]
	padW := c.Pads[1]

	// if autoPad == "SAME_UPPER" {

	//   } else if autoPad == "SAME_LOWER" {

	//   } else if autoPad != "VALID" {
	//     log.WithField("layer", "conv").Error("unknown auto_pad, auto_pad must be either SAME_UPPER, SAME_LOWER or VALID")
	//   }
	// }

	hOut := int64(math.Floor(float64(hIn+2*padH-kernelH)/float64(c.Strides[0]))) + 1
	wOut := int64(math.Floor(float64(wIn+2*padW-kernelW)/float64(c.Strides[1]))) + 1

	cOut := outputDimensions[0] * outputDimensions[1] * outputDimensions[2] * outputDimensions[3]

	flops := dlperf.FlopsInformation{
		MultiplyAdds: int64(c.KernelShape[1]*c.KernelShape[0]*wOut*hOut*cIn*cOut*nIn) / c.Group,
	}

	info := &Information{
		name:             c.name,
		typ:              c.Type(),
		flops:            flops,
		inputDimensions:  inputDimensions,
		outputDimensions: outputDimensions,
	}

	return info
}

func init() {
	dlperf.Register(&Conv{})
}
