package layer

import (
	"math"

	"github.com/rai-project/dlperf"
)

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv
// NCHW tensor layout for passing inputs and outputs

type Conv struct {
	Base             `json:",inline,flatten",omitempty"`
	InputDimensions  []int64 `json:"input_dimensions,omitempty"`
	OutputDimensions []int64 `json:"output_dimensions,omitempty"`
	AutoPad          string  `json:"auto_pad,omitempty"`
	Dilations        []int64 `json:"dilation,omitempty"`
	Group            int64   `json:"group,omitempty"`
	KernelShape      []int64 `json:"kernel_shape,omitempty"`
	Pads             []int64 `json:"pad_h,omitempty"`
	Strides          []int64 `json:"stride_h,omitempty"`
}

func (Conv) Type() string {
	return "Conv"
}

func (Conv) Description() string {
	return ``
}

func (c *Conv) LayerInformation() dlperf.LayerInfo {
	nIn := c.InputDimensions[0]
	cIn := c.InputDimensions[1]
	hIn := c.InputDimensions[2]
	wIn := c.InputDimensions[3]

	kernelW := int64(c.Dilations[1]*(c.KernelShape[1]-1) + 1)
	wOut := int64(math.Floor(float64(wIn+int64(2*c.Pads[1])-kernelW)/float64(c.Strides[1]))) + 1
	kernelH := int64(c.Dilations[0]*(c.KernelShape[0]-1) + 1)
	hOut := int64(math.Floor(float64(hIn+int64(2*c.Pads[0])-kernelH)/float64(c.Strides[0]))) + 1
	cOut := c.OutputDimensions[0] * c.OutputDimensions[1] * c.OutputDimensions[2] * c.OutputDimensions[3]

	flops := dlperf.FlopsInformation{
		MultiplyAdds: (int64(c.KernelShape[1]*c.KernelShape[0]) * (wOut * hOut) * cIn * cOut * nIn) / int64(c.Group),
	}

	info := &Information{
		name:             c.name,
		typ:              c.Type(),
		flops:            flops,
		inputDimensions:  c.InputDimensions,
		outputDimensions: []int64{nIn, cOut, hOut, wOut},
	}

	return info
}

func init() {
	dlperf.Register(&Conv{})
}
