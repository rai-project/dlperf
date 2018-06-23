package layer

import (
	"math"

	"github.com/rai-project/dlperf"
	"github.com/rai-project/onnx"
)

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv
// NCHW tensor layout for passing inputs and outputs

type Conv struct {
	Base        `json:",inline,flatten",omitempty"`
	AutoPad     string   `json:"auto_pad,omitempty"`
	Dilations   []uint32 `json:"dilation,omitempty"`
	Group       uint32   `json:"group,omitempty"`
	KernelShape []uint32 `json:"kernel_shape,omitempty"`
	Pads        []uint32 `json:"pad_h,omitempty"`
	Strides     []uint32 `json:"stride_h,omitempty"`
}

func NewConv(node *onnx.NodeProto) (*Conv, error) {

  autoPad, err := getNodeAttributeFromName(node)

	return &Conv{
		AutoPad:      node.,
		nodes:            nodes,
		initializers:     initializers,
		layerInformation: make(map[string]dlperf.LayerInfo),
	}, nil
}

func (Conv) Type() string {
	return "Conv"
}

func (Conv) Description() string {
	return ``
}

func (c *Conv) LayerInformation() dlperf.LayerInfo {
	/*
	  ## Add Input/Output Dimensions + Channels to each Node / Layer
	  # shape.dim: (    N   x   K   x   W   x   H   )
	  #              batch   channel  width   height
	  #               nIn      cIn     wIn     wOut
	*/

	nIn := inputDimensions[0]
	cIn := inputDimensions[1]
	wIn := inputDimensions[2]
	hIn := inputDimensions[3]

	kernelW := int64(c.Dilation*(c.KernelW-1) + 1)
	wOut := int64(math.Floor(float64(wIn+int64(2*c.PadW)-kernelW)/float64(c.StrideW))) + 1
	kernelH := int64(c.Dilation*(c.KernelH-1) + 1)
	hOut := int64(math.Floor(float64(hIn+int64(2*c.PadH)-kernelH)/float64(c.StrideH))) + 1
	cOut := int64(c.NumOutput)

	flops := dlperf.FlopsInformation{
		MultiplyAdds: (int64(c.KernelW*c.KernelH) * (wOut * hOut) * cIn * cOut * nIn) / int64(c.Group),
	}

	info := &Information{
		name:             c.name,
		typ:              c.Type(),
		flops:            flops,
		inputDimensions:  inputDimensions,
		outputDimensions: []int64{nIn, cOut, hOut, wOut},
	}

	return info
}

func init() {
	dlperf.Register(&Conv{})
}
