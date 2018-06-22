package layer

import (
	"math"

	"github.com/rai-project/dlperf"
)

type Convolution struct {
	Base      `json:",inline,flatten",omitempty"`
	NumOutput uint32 `json:"num_output,omitempty"`
	PadH      uint32 `json:"pad_h,omitempty"`
	PadW      uint32 `json:"pad_w,omitempty"`
	KernelH   uint32 `json:"kernel_h,omitempty"`
	KernelW   uint32 `json:"kernel_w,omitempty"`
	StrideH   uint32 `json:"stride_h,omitempty"`
	StrideW   uint32 `json:"stride_w,omitempty"`
	Dilation  uint32 `json:"dilation,omitempty"`
	Group     uint32 `json:"group,omitempty"`
}

func (Convolution) Type() string {
	return "Convolution"
}

func (Convolution) Aliases() []string {
	return []string{"conv", "SpatialConvolution"}
}

func (Convolution) Description() string {
	return ``
}

func (c *Convolution) LayerInformation(inputDimensions []int64) dlperf.LayerInfo {
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
	dlperf.Register(&Convolution{})
}
