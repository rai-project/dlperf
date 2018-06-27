package layer

import (
	"math"
	"strings"

	"github.com/rai-project/dlperf"
)

type Pooling struct {
	Base             `json:",inline,flatten",omitempty"`
	Operator         string  `json:"operator,omitempty"`
	PadH             uint32  `json:"pad_h,omitempty"`
	PadW             uint32  `json:"pad_w,omitempty"`
	KernelH          uint32  `json:"kernel_h,omitempty"`
	KernelW          uint32  `json:"kernel_w,omitempty"`
	StrideH          uint32  `json:"stride_h,omitempty"`
	StrideW          uint32  `json:"stride_w,omitempty"`
	Global           bool    `json:"global,omitempty"`
	InputDimensions  []int64 `json:"input_dimensions,omitempty"`
	OutputDimensions []int64 `json:"output_dimensions,omitempty"`
}

func (Pooling) OperatorType() string {
	return "Pooling"
}

func (Pooling) Aliases() []string {
	return []string{"pooling"}
}

func (Pooling) Description() string {
	return ``
}

func (c *Pooling) LayerInformation() dlperf.LayerInfo {
	nIn := c.InputDimensions[0]
	cIn := c.InputDimensions[1]
	hIn := c.InputDimensions[2]
	wIn := c.InputDimensions[3]

	batchOut := nIn

	wOut := int64(math.Ceil(float64(wIn+2*int64(c.PadW)-int64(c.KernelW))/float64(c.StrideW))) + 1
	hOut := int64(math.Ceil(float64(hIn+2*int64(c.PadH)-int64(c.KernelH))/float64(c.StrideH))) + 1
	cOut := cIn

	if c.Global {
		wOut = 1
		hOut = 1
	}

	var numOps int64
	if c.Global {
		numOps = wIn * hIn * cIn * batchOut
	} else {
		numOps = wOut * hOut * int64(c.KernelH) * int64(c.KernelW) * cIn * batchOut
	}

	flops := dlperf.FlopsInformation{}
	switch strings.ToLower(c.Operator) {
	case "max":
		flops.Comparisons = numOps
	case "ave":
		flops.Additions = numOps
	}

	return &Information{
		name:             c.name,
		operatorType:     c.OperatorType(),
		flops:            flops,
		inputDimensions:  c.InputDimensions,
		outputDimensions: []int64{nIn, cOut, hOut, wOut},
	}
}

func init() {
	dlperf.Register(&Pooling{})
}
