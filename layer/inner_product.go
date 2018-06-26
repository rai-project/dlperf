package layer

import (
	"github.com/rai-project/dlperf"
)

type InnerProduct struct {
	Base             `json:",inline,flatten",omitempty"`
	NumOutput        uint32  `json:"num_output,omitempty"`
	InputDimensions  []int64 `json:"input_dimensions,omitempty"`
	OutputDimensions []int64 `json:"output_dimensions,omitempty"`
}

func (InnerProduct) Type() string {
	return "InnerProduct"
}

func (InnerProduct) Aliases() []string {
	return []string{"inner_product"}
}

func (InnerProduct) Description() string {
	return ``
}

func (c *InnerProduct) LayerInformation() dlperf.LayerInfo {
	nIn := c.InputDimensions[0]
	cIn := c.InputDimensions[1]
	hIn := c.InputDimensions[2]
	wIn := c.InputDimensions[3]

	batchOut := nIn

	wOut := int64(1)
	hOut := int64(1)
	cOut := int64(c.NumOutput)

	flops := dlperf.FlopsInformation{
		MultiplyAdds: (wIn * hIn) * cIn * cOut * batchOut,
	}

	return &Information{
		name:             c.name,
		typ:              c.Type(),
		flops:            flops,
		inputDimensions:  c.InputDimensions,
		outputDimensions: []int64{nIn, cOut, hOut, wOut},
	}
}

func init() {
	dlperf.Register(&InnerProduct{})
}
