package layer

import (
	"github.com/rai-project/dlperf"
)

type SoftMax struct {
	Base             `json:",inline,flatten""`
	InputDimensions  []int64 `json:"input_dimensions,omitempty"`
	OutputDimensions []int64 `json:"output_dimensions,omitempty"`
}

func (SoftMax) Type() string {
	return "SoftMax"
}

func (SoftMax) Aliases() []string {
	return []string{"relu"}
}

func (SoftMax) Description() string {
	return ``
}

func (c *SoftMax) LayerInformation() dlperf.LayerInfo {
	nIn := c.InputDimensions[0]
	cIn := c.InputDimensions[1]
	hIn := c.InputDimensions[2]
	wIn := c.InputDimensions[3]

	numOps := wIn * hIn * cIn * nIn
	flops := dlperf.FlopsInformation{
		Exponentiations: numOps,
		Additions:       numOps,
		Divisions:       numOps,
	}

	return &Information{
		name:             c.name,
		typ:              c.Type(),
		flops:            flops,
		inputDimensions:  c.InputDimensions,
		outputDimensions: c.OutputDimensions,
	}
}

func init() {
	dlperf.Register(&SoftMax{})
}
