package layer

import (
	"github.com/rai-project/dlperf"
)

type ReLU struct {
	Base             `json:",inline,flatten""`
	InputDimensions  []int64 `json:"input_dimensions,omitempty"`
	OutputDimensions []int64 `json:"output_dimensions,omitempty"`
}

func (ReLU) Type() string {
	return "ReLU"
}

func (ReLU) Aliases() []string {
	return []string{"relu"}
}

func (ReLU) Description() string {
	return ``
}

func (c *ReLU) LayerInformation() dlperf.LayerInfo {
	nIn := c.InputDimensions[0]
	cIn := c.InputDimensions[1]
	hIn := c.InputDimensions[2]
	wIn := c.InputDimensions[3]

	flops := dlperf.FlopsInformation{
		Comparisons: wIn * hIn * cIn * nIn,
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
	dlperf.Register(&ReLU{})
}
