package layer

import (
	"github.com/rai-project/dlperf"
)

type Scale struct {
	Base             `json:",inline,flatten""`
	InputDimensions  []int64 `json:"input_dimensions,omitempty"`
	OutputDimensions []int64 `json:"output_dimensions,omitempty"`
}

func (Scale) OperatorType() string {
	return "Scale"
}

func (Scale) Aliases() []string {
	return []string{"scale"}
}

func (Scale) Description() string {
	return ``
}

func (c *Scale) LayerInformation() dlperf.LayerInfo {
	nIn := c.InputDimensions[0]
	cIn := c.InputDimensions[1]
	hIn := c.InputDimensions[2]
	wIn := c.InputDimensions[3]

	flops := dlperf.FlopsInformation{
		MultiplyAdds: wIn * hIn * cIn * nIn,
	}

	return &Information{
		name:             c.name,
		operatorType:     c.OperatorType(),
		flops:            flops,
		inputDimensions:  c.InputDimensions,
		outputDimensions: c.OutputDimensions,
	}
}

func init() {
	dlperf.Register(&Scale{})
}
