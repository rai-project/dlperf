package layer

import (
	"github.com/rai-project/dlperf"
)

type Constant struct {
	Base              `json:",inline,flatten,omitempty"`
	inputs            []string  `json:",inputs,omitempty"`
	outputs           []string  `json:",outputs,omitempty"`
	InputsDimensions  [][]int64 `json:"inputs_dimensions,omitempty"`
	OutputsDimensions [][]int64 `json:"outputs_dimensions,omitempty"`
}

func (Constant) OperatorType() string {
	return "Constant"
}

func (Constant) Description() string {
	return ``
}

func (c *Constant) Information() dlperf.LayerInformation {
	return &Information{
		name:         c.name,
		operatorType: c.OperatorType(),
		flops:        dlperf.FlopsInformation{},
	}
}

func init() {
	dlperf.Register(&Constant{})
}
