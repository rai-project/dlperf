package layer

import (
	"github.com/rai-project/dlperf"
)

type Constant struct {
	Base             `json:",inline,flatten",omitempty"`
	N                int64   `json:"n,omitempty"`
	C                int64   `json:"c,omitempty"`
	W                int64   `json:"w,omitempty"`
	H                int64   `json:"h,omitempty"`
	InputDimensions  []int64 `json:"input_dimensions,omitempty"`
	OutputDimensions []int64 `json:"output_dimensions,omitempty"`
}

func (Constant) OperatorType() string {
	return "Constant"
}

func (Constant) Aliases() []string {
	return []string{"Constant"}
}

func (Constant) Description() string {
	return ``
}

func (c *Constant) LayerInformation() dlperf.LayerInfo {
	batchSize := c.N
	batchSize = 1
	return &Information{
		name:             c.name,
		operatorType:     c.OperatorType(),
		inputDimensions:  []int64{batchSize, c.C, c.W, c.H},
		outputDimensions: []int64{batchSize, c.C, c.W, c.H},
	}
}

func init() {
	dlperf.Register(&Constant{})
}
