package layer

import (
	"github.com/rai-project/dlperf"
)

type Input struct {
	Base             `json:",inline,flatten",omitempty"`
	N                int64   `json:"n,omitempty"`
	C                int64   `json:"c,omitempty"`
	W                int64   `json:"w,omitempty"`
	H                int64   `json:"h,omitempty"`
	InputDimensions  []int64 `json:"input_dimensions,omitempty"`
	OutputDimensions []int64 `json:"output_dimensions,omitempty"`
}

func (Input) OperatorType() string {
	return "Input"
}

func (Input) Aliases() []string {
	return []string{"input"}
}

func (Input) Description() string {
	return ``
}

func (c *Input) LayerInformation() dlperf.LayerInfo {
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
	dlperf.Register(&Input{})
}
