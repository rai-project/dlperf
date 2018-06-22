package layer

import (
	"github.com/rai-project/dlperf"
)

type Data struct {
	Base `json:",inline,flatten",omitempty"`
	N    int64 `json:"n,omitempty"`
	C    int64 `json:"c,omitempty"`
	W    int64 `json:"w,omitempty"`
	H    int64 `json:"h,omitempty"`
}

func (Data) Type() string {
	return "Data"
}

func (Data) Aliases() []string {
	return []string{"Data"}
}

func (Data) Description() string {
	return ``
}

func (c *Data) LayerInformation(inputDimensions []int64) dlperf.LayerInfo {
	batchSize := c.N
	batchSize = 1
	return &Information{
		name:             c.name,
		typ:              c.Type(),
		inputDimensions:  []int64{batchSize, c.C, c.W, c.H},
		outputDimensions: []int64{batchSize, c.C, c.W, c.H},
	}
}

func init() {
	dlperf.Register(&Data{})
}
