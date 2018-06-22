package layer

import (
	"github.com/rai-project/dlperf"
)

type ReLU struct {
	Base `json:",inline,flatten""`
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

func (c *ReLU) LayerInformation(inputDimensions []int64) dlperf.LayerInfo {
	nIn := inputDimensions[0]
	cIn := inputDimensions[1]
	wIn := inputDimensions[2]
	hIn := inputDimensions[3]

	flops := dlperf.FlopsInformation{
		Comparisons: wIn * hIn * cIn * nIn,
	}

	return &Information{
		name:             c.name,
		typ:              c.Type(),
		flops:            flops,
		inputDimensions:  inputDimensions,
		outputDimensions: inputDimensions,
	}
}

func init() {
	dlperf.Register(&ReLU{})
}
