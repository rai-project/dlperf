package layer

import (
	"github.com/rai-project/dlperf"
)

type SoftMax struct {
	Base `json:",inline,flatten""`
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

func (c *SoftMax) LayerInformation(inputDimensions []int64) dlperf.LayerInfo {
	nIn := inputDimensions[0]
	cIn := inputDimensions[1]
	wIn := inputDimensions[2]
	hIn := inputDimensions[3]

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
		inputDimensions:  inputDimensions,
		outputDimensions: inputDimensions,
	}
}

func init() {
	dlperf.Register(&SoftMax{})
}
