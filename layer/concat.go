package layer

import (
	"github.com/rai-project/dlperf"
)

type Concat struct {
	Base               `json:",inline,flatten",omitempty"`
	InputDimensions    []int64            `json:"input_dimensions,omitempty"`
	OutputDimensions   []int64            `json:"output_dimensions,omitempty"`
	ParentsInformation []dlperf.LayerInfo `json:"parents_information,omitempty"`
}

func (Concat) OperatorType() string {
	return "Concat"
}

func (Concat) Aliases() []string {
	return []string{"concat"}
}

func (Concat) Description() string {
	return ``
}

func (c *Concat) LayerInformation() dlperf.LayerInfo {
	nIn := c.InputDimensions[0]
	cIn := c.InputDimensions[1]
	hIn := c.InputDimensions[2]
	wIn := c.InputDimensions[3]

	wOut := wIn
	hOut := hIn
	cIn = 0
	for _, parent := range c.ParentsInformation {
		// outputDimensions := parent.OutputDimensions()
		_ = parent
		outputDimensions := c.OutputDimensions
		cIn += outputDimensions[1]
	}
	cOut := cIn

	flops := dlperf.FlopsInformation{}

	return &Information{
		name:             c.name,
		operatorType:     c.OperatorType(),
		flops:            flops,
		inputDimensions:  c.InputDimensions,
		outputDimensions: []int64{nIn, cOut, hOut, wOut},
	}
}

func init() {
	dlperf.Register(&Concat{})
}
