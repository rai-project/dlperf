package layer

import (
	"github.com/rai-project/dlperf"
)

type LRN struct {
	Base             `json:",inline,flatten",omitempty"`
	Region           string  `json:"region,omitempty"`
	Size             uint32  `json:"size,omitempty"`
	InputDimensions  []int64 `json:"input_dimensions,omitempty"`
	OutputDimensions []int64 `json:"output_dimensions,omitempty"`
}

func (LRN) Type() string {
	return "LRN"
}

func (LRN) Aliases() []string {
	return []string{"lrn"}
}

func (LRN) Description() string {
	return ``
}

func (c *LRN) LayerInformation() dlperf.LayerInfo {
	nIn := c.InputDimensions[0]
	cIn := c.InputDimensions[1]
	hIn := c.InputDimensions[2]
	wIn := c.InputDimensions[3]

	//  Each input value is divided by (1+(α/n)∑xi^2)^β
	numInputs := wIn * hIn * cIn * nIn
	size := int64(c.Size)
	if c.Region == "WITHIN_CHANNEL" {
		size = size * size
	}

	flops := dlperf.FlopsInformation{
		MultiplyAdds:    numInputs * size, // (∑xi^2)
		Additions:       numInputs,        //  (1+...)
		Exponentiations: numInputs,        // (...)^β
		Divisions:       numInputs * 2,    // (α/n)*... + divide by sum
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
	dlperf.Register(&LRN{})
}
