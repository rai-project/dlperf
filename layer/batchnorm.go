package layer

import (
	"github.com/rai-project/dlperf"
)

type BatchNorm struct {
	Base             `json:",inline,flatten""`
	InputDimensions  []int64 `json:"input_dimensions,omitempty"`
	OutputDimensions []int64 `json:"output_dimensions,omitempty"`
}

func (BatchNorm) OperatorType() string {
	return "BatchNorm"
}

func (BatchNorm) Aliases() []string {
	return []string{"batchnorm", "bn"}
}

func (BatchNorm) Description() string {
	return ``
}

func (c *BatchNorm) LayerInformation() dlperf.LayerInfo {
	nIn := c.InputDimensions[0]
	cIn := c.InputDimensions[1]
	hIn := c.InputDimensions[2]
	wIn := c.InputDimensions[3]

	numOps := wIn * hIn * cIn * nIn
	flops := dlperf.FlopsInformation{
		Additions: numOps,
		Divisions: numOps,
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
	dlperf.Register(&BatchNorm{})
}
