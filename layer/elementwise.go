package layer

import (
	"strings"

	"github.com/rai-project/dlperf"
)

type ElementWise struct {
	Base               `json:",inline,flatten",omitempty"`
	Operation          string             `json:"operation,omitempty"`
	ParentsInformation []dlperf.LayerInfo `json:"parents_information,omitempty"`
	InputDimensions    []int64            `json:"input_dimensions,omitempty"`
	OutputDimensions   []int64            `json:"output_dimensions,omitempty"`
}

func (ElementWise) OperatorType() string {
	return "ElementWise"
}

func (ElementWise) Aliases() []string {
	return []string{"eltwise"}
}

func (ElementWise) Description() string {
	return ``
}

func (c *ElementWise) LayerInformation() dlperf.LayerInfo {
	nIn := c.InputDimensions[0]
	cIn := c.InputDimensions[1]
	hIn := c.InputDimensions[2]
	wIn := c.InputDimensions[3]

	numOps := wIn * hIn * cIn * nIn

	flops := dlperf.FlopsInformation{}
	switch strings.ToLower(c.Operation) {
	case "sum":
		flops.Additions = numOps
	case "prod":
		flops.MultiplyAdds = numOps
	case "max":
		flops.Comparisons = numOps
	default:
		log.WithField("layer", c.OperatorType()).WithField("operation", c.Operation).Error("invalid layer operation")
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
	dlperf.Register(&ElementWise{})
}
