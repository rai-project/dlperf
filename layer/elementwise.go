package layer

import (
	"strings"

	"github.com/rai-project/dlperf"
)

type ElementWise struct {
	Base               `json:",inline,flatten",omitempty"`
	Operation          string              `json:"operation,omitempty"`
	ParentsInformation []dlperf.LayerInfo `json:"parents_information,omitempty"`
}

func (ElementWise) Type() string {
	return "ElementWise"
}

func (ElementWise) Aliases() []string {
	return []string{"eltwise"}
}

func (ElementWise) Description() string {
	return ``
}

func (c *ElementWise) LayerInformation(inputDimensions []int64) dlperf.LayerInfo {
	nIn := inputDimensions[0]
	cIn := inputDimensions[1]
	wIn := inputDimensions[2]
	hIn := inputDimensions[3]

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
		log.WithField("layer", c.Type()).WithField("operation", c.Operation).Error("invalid layer operation")
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
	dlperf.Register(&ElementWise{})
}
