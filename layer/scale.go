package layer

import (
	"github.com/rai-project/dlperf"
)

type Scale struct {
	Base `json:",inline,flatten""`
}

func (Scale) Type() string {
	return "Scale"
}

func (Scale) Aliases() []string {
	return []string{"scale"}
}

func (Scale) Description() string {
	return ``
}

func (c *Scale) LayerInformation(inputDimensions []int64) dlperf.LayerInfo {
	nIn := inputDimensions[0]
	cIn := inputDimensions[1]
	wIn := inputDimensions[2]
	hIn := inputDimensions[3]

	flops := dlperf.FlopsInformation{
		MultiplyAdds: wIn * hIn * cIn * nIn,
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
	dlperf.Register(&Scale{})
}
