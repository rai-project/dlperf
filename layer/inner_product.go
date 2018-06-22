package layer

import (
	"github.com/rai-project/dlperf"
)

type InnerProduct struct {
	Base      `json:",inline,flatten",omitempty"`
	NumOutput uint32 `json:"num_output,omitempty"`
}

func (InnerProduct) Type() string {
	return "InnerProduct"
}

func (InnerProduct) Aliases() []string {
	return []string{"inner_product"}
}

func (InnerProduct) Description() string {
	return ``
}

func (c *InnerProduct) LayerInformation(inputDimensions []int64) dlperf.LayerInfo {
	nIn := inputDimensions[0]
	cIn := inputDimensions[1]
	wIn := inputDimensions[2]
	hIn := inputDimensions[3]

	batchOut := nIn

	wOut := int64(1)
	hOut := int64(1)
	cOut := int64(c.NumOutput)

	flops := dlperf.FlopsInformation{
		MultiplyAdds: (wIn * hIn) * cIn * cOut * batchOut,
	}

	return &Information{
		name:             c.name,
		typ:              c.Type(),
		flops:            flops,
		inputDimensions:  inputDimensions,
		outputDimensions: []int64{nIn, cOut, hOut, wOut},
	}
}

func init() {
	dlperf.Register(&InnerProduct{})
}
