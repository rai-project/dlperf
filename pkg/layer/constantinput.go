package layer

import (
	"github.com/rai-project/dlperf/pkg"
)

//easyjson:json
type ConstantInput struct {
	*Base `json:",inline,flatten,omitempty"`
}

func (ConstantInput) OperatorType() string {
	return "ConstantInput"
}

func (ConstantInput) Description() string {
	return ``
}

func (c *ConstantInput) InferShape(inputLayers dlperf.Layers) {
	// intentionally blank, set in mkConstantInput
}

func (c ConstantInput) FwdBenchmarkGenerator() string {
	return ""
}

func (c ConstantInput) FwdBenchmarkGeneratorPrefix(standAloneGenerate bool) string {
	return ""
}

func (c ConstantInput) FwdBenchmarkGeneratorSuffix(standAloneGenerate bool) string {
	return ""
}

func (c ConstantInput) Information() dlperf.LayerInformation {
	info := &Information{
		Base: c.Base,
		shape: dlperf.ShapeInformation{
			InputShapes:  c.InputShapes(),
			OutputShapes: c.OutputShapes(),
		},
	}

	if isAnyEmpty(c.OutputShapes()) {
		log.WithField("layer", c.OperatorType()).Info("len(OutputShapes) is 0")
		return info
	}

	info.flops = dlperf.FlopsInformation{}

	return info
}

func init() {
	dlperf.Register(&ConstantInput{})
}
