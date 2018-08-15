package layer

import (
	"encoding/json"

	"github.com/fatih/structs"
	"github.com/mitchellh/mapstructure"
	"github.com/rai-project/dlperf/pkg"
)

type Information struct {
	Base   `json:",inline,flatten,omitempty"`
	shape  dlperf.ShapeInformation  `json:"shape,omitempty"`
	flops  dlperf.FlopsInformation  `json:"flops,omitempty"`
	memory dlperf.MemoryInformation `json:"memory,omitempty"`
}

func (layer *Information) Name() string {
	return layer.name
}

func (layer *Information) OperatorType() string {
	return layer.operatorType
}

func (layer *Information) Inputs() []string {
	return layer.inputs
}

func (layer *Information) Outputs() []string {
	return layer.outputs
}

func (layer *Information) Shape() dlperf.ShapeInformation {
	return dlperf.ShapeInformation{
		InputDimensions:  layer.inputsDimensions[0],
		OutputDimensions: layer.outputsDimensions[0],
	}
}

func (layer *Information) Flops() dlperf.FlopsInformation {
	return layer.flops
}

func (layer *Information) Memory() dlperf.MemoryInformation {
	return layer.memory
}

func (layer Information) MarshalJSON() ([]byte, error) {
	s := structs.New(layer)
	s.TagName = "json"
	data := s.Map()
	return json.Marshal(data)
}

func (layer *Information) UnmarshalJSON(b []byte) error {
	data := map[string]interface{}{}
	err := json.Unmarshal(b, &data)
	if err != nil {
		return err
	}

	config := &mapstructure.DecoderConfig{
		Metadata: nil,
		TagName:  "json",
		Result:   layer,
	}

	decoder, err := mapstructure.NewDecoder(config)
	if err != nil {
		return err
	}

	err = decoder.Decode(data)
	if err != nil {
		return err
	}

	return nil
}
