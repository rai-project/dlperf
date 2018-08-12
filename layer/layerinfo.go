package layer

import (
	"encoding/json"

	"github.com/fatih/structs"
	"github.com/mitchellh/mapstructure"
	"github.com/rai-project/dlperf"
)

type Information struct {
	name         string                   `json:"name,omitempty"`
	operatorType string                   `json:"operatorType,omitempty"`
	inputs       []string                 `json:"inputs,omitempty"`
	outputs      []string                 `json:"outputs,omitempty"`
	shape        dlperf.ShapeInformation  `json:"shape,omitempty"`
	flops        dlperf.FlopsInformation  `json:"flops,omitempty"`
	memory       dlperf.MemoryInformation `json:"memory,omitempty"`
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
	return layer.shape
}

func (layer *Information) Flops() dlperf.FlopsInformation {
	layer.flops.InputDimensions = layer.shape.InputDimensions
	layer.flops.OutputDimensions = layer.shape.utputDimensions
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
