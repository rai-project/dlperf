package layer

import (
	"encoding/json"

	"github.com/fatih/structs"
	"github.com/mitchellh/mapstructure"
	"github.com/rai-project/dlperf"
)

type Information struct {
	typ              string                    `json:"typ,omitempty"`
	name             string                    `json:"name,omitempty"`
	flops            dlperf.FlopsInformation  `json:"flops,omitempty"`
	memory           dlperf.MemoryInformation `json:"memory,omitempty"`
	inputDimensions  []int64                   `json:"input_dimensions,omitempty"`
	outputDimensions []int64                   `json:"output_dimensions,omitempty"`
}

func (layer *Information) Type() string {
	return layer.typ
}

func (layer *Information) Name() string {
	return layer.name
}

func (layer *Information) Flops() dlperf.FlopsInformation {
	layer.flops.InputDimensions = layer.inputDimensions
	layer.flops.OutputDimensions = layer.outputDimensions
	return layer.flops
}
func (layer *Information) Memory() dlperf.MemoryInformation {
	return layer.memory
}
func (layer *Information) InputDimensions() []int64 {
	return layer.inputDimensions
}
func (layer *Information) OutputDimensions() []int64 {
	return layer.outputDimensions
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
	if len(layer.inputDimensions) == 0 {
		layer.inputDimensions = layer.flops.InputDimensions
	}
	if len(layer.outputDimensions) == 0 {
		layer.outputDimensions = layer.flops.OutputDimensions
	}
	return nil
}
