package layer

import (
	"encoding/json"

	"github.com/fatih/structs"
	"github.com/mitchellh/mapstructure"
	"github.com/rai-project/dlperf/pkg"
)

type Information struct {
	*Base  `json:",inline,flatten,omitempty"`
	shape  dlperf.ShapeInformation  `json:"shape,omitempty"`
	flops  dlperf.FlopsInformation  `json:"flops,omitempty"`
	memory dlperf.MemoryInformation `json:"memory,omitempty"`
}

func (info *Information) Name() string {
	return info.Name_
}

func (info *Information) Weigths() []float32 {
	var ret []float32
	for _, initializer := range info.initializers {
		ret = append(ret, initializer.FloatData...)
	}
	return ret
}

func (info *Information) OnnxOperatorType() string {
	return info.OnnxOperatorType_
}

func (info *Information) InputNames() []string {
	return info.InputNames_
}

func (info *Information) OutputNames() []string {
	return info.OutputNames_
}

func (info *Information) Shape() dlperf.ShapeInformation {
	return info.shape
}

func (info *Information) Flops() dlperf.FlopsInformation {
	return info.flops
}

func (info *Information) Memory() dlperf.MemoryInformation {
	return info.memory
}

func (layer Information) MarshalJSON() ([]byte, error) {
	s := structs.New(layer)
	s.TagName = "json"
	data := s.Map()
	return json.Marshal(data)
}

func (info *Information) UnmarshalJSON(b []byte) error {
	data := map[string]interface{}{}
	err := json.Unmarshal(b, &data)
	if err != nil {
		return err
	}

	config := &mapstructure.DecoderConfig{
		Metadata: nil,
		TagName:  "json",
		Result:   info,
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
