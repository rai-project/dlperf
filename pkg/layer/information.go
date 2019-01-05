package layer

import (
	"encoding/json"
	"unsafe"

	"github.com/fatih/structs"
	"github.com/mitchellh/mapstructure"

	dlperf "github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/onnx"
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

func byteSliceToFloat32Slice(src []byte) []float32 {
	if len(src) == 0 {
		return nil
	}

	l := len(src) / 4
	ptr := unsafe.Pointer(&src[0])
	// It is important to keep in mind that the Go garbage collector
	// will not interact with this data, and that if src if freed,
	// the behavior of any Go code using the slice is nondeterministic.
	// Reference: https://github.com/golang/go/wiki/cgo#turning-c-arrays-into-go-slices
	return (*[1 << 26]float32)((*[1 << 26]float32)(ptr))[:l:l]
}

func (info *Information) Weigths() []float32 {
	var ret []float32
	for _, t := range info.WeightTensors() {
		if t == nil {
			continue
		}
		if t.DataType == onnx.TensorProto_FLOAT {
			if t.FloatData != nil {
				ret = append(ret, t.FloatData...)
			} else if t.RawData != nil {
				ret = append(ret, byteSliceToFloat32Slice(t.RawData)...)
			}
		}
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
