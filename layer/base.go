package layer

import "encoding/json"

type Base struct {
	name              string    `json:"name,omitempty"`
	operatorType      string    `json:"operator_type,omitempty"`
	inputs            []string  `json:",inputs,omitempty"`
	outputs           []string  `json:",outputs,omitempty"`
	inputsDimensions  [][]int64 `json:"inputs_dimensions,omitempty"`
	outputsDimensions [][]int64 `json:"outputs_dimensions,omitempty"`
}

func (b Base) Name() string {
	return b.name
}

func (b *Base) SetName(s string) {
	b.name = s
}

func (b Base) OperatorType() string {
	if b.operatorType == "" {
		return "unkown operator"
	}
	return b.operatorType
}

func (b Base) Inputs() []string {
	return b.inputs
}

func (b Base) Outputs() []string {
	return b.outputs
}

func (b Base) InputsDimensions() []string {
	return b.inputsDimensions
}

func (b Base) OutputsDimensions() []string {
	return b.outputsDimensions
}

func (b *Base) UnmarshalJSON(d []byte) error {
	return json.Unmarshal(d, &b.name)
}

func (b Base) MarshalJSON() ([]byte, error) {
	return []byte(b.Name()), nil
}
