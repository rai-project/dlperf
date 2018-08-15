package layer

import (
	"encoding/json"

	"github.com/rai-project/onnx"
)

type Base struct {
	node              *onnx.NodeProto `json:"-"`
	name              string          `json:"name,omitempty"`
	operatorType      string          `json:"operator_type,omitempty"`
	inputs            []string        `json:",inputs,omitempty"`
	outputs           []string        `json:",outputs,omitempty"`
	inputsDimensions  [][]int64       `json:"inputs_dimensions,omitempty"`
	outputsDimensions [][]int64       `json:"outputs_dimensions,omitempty"`
}

func (b Base) Name() string {
	return b.name
}

func (b *Base) SetNode(node *onnx.NodeProto) {
	b.node = node
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

func (b *Base) SetOperatorType(op string) {
	b.operatorType = op
}

func (b Base) Inputs() []string {
	return b.inputs
}

func (b *Base) SetInputs(in []string) {
	b.inputs = in
}

func (b Base) Outputs() []string {
	return b.outputs
}

func (b *Base) SetOutputs(out []string) {
	b.outputs = out
}

func (b Base) InputsDimensions() [][]int64 {
	return b.inputsDimensions
}

func (b *Base) SetInputsDimensions(in [][]int64) {
	b.inputsDimensions = in
}

func (b Base) OutputsDimensions() [][]int64 {
	return b.outputsDimensions
}

func (b *Base) SetOutputsDimensions(out [][]int64) {
	b.outputsDimensions = out
}

func (b *Base) UnmarshalJSON(d []byte) error {
	return json.Unmarshal(d, &b.name)
}

func (b Base) MarshalJSON() ([]byte, error) {
	return []byte(b.Name()), nil
}
