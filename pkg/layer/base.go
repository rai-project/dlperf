package layer

import (
	"encoding/json"

	dlperf "github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/dlperf/pkg/benchmark"
	"github.com/rai-project/onnx"
)

type Base struct {
	node             *onnx.NodeProto `json:"-"`
	name             string          `json:"name,omitempty"`
	onnxOperatorType string          `json:"onnx_operator_type,omitempty"`
	inputs           dlperf.Layers   `json:-,omitempty"`
	outputs          dlperf.Layers   `json:-,omitempty"`
	inputNames       []string        `json:"inputNames,omitempty"`
	outputNames      []string        `json:"outputNames,omitempty"`
	inputShapes      []dlperf.Shape  `json:"input_shapes,omitempty"`
	outputShapes     []dlperf.Shape  `json:"output_shapes,omitempty"`
}

func (b *Base) Name() string {
	if b == nil {
		return ""
	}
	return b.name
}

func (b *Base) SetName(s string) {
	b.name = s
}

func (b Base) Node() *onnx.NodeProto {
	return b.node
}

func (b *Base) SetNode(node *onnx.NodeProto) {
	b.node = node
}

func (b Base) OperatorType() string {
	panic("invalid operator type")
}

func (b Base) OnnxOperatorType() string {
	if b.onnxOperatorType == "" {
		return "unkown onnx operator"
	}
	return b.onnxOperatorType
}

func (b *Base) SetOnnxOperatorType(op string) {
	b.onnxOperatorType = op
}

func (b Base) Inputs() dlperf.Layers {
	return b.inputs
}

func (b *Base) SetInputs(in dlperf.Layers) {
	b.inputs = in
}

func (b Base) Outputs() dlperf.Layers {
	return b.outputs
}

func (b *Base) SetOutputs(out dlperf.Layers) {
	b.outputs = out
}

func (b Base) InputNames() []string {
	var inputNames []string
	for _, input := range b.inputs {
		inputNames = append(inputNames, input.Name())
	}

	return inputNames
}

func (b Base) SetInputNames(names []string) {
	b.inputNames = names
}

func (b Base) OutputNames() []string {
	var outputNames []string
	for _, input := range b.inputs {
		outputNames = append(outputNames, input.Name())
	}

	return outputNames
}

func (b Base) SetOutputNames(names []string) {
	b.outputNames = names
}

func (b Base) InputShapes() []dlperf.Shape {
	return b.inputShapes
}

func (b *Base) SetInputShapes(in []dlperf.Shape) {
	b.inputShapes = in
}

func (b Base) OutputShapes() []dlperf.Shape {
	return b.outputShapes
}

func (b *Base) SetOutputShapes(out []dlperf.Shape) {
	b.outputShapes = out
}

func (b *Base) UnmarshalJSON(d []byte) error {
	return json.Unmarshal(d, &b.name)
}

func (b Base) MarshalJSON() ([]byte, error) {
	return []byte(b.Name()), nil
}

func (b Base) FwdBenchmarkFilter(datatype, algorithm string) benchmark.Benchmark {
	// panic("unimplemented FwdBenchmarkFilter")
	return benchmark.Benchmark{}
}

func (b Base) FwdBenchmarkName() string {
	panic("unimplemented FwdBenchmarkName")
}

func (b Base) FwdBenchmarkArgs() interface{} {
	panic("FwdBenchmarkArgs not implemented")
	return nil
}
