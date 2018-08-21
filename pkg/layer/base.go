package layer

import (
	"encoding/json"

	dlperf "github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/dlperf/pkg/benchmark"
	"github.com/rai-project/onnx"
)

type BaseBenchmarkInputArgs struct {
	Input0 int64 `args:"input[0]"`
	Input1 int64 `args:"input[1]"`
	Input2 int64 `args:"input[2]"`
	Input3 int64 `args:"input[3]"`
	Input4 int64 `args:"input[4]"`
	Input5 int64 `args:"input[5]"`
	Input6 int64 `args:"input[6]"`
	Input7 int64 `args:"input[7]"`
}

type baseBenchmarkArgs struct {
	ArgNames          []string          `args:"-"`
	UniqueBenchmarkID uint64            `args:"-"`
	BenchmarkName     string            `args:"-"`
	Algorithms        []string          `args:"-"`
	DataTypes         []dlperf.DataType `args:"-"`
}

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

func mkBaseBenchmarkArgs(c dlperf.Layer) baseBenchmarkArgs {
	return baseBenchmarkArgs{
		BenchmarkName: c.FwdBenchmarkName(),
		ArgNames:      c.FwdBenchmarkGeneratorArgNames(),
		Algorithms:    c.FwdBenchmarkAlgorithms(),
		DataTypes:     c.DataTypes(),
	}
}

func mkBaseBenchmarkInputArgs(c dlperf.Layer) BaseBenchmarkInputArgs {
	input := c.InputShapes()[0]
	return BaseBenchmarkInputArgs{
		Input0: getOrMinus1(input, 0),
		Input1: getOrMinus1(input, 1),
		Input2: getOrMinus1(input, 2),
		Input3: getOrMinus1(input, 3),
		Input4: getOrMinus1(input, 4),
		Input5: getOrMinus1(input, 5),
		Input6: getOrMinus1(input, 6),
		Input7: getOrMinus1(input, 7)}
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
func (b Base) FwdBenchmarkGeneratorArgNames() []string {
	panic("FwdBenchmarkGeneratorArgNames not implemented")
	return nil
}

func (b Base) FwdBenchmarkAlgorithms() []string {
	panic("FwdBenchmarkAlgorithms not implemented")
	return nil
}

func (c Base) DataTypes() []dlperf.DataType {
	return dlperf.AllDataTypes
}
