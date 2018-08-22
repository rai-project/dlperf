package layer

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/getlantern/deepcopy"
	dlperf "github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/dlperf/pkg/benchmark"
	"github.com/rai-project/onnx"
)

type BaseBenchmarkInputArgs struct {
	Input0 int64 `args:"input[0]" hash:"input[0]" json:"input_0,omitempty"`
	Input1 int64 `args:"input[1]" hash:"input[1]" json:"input_1,omitempty"`
	Input2 int64 `args:"input[2]" hash:"input[2]" json:"input_2,omitempty"`
	Input3 int64 `args:"input[3]" hash:"input[3]" json:"input_3,omitempty"`
	Input4 int64 `args:"input[4]" hash:"input[4]" json:"input_4,omitempty"`
	Input5 int64 `args:"input[5]" hash:"input[5]" json:"input_5,omitempty"`
	Input6 int64 `args:"input[6]" hash:"input[6]" json:"input_6,omitempty"`
	Input7 int64 `args:"input[7]" hash:"input[7]" json:"input_7,omitempty"`
}

type BaseBenchmarkArgs struct {
	ArgNames          []string          `args:"-" json:"arg_names,omitempty"`
	UniqueBenchmarkID uint64            `args:"-" json:"unique_benchmark_id,omitempty"`
	BenchmarkName     string            `args:"-" hash:"name" json:"benchmark_name,omitempty"`
	Algorithms        []string          `args:"-" json:"algorithms,omitempty"`
	DataTypes         []dlperf.DataType `args:"-" json:"data_types,omitempty"`
}

type Base struct {
	node             *onnx.NodeProto `json:"-"`
	name             string          `json:"name,omitempty"`
	operatorType     string          `json:"operator_type,omitempty"`
	onnxOperatorType string          `json:"onnx_operator_type,omitempty"`
	inputs           dlperf.Layers   `json:-,omitempty"`
	outputs          dlperf.Layers   `json:-,omitempty"`
	inputNames       []string        `json:"inputNames,omitempty"`
	outputNames      []string        `json:"outputNames,omitempty"`
	inputShapes      []dlperf.Shape  `json:"input_shapes,omitempty"`
	outputShapes     []dlperf.Shape  `json:"output_shapes,omitempty"`
}

func mkBaseBenchmarkArgs(c dlperf.Layer) BaseBenchmarkArgs {
	return BaseBenchmarkArgs{
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
	if b.operatorType == "" {
		panic("invalid operator type")
	}
	return b.operatorType
}

func (b *Base) SetOperatorType(s string) {
	b.operatorType = s
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
	cpy := []dlperf.Shape{}
	deepcopy.Copy(&cpy, b.inputShapes)
	return cpy
}

func (b *Base) SetInputShapes(in []dlperf.Shape) {
	cpy := []dlperf.Shape{}
	deepcopy.Copy(&cpy, in)
	b.inputShapes = cpy
}

func (b Base) OutputShapes() []dlperf.Shape {
	cpy := []dlperf.Shape{}
	deepcopy.Copy(&cpy, b.outputShapes)
	return cpy
}

func (b *Base) SetOutputShapes(out []dlperf.Shape) {
	cpy := []dlperf.Shape{}
	deepcopy.Copy(&cpy, out)
	b.outputShapes = cpy
}

func (b *Base) UnmarshalJSON(d []byte) error {
	return json.Unmarshal(d, &b.name)
}

func (b Base) MarshalJSON() ([]byte, error) {
	return []byte(b.Name()), nil
}

func (b Base) FwdBenchmarkFilter(datatype, algorithm string) benchmark.Benchmark {
	// panic("unimplemented FwdBenchmarkFilter")
	if b.operatorType == "" {
		return benchmark.Benchmark{}
	}
	return benchmark.Benchmark{
		Name: fmt.Sprintf("LAYER_CUDNN_%s_FWD", strings.ToUpper(b.OperatorType())),
	}
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
