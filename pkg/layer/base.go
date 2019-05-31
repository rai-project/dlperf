package layer

import (
	"fmt"
	"strings"

	dlperf "github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/dlperf/pkg/benchmark"
	"github.com/rai-project/onnx"
)

type BaseBenchmarkInputArgs struct {
	Input0    int64 `args:"input[0]" hash:"input[0]" json:"input_0,omitempty"`
	Input1    int64 `args:"input[1]" hash:"input[1]" json:"input_1,omitempty"`
	Input2    int64 `args:"input[2]" hash:"input[2]" json:"input_2,omitempty"`
	Input3    int64 `args:"input[3]" hash:"input[3]" json:"input_3,omitempty"`
	Input4    int64 `args:"input[4]" hash:"input[4]" json:"input_4,omitempty"`
	Input5    int64 `args:"input[5]" hash:"input[5]" json:"input_5,omitempty"`
	Input6    int64 `args:"input[6]" hash:"input[6]" json:"input_6,omitempty"`
	Input7    int64 `args:"input[7]" hash:"input[7]" json:"input_7,omitempty"`
	BatchSize int64 `args:"batch_size" hash:"batch_size" json:"batch_size,omitempty"`
}

type BaseBenchmarkArgs struct {
	ArgNames          []string          `args:"-" json:"arg_names,omitempty"`
	UniqueBenchmarkID uint64            `args:"-" json:"unique_benchmark_id,omitempty"`
	BenchmarkName     string            `args:"-" hash:"name" json:"benchmark_name,omitempty"`
	Algorithms        []string          `args:"-" json:"algorithms,omitempty"`
	DataTypes         []dlperf.DataType `args:"-" json:"data_types,omitempty"`
	IsTraining        bool              `args:"-" json:"is_training,omitempty"`
	BatchSize        int64              `args:"batch_size" json:"batch_size,omitempty"`
}

type Base struct {
	node              *onnx.NodeProto     `json:"-"`
	weightTensors     []*onnx.TensorProto `json:"-"`
	Name_             string              `json:"name,omitempty"`
	OperatorType_     string              `json:"operator_type,omitempty"`
	OnnxOperatorType_ string              `json:"onnx_operator_type,omitempty"`
	inputs            dlperf.Layers       `json:-,omitempty"`
	outputs           dlperf.Layers       `json:-,omitempty"`
	InputNames_       []string            `json:"input_names,omitempty"`
	OutputNames_      []string            `json:"output_names,omitempty"`
	InputShapes_      []dlperf.Shape      `json:"input_shapes,omitempty"`
	OutputShapes_     []dlperf.Shape      `json:"output_shapes,omitempty"`
}

func mkBaseBenchmarkFWDArgs(c dlperf.Layer, opts ...dlperf.FwdBenchmarkArgsOptionFunc) BaseBenchmarkArgs {
	return BaseBenchmarkArgs{
		BenchmarkName: c.FwdBenchmarkName(opts...),
		ArgNames:      c.FwdBenchmarkGeneratorArgNames(),
		Algorithms:    c.FwdBenchmarkAlgorithms(opts...),
    DataTypes:     c.DataTypes(),
    BatchSize: dlperf.GetBatchSize(),
	}
}

func mkBaseBenchmarkBWDArgs(c dlperf.Layer, opts ...dlperf.BwdBenchmarkArgsOptionFunc) BaseBenchmarkArgs {
	return BaseBenchmarkArgs{
		BenchmarkName: c.BwdBenchmarkName(opts...),
		ArgNames:      c.BwdBenchmarkGeneratorArgNames(),
		Algorithms:    c.BwdBenchmarkAlgorithms(opts...),
		DataTypes:     c.DataTypes(),
    BatchSize: dlperf.GetBatchSize(),
	}
}

func mkBaseBenchmarkInputArgs(c dlperf.Layer) BaseBenchmarkInputArgs {
	input := c.InputShapes()[0]
	return BaseBenchmarkInputArgs{
		Input0:    getOrMinus1(input, 0),
		Input1:    getOrMinus1(input, 1),
		Input2:    getOrMinus1(input, 2),
		Input3:    getOrMinus1(input, 3),
		Input4:    getOrMinus1(input, 4),
		Input5:    getOrMinus1(input, 5),
		Input6:    getOrMinus1(input, 6),
		Input7:    getOrMinus1(input, 7),
		BatchSize: dlperf.GetBatchSize(),
	}
}

func (b *Base) Name() string {
	if b == nil {
		return ""
	}
	return b.Name_
}

func (b *Base) SetName(s string) {
	b.Name_ = s
}

func (b Base) Node() *onnx.NodeProto {
	return b.node
}

func (b *Base) SetNode(node *onnx.NodeProto) {
	b.node = node
}

func (b Base) WeightTensors() []*onnx.TensorProto {
	return b.weightTensors
}

func (b *Base) SetWeightTensors(tensors []*onnx.TensorProto) {
	b.weightTensors = tensors
}

func (b Base) OperatorType() string {
	if b.OperatorType_ == "" {
		panic("invalid operator type")
	}
	return b.OperatorType_
}

func (b *Base) SetOperatorType(s string) {
	b.OperatorType_ = s
}

func (b Base) OnnxOperatorType() string {
	if b.OnnxOperatorType_ == "" {
		return "unkown onnx operator"
	}
	return b.OnnxOperatorType_
}

func (b *Base) SetOnnxOperatorType(op string) {
	b.OnnxOperatorType_ = op
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
	inputNames := make([]string, len(b.inputs))
	for ii, input := range b.inputs {
		inputNames[ii] = input.Name()
	}

	return inputNames
}

func (b *Base) SetInputNames(names []string) {
	inputNames := make([]string, len(names))
	for ii, name := range names {
		inputNames[ii] = name
	}
	b.InputNames_ = inputNames
}

func (b Base) OutputNames() []string {
	outputNames := make([]string, len(b.outputs))
	for ii, output := range b.outputs {
		outputNames[ii] = output.Name()
	}

	return outputNames
}

func (b *Base) SetOutputNames(names []string) {
	outputNames := make([]string, len(names))
	for ii, name := range names {
		outputNames[ii] = name
	}
	b.OutputNames_ = outputNames
}

func (b Base) InputShapes() []dlperf.Shape {
	in := b.InputShapes_
	cpy := make([]dlperf.Shape, len(in))
	for ii, e := range in {
		tmp := make([]int64, len(e))
		for jj, m := range e {
			tmp[jj] = m
		}
		cpy[ii] = dlperf.Shape(tmp)
	}
	return cpy
}

func (b *Base) SetInputShapes(in []dlperf.Shape) {
	cpy := make([]dlperf.Shape, len(in))
	for ii, e := range in {
		tmp := make([]int64, len(e))
		for jj, m := range e {
			tmp[jj] = m
		}
		cpy[ii] = dlperf.Shape(tmp)
	}
	b.InputShapes_ = cpy
}

func (b Base) OutputShapes() []dlperf.Shape {
	out := b.OutputShapes_
	cpy := make([]dlperf.Shape, len(out))
	for ii, e := range out {
		tmp := make([]int64, len(e))
		for jj, m := range e {
			tmp[jj] = m
		}
		cpy[ii] = dlperf.Shape(tmp)
	}
	return cpy
}

func (b *Base) SetOutputShapes(out []dlperf.Shape) {
	cpy := make([]dlperf.Shape, len(out))
	for ii, e := range out {
		tmp := make([]int64, len(e))
		for jj, m := range e {
			tmp[jj] = m
		}
		cpy[ii] = dlperf.Shape(tmp)
	}
	b.OutputShapes_ = cpy
}

// func (b *Base) UnmarshalJSON(d []byte) error {
// 	return json.Unmarshal(d, &b.Name_)
// }

// func (b Base) MarshalJSON() ([]byte, error) {
// 	return []byte(b.Name()), nil
// }

func (b Base) FwdBenchmarkFilter(datatype, algorithm string, opts ...dlperf.FwdBenchmarkArgsOptionFunc) benchmark.Benchmark {
	// panic("unimplemented FwdBenchmarkFilter")
	if b.OperatorType_ == "" {
		return benchmark.Benchmark{}
	}
	return benchmark.Benchmark{
		Name: fmt.Sprintf("LAYER_CUDNN_%s_FWD", strings.ToUpper(b.OperatorType())),
	}
}

func (b Base) FwdBenchmarkName(...dlperf.FwdBenchmarkArgsOptionFunc) string {
	panic("unimplemented FwdBenchmarkName")
}

func (b Base) FwdBenchmarkArgs(...dlperf.FwdBenchmarkArgsOptionFunc) interface{} {
	panic("FwdBenchmarkArgs not implemented")
	return nil
}
func (b Base) FwdBenchmarkGeneratorArgNames() []string {
	panic("FwdBenchmarkGeneratorArgNames not implemented")
	return nil
}

func (b Base) FwdBenchmarkAlgorithms(...dlperf.FwdBenchmarkArgsOptionFunc) []string {
	panic("FwdBenchmarkAlgorithms not implemented")
	return nil
}

func (b Base) BwdBenchmarkFilter(datatype, algorithm string, opts ...dlperf.BwdBenchmarkArgsOptionFunc) benchmark.Benchmark {
	// panic("unimplemented BwdBenchmarkFilter")
	if b.OperatorType_ == "" {
		return benchmark.Benchmark{}
	}
	return benchmark.Benchmark{
		Name: fmt.Sprintf("LAYER_CUDNN_%s_BWD", strings.ToUpper(b.OperatorType())),
	}
}

func (b Base) BwdBenchmarkName(opts ...dlperf.BwdBenchmarkArgsOptionFunc) string {
	panic("unimplemented BwdBenchmarkName")
}

func (b Base) BwdBenchmarkArgs(opts ...dlperf.BwdBenchmarkArgsOptionFunc) interface{} {
	panic("BwdBenchmarkArgs not implemented")
	return nil
}
func (b Base) BwdBenchmarkGeneratorArgNames() []string {
	panic("BwdBenchmarkGeneratorArgNames not implemented")
	return nil
}

func (b Base) BwdBenchmarkAlgorithms(...dlperf.BwdBenchmarkArgsOptionFunc) []string {
	panic("BwdBenchmarkAlgorithms not implemented")
	return nil
}

func (c Base) DataTypes() []dlperf.DataType {
	return dlperf.AllDataTypes
}
