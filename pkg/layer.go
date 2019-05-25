package dlperf

import (
	"encoding/json"
	"fmt"

	"github.com/rai-project/dlperf/pkg/benchmark"
	"github.com/rai-project/onnx"
	"github.com/rai-project/utils"
)

type Shape []int64

type Layers []Layer

type Layer interface {
	Name() string
	Node() *onnx.NodeProto
	WeightTensors() []*onnx.TensorProto
	OperatorType() string
	InferShape(Layers)
	Inputs() Layers
	SetInputs(Layers)
	Outputs() Layers
	SetOutputs(Layers)
	InputShapes() []Shape
	SetInputShapes([]Shape)
	OutputShapes() []Shape
	Information() LayerInformation
	FwdBenchmarkName() string
	FwdBenchmarkFilter(string, string) benchmark.Benchmark
	FwdBenchmarkArgs() interface{}
	FwdBenchmarkGeneratorArgNames() []string
	FwdBenchmarkAlgorithms() []string

	BwdBenchmarkName() string
	BwdBenchmarkFilter(string, string) benchmark.Benchmark
	BwdBenchmarkArgs() interface{}
	BwdBenchmarkGeneratorArgNames() []string
	BwdBenchmarkAlgorithms() []string
	DataTypes() []DataType
}

type LayerInformation interface {
	Name() string
	OperatorType() string
	InputNames() []string
	OutputNames() []string
	Weigths() []float32
	Shape() ShapeInformation
	Flops() FlopsInformation
	Memory() MemoryInformation
}

type ShapeInformation struct {
	InputShapes  []Shape `json:"input_shapes,omitempty"`
	OutputShapes []Shape `json:"output_shapes,omitempty"`
}

func (ShapeInformation) Header() []string {
	return []string{"InputShapes", "OutputShapes"}
}

func (this ShapeInformation) Row() []string {
	// inputShapes := strings.Trim(strings.Replace(fmt.Sprint(l.Shape.InputShapes), " ", ";", -1), "[]")
	// outputShapes := strings.Trim(strings.Replace(fmt.Sprint(l.Shape.OutputShapes), " ", ";", -1), "[]")

	dimsToString := func(e []Shape) string {
		if len(e) == 0 {
			return ""
		}
		bts, err := json.Marshal(e)
		if err != nil {
			return fmt.Sprintf("%v", e)
		}
		return string(bts)
	}
	return []string{
		dimsToString(this.InputShapes),
		dimsToString(this.OutputShapes),
	}
}

type FlopsInformation struct {
	MultiplyAdds    int64 `json:"multiply_adds"`
	Additions       int64 `json:"additions"`
	Divisions       int64 `json:"divisions"`
	Exponentiations int64 `json:"exponentiations"`
	Comparisons     int64 `json:"comparisons"`
	General         int64 `json:"general"`
}

func (FlopsInformation) Header() []string {
	return []string{
		"MultiplyAdds",
		"Additions",
		"Divisions",
		"Exponentiations",
		"Comparisons",
		"General",
	}
}

func (this FlopsInformation) Row(humanFlops bool) []string {
	flopsToString := func(e int64) string {
		return fmt.Sprintf("%v", e)
	}
	if humanFlops {
		flopsToString = func(e int64) string {
			return utils.Flops(uint64(e))
		}
	}
	return []string{
		flopsToString(this.MultiplyAdds),
		flopsToString(this.Additions),
		flopsToString(this.Divisions),
		flopsToString(this.Exponentiations),
		flopsToString(this.Comparisons),
		flopsToString(this.General),
	}
}

func (this FlopsInformation) Total() int64 {
	return this.MultiplyAdds + this.Additions + this.Divisions +
		this.Exponentiations + this.Comparisons + this.General
}

func (this FlopsInformation) Add(other FlopsInformation) FlopsInformation {
	return FlopsInformation{
		MultiplyAdds:    this.MultiplyAdds + other.MultiplyAdds,
		Additions:       this.Additions + other.Additions,
		Divisions:       this.Divisions + other.Divisions,
		Exponentiations: this.Exponentiations + other.Exponentiations,
		Comparisons:     this.Comparisons + other.Comparisons,
		General:         this.General + other.General,
	}
}

type MemoryInformation struct {
	Weights    int64 `json:"weights,omitempty"`
	Activation int64 `json:"activation,omitempty"`
}

func (this MemoryInformation) Add(other MemoryInformation) MemoryInformation {
	return MemoryInformation{
		Weights:    this.Weights + other.Weights,
		Activation: this.Activation + other.Activation,
	}
}
