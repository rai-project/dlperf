package onnx

import (
	"errors"

	"github.com/cevaris/ordered_map"

	"github.com/rai-project/onnx"
)

type Onnx struct {
	*onnx.ModelProto
	*onnx.GraphProto
	path         string
	network      *Graph
	nodes        *ordered_map.OrderedMap // map[string]*onnx.NodeProto
	valueInfo    map[string]*onnx.ValueInfoProto
	inputs       map[string]*onnx.ValueInfoProto
	outputs      map[string]*onnx.ValueInfoProto
	initializers map[string]*onnx.TensorProto
	batchSize    int64
}

func New(protoFileName string, batchSize int64) (*Onnx, error) {
	model, err := onnx.New(protoFileName, onnx.Steps([]string{}))

	if err != nil {
		return nil, err
	}

	graph := model.GetGraph()
	nodes := ordered_map.NewOrderedMap()
	for _, n := range graph.Node {
		nodes.Set(n.Name, n)
	}

	valueInfo := map[string]*onnx.ValueInfoProto{}
	for _, v := range graph.ValueInfo {
		valueInfo[v.Name] = v
	}

	inputs := map[string]*onnx.ValueInfoProto{}
	if len(graph.Input) == 0 {
		return nil, errors.New("the onnx model has no input")
	}
	input0 := graph.Input[0]
	input0Shape := getValueInfoDimensions(input0)
	if len(input0Shape) != 4 {
		return nil, errors.New("supports image input")
	}
	dim := input0.GetType().GetTensorType().GetShape().GetDim()[0]
	val := onnx.TensorShapeProto_Dimension_DimValue{DimValue: batchSize}
	dim.Value = &val

	for _, i := range graph.Input {
		// Assume the input is image and len(shape) == 4 && shape[0] == 1
		shape := getValueInfoDimensions(i)

		inputs[i.Name] = i
	}

	outputs := map[string]*onnx.ValueInfoProto{}
	for _, o := range graph.Output {
		outputs[o.Name] = o
	}

	initializers := map[string]*onnx.TensorProto{}
	for _, i := range graph.Initializer {
		initializers[i.Name] = i
	}

	o := &Onnx{
		path:         protoFileName,
		ModelProto:   model,
		nodes:        nodes,
		valueInfo:    valueInfo,
		inputs:       inputs,
		outputs:      outputs,
		initializers: initializers,
		batchSize:    batchSize,
	}

	o.Information()

	return o, nil
}

func (o Onnx) Network() *Graph {
	return o.network
}

func (o Onnx) BatchSize() int64 {
	return o.batchSize
}
