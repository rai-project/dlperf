package onnx

import (
	"github.com/cevaris/ordered_map"
	"github.com/rai-project/dlperf"
	"github.com/rai-project/onnx"
)

type Onnx struct {
	*onnx.ModelProto
	*onnx.GraphProto
	nodes            *ordered_map.OrderedMap // map[string]*onnx.NodeProto
	valueInfo        map[string]*onnx.ValueInfoProto
	inputs           map[string]*onnx.ValueInfoProto
	outputs          map[string]*onnx.ValueInfoProto
	initializers     map[string]*onnx.TensorProto
	layerInformation map[string]dlperf.LayerInfo
}

func NewOnnx(protoFileName string) (*Onnx, error) {

	model, err := onnx.ReadModel(protoFileName)
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
	for _, i := range graph.Input {
		inputs[i.Name] = i
	}

	outputs := map[string]*onnx.ValueInfoProto{}
	for _, o := range graph.Output {
		outputs[o.Name] = o
	}

	initializers := map[string]*onnx.TensorProto{}
	for _, t := range graph.Initializer {
		initializers[t.Name] = t
	}

	return &Onnx{
		ModelProto:       model,
		nodes:            nodes,
		valueInfo:        valueInfo,
		inputs:           inputs,
		outputs:          outputs,
		initializers:     initializers,
		layerInformation: make(map[string]dlperf.LayerInfo),
	}, nil
}
