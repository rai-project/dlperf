package onnx

import (
	"github.com/rai-project/dlperf"
	"github.com/rai-project/onnx"
)

type Onnx struct {
	*onnx.ModelProto
	*onnx.GraphProto
	nodes            map[string]*onnx.NodeProto
	values           map[string]*onnx.ValueInfoProto
	initializers     map[string]*onnx.TensorProto
	layerInformation map[string]dlperf.LayerInfo
}

func NewOnnx(protoFileName string) (*Onnx, error) {
	model, err := onnx.ReadModel(protoFileName)
	if err != nil {
		return nil, err
	}

	graph := model.GetGraph()
	nodes := map[string]*onnx.NodeProto{}
	for _, n := range graph.Node {
		nodes[n.Name] = n
	}

	values := map[string]*onnx.ValueInfoProto{}
	for _, v := range graph.ValueInfo {
		values[v.Name] = v
	}

	initializers := map[string]*onnx.TensorProto{}
	for _, t := range graph.Initializer {
		initializers[t.Name] = t
	}

	return &Onnx{
		ModelProto:       model,
		nodes:            nodes,
		initializers:     initializers,
		layerInformation: make(map[string]dlperf.LayerInfo),
	}, nil
}
