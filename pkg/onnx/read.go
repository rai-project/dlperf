package onnx

import (
	"github.com/cevaris/ordered_map"
	"github.com/rai-project/onnx"
)

type Onnx struct {
	*onnx.ModelProto
	*onnx.GraphProto
	network      *Graph
	nodes        *ordered_map.OrderedMap // map[string]*onnx.NodeProto
	valueInfo    map[string]*onnx.ValueInfoProto
	inputs       map[string]*onnx.ValueInfoProto
	outputs      map[string]*onnx.ValueInfoProto
	initializers map[string]*onnx.TensorProto
}

func New(protoFileName string) (*Onnx, error) {

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
	for _, i := range graph.Input {
		inputs[i.Name] = i
	}

	outputs := map[string]*onnx.ValueInfoProto{}
	for _, o := range graph.Output {
		outputs[o.Name] = o
	}

	initializers := map[string]*onnx.TensorProto{}
	for _, i := range graph.Initializer {
		// if i.Name == "OC2_DUMMY_1" {
		// 	pp.Println(i)
		// 	pp.Println(i.DataType.String())
		// 	buf := bytes.NewBuffer(i.RawData)
		// 	var f1 float32
		// 	binary.Read(buf, binary.LittleEndian, &f1)
		// 	pp.Println(f1)
		// 	// pp.Println(int64(binary.LittleEndian.Uint64(i.RawData[8:])))
		// }
		initializers[i.Name] = i
	}

	o := &Onnx{
		ModelProto:   model,
		nodes:        nodes,
		valueInfo:    valueInfo,
		inputs:       inputs,
		outputs:      outputs,
		initializers: initializers,
	}

	o.Information()

	return o, nil
}

func (o Onnx) Network() *Graph {
	return o.network
}
