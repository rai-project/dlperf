package onnx

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/pkg/errors"
	dlperf "github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/onnx"
	"gonum.org/v1/gonum/graph/encoding"
	"gonum.org/v1/gonum/graph/simple"
	"gonum.org/v1/gonum/graph/topo"
)

type Graph struct {
	Root *GraphNode
	*simple.DirectedGraph
}

type GraphNode struct {
	id    int64
	name  string
	layer dlperf.Layer
	*onnx.NodeProto
}

type GraphNodes []*GraphNode

func (nd GraphNode) ID() int64 {
	return nd.id
}

func (nd GraphNode) Layer() dlperf.Layer {
	return nd.layer
}

func (nd GraphNode) DOTID() string {
	return fmt.Sprintf("\"%s\"", nd.name)
}

func (nd GraphNode) Attributes() []encoding.Attribute {
	lbl := fmt.Sprintf("\"{  %s  | %s }\"", nd.Name, nd.OpType)
	if nd.layer != nil {
		outputShapes, err := json.Marshal(nd.layer.OutputShapes())
		if err == nil {
			lbl = fmt.Sprintf("\"{  %s  | %s | %s }\"", nd.Name, nd.OpType, string(outputShapes))
		}
	}
	return []encoding.Attribute{
		encoding.Attribute{
			Key:   "id",
			Value: fmt.Sprintf("%v", nd.ID()),
		},
		encoding.Attribute{
			Key:   "name",
			Value: fmt.Sprintf("\"%s\"", nd.name),
		},
		encoding.Attribute{
			Key:   "type",
			Value: nd.OpType,
		},
		encoding.Attribute{
			Key:   "shape",
			Value: "record",
		},
		encoding.Attribute{
			Key:   "label",
			Value: lbl,
		},
		encoding.Attribute{
			Key:   "inputs",
			Value: fmt.Sprintf("\"%s\"", strings.Join(nd.GetInput(), ";")),
		},
		encoding.Attribute{
			Key:   "outputs",
			Value: fmt.Sprintf("\"%s\"", strings.Join(nd.GetOutput(), ";")),
		},
	}
}

func (o *Onnx) ToGraph(oo ...GraphOption) *Graph {
	opts := NewGraphOptions(oo...)
	onnxGraph := o.GetGraph()
	graphIds := map[string]int64{}

	grph := simple.NewDirectedGraph()

	isInputNode := func(name string) bool {
		inputs := onnxGraph.GetInput()
		if len(inputs) <= 1 {
			return false
		}
		if !opts.InputsAsConstantNodes {
			inputs = inputs[1:]
		}
		for _, input := range inputs {
			if input.GetName() == name {
				return true
			}
		}
		return false
	}

	isOutputNode := func(name string) bool {
		outputs := onnxGraph.GetOutput()
		if len(outputs) <= 1 {
			return false
		}
		for _, output := range outputs {
			if output.GetName() == name {
				return true
			}
		}
		return false
	}
	_ = isOutputNode

	mkOnnxConstantInputNode := func(name string, src *onnx.NodeProto) *onnx.NodeProto {
		return &onnx.NodeProto{
			Input:  []string{},
			Name:   name,
			OpType: "constant_input",
		}
	}

	mkOnnxConstantOutputNode := func(name string, src *onnx.NodeProto) *onnx.NodeProto {
		return &onnx.NodeProto{
			Input:  []string{src.Name},
			Name:   name,
			OpType: "constant_output",
		}
	}

	_ = mkOnnxConstantOutputNode

	skipNode := func(name string) bool {
		// if strings.HasPrefix(name, "_") {
		// 	return true
		// }
		if !opts.PruneInputs {
			return false
		}
		return isInputNode(name)
	}

	addNode := func(onnxNode *onnx.NodeProto, name string) {
		if _, ok := graphIds[name]; ok {
			return
		}
		if skipNode(name) {
			return
		}
		id := grph.NewNode().ID()
		if opts.InputsAsConstantNodes && isInputNode(name) {
			onnxNode = mkOnnxConstantInputNode(name, onnxNode)
		} else {
			onnxNode.Attribute = append(onnxNode.Attribute, &onnx.AttributeProto{
				Name: "edge_name",
				S:    []byte(name),
			})
		}
		nd := &GraphNode{
			id:        id,
			name:      name,
			NodeProto: onnxNode,
		}
		grph.AddNode(nd)
		graphIds[name] = nd.ID()
	}

	for _, onnxNode := range onnxGraph.Node {
		for _, inputNode := range onnxNode.GetInput() {
			addNode(onnxNode, inputNode)
		}
		for _, outputNode := range onnxNode.GetOutput() {
			addNode(onnxNode, outputNode)
		}
	}

	for _, nd := range onnxGraph.GetNode() {
		for _, inputNode := range nd.GetInput() {
			if skipNode(inputNode) {
				continue
			}
			for _, outputNode := range nd.GetOutput() {
				if skipNode(outputNode) {
					continue
				}
				inId := graphIds[inputNode]
				outId := graphIds[outputNode]
				if inId == outId {
					continue
				}
				inNd := grph.Node(inId)
				outNd := grph.Node(outId)
				edge := grph.NewEdge(inNd, outNd)
				grph.SetEdge(edge)
			}
		}
	}

	input := onnxGraph.GetInput()[0]
	inId := graphIds[input.GetName()]

	network := &Graph{
		Root:          grph.Node(inId).(*GraphNode),
		DirectedGraph: grph,
	}

	o.network = network

	return network
}

func (o Onnx) TopologicallyOrderedNodes(oo ...GraphOption) (GraphNodes, error) {
	g := o.ToGraph(oo...)
	return g.TopologicallyOrderedNodes()
}

func (g Graph) TopologicallyOrderedNodes() (GraphNodes, error) {
	nds, err := topo.SortStabilized(g, sortById)
	if err != nil {
		return nil, errors.Wrap(err, "failed to topologically sort graph")
	}
	res := make([]*GraphNode, len(nds))
	for ii, nd := range nds {
		e, ok := nd.(*GraphNode)
		if !ok {
			panic("expecting a *GraphNode")
		}
		res[ii] = e
	}
	return res, nil
}
