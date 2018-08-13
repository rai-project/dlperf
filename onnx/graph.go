package onnx

import (
	"github.com/rai-project/onnx"
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/simple"
)

type Graph struct {
	Root GraphNode
	*simple.DirectedGraph
}

type GraphNode struct {
	id   int64
	name string
	*onnx.NodeProto
}

func (nd GraphNode) ID() int64 {
	return nd.id
}

func (o Onnx) ToGraph() graph.Directed {
	onnxGraph := o.GetGraph()
	graphIds := map[string]int64{}

	grph := simple.NewDirectedGraph()

	skipNode := func(name string) bool {
		inputs := onnxGraph.GetInput()
		if len(inputs) <= 1 {
			return false
		}
		for _, input := range inputs[1:] {
			if input.GetName() == name {
				return true
			}
		}
		return false
	}

	addNode := func(onnxNode *onnx.NodeProto, name string) {
		if _, ok := graphIds[name]; ok {
			return
		}
		if skipNode(name) {
			return
		}
		id := grph.NewNode().ID()
		nd := GraphNode{
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

	return Graph{
		Root:          grph.Node(inId).(GraphNode),
		DirectedGraph: grph,
	}
}
