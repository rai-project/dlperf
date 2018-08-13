package onnx

import (
	"strings"

	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/simple"
	"gonum.org/v1/gonum/graph/topo"
)

var DefaultPrunedLayerTypes = []string{
	"identity",
	"mul",
	"concat",
	"maxpool",
	"sum",
	"reshape",
	"prelu",
	"gemm",
	"dropout",
	"averagepool",
	"transpose",
	"globalaveragepool",
	"lrn",
	"leakyrelu",
	"matmul",
	"imagescaler",
	"unsqueeze",
	"sub",
	"identity",
}

func (g Graph) Prune(layerTypes []string) graph.Directed {

	isSameGraph := func(a, b graph.Directed) bool {
		if a == nil {
			return false
		}
		if b == nil {
			return false
		}
		aNodes, err := topo.SortStabilized(a, nil)
		if err != nil {
			return true
		}
		bNodes, err := topo.SortStabilized(b, nil)
		if err != nil {
			return true
		}
		if len(aNodes) != len(bNodes) {
			return false
		}
		for ii := range aNodes {
			aElem := aNodes[ii]
			bElem := bNodes[ii]
			if aElem.ID() != bElem.ID() {
				return false
			}
		}
		return true
	}

	var new Graph
	old := g.doPrune(layerTypes).(Graph)

	for {
		new = old.doPrune(layerTypes).(Graph)
		if isSameGraph(new, old) {
			break
		}
		old = new
	}
	return new
}

func (g Graph) doPrune(layerTypes []string) graph.Directed {
	var edgeContract func(start, end graph.Node)

	if len(layerTypes) == 0 {
		layerTypes = DefaultPrunedLayerTypes
	}

	toPrune := func(nd0 graph.Node) bool {
		nd, ok := nd0.(GraphNode)
		if !ok {
			return false
		}
		preds := g.To(nd.ID())
		succs := g.From(nd.ID())
		if len(preds) > 1 || len(succs) > 1 {
			return false
		}
		s := strings.ToLower(nd.OpType)
		for _, prune := range layerTypes {
			if s == strings.ToLower(prune) {
				return true
			}
		}
		return false
	}

	newgrph := simple.NewDirectedGraph()

	for _, nd := range g.Nodes() {
		if toPrune(nd) {
			continue
		}
		newgrph.AddNode(nd)
	}

	edgeContract = func(start, end graph.Node) {
		if !toPrune(start) && !toPrune(end) {
			edge := newgrph.NewEdge(start, end)
			newgrph.SetEdge(edge)
		}
		if toPrune(end) {
			for _, succ := range g.From(end.ID()) {
				edgeContract(start, succ)
			}
		}
		if toPrune(start) {
			for _, pred := range g.To(start.ID()) {
				edgeContract(pred, end)
			}
		}
	}

	for _, nd := range g.Nodes() {
		if toPrune(nd) {
			preds := g.To(nd.ID())
			succs := g.From(nd.ID())
			for _, pred := range preds {
				for _, succ := range succs {
					edgeContract(pred, succ)
				}
			}
		} else {
			preds := g.To(nd.ID())
			succs := g.From(nd.ID())
			for _, pred := range preds {
				if toPrune(pred) {
					continue
				}
				edge := newgrph.NewEdge(pred, nd)
				newgrph.SetEdge(edge)
			}
			for _, succ := range succs {
				if toPrune(succ) {
					continue
				}
				edge := newgrph.NewEdge(nd, succ)
				newgrph.SetEdge(edge)
			}
		}
	}
	return Graph{
		Root:          g.Root,
		DirectedGraph: newgrph,
	}
}
