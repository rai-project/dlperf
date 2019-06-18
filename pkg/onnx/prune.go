package onnx

import (
	"math"
	"strings"

	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/simple"
	"gonum.org/v1/gonum/graph/topo"
)

var DefaultPrunedLayerTypes = []string{
	"identity",
	"Relu",
	"mul",
	"concat",
	"maxpool",
	"sum",
	"reshape",
	"pRelu",
	"gemm",
	"dropout",
	"averagepool",
	"transpose",
	"globalaveragepool",
	"lrn",
	"leakyRelu",
	"matmul",
	"imagescaler",
	"unsqueeze",
	"sub",
	"identity",
}

func (g Graph) Prune(layerTypes []string) *Graph {

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

	var new *Graph
	old := g.doPrune(layerTypes)

	for {
		new = old.doPrune(layerTypes)
		if isSameGraph(new, old) {
			break
		}
		old = new
	}
	return new
}

func (g Graph) doPrune(layerTypes []string) *Graph {
	var edgeContract func(start, end graph.Node)

	if len(layerTypes) == 0 {
		layerTypes = DefaultPrunedLayerTypes
	}

	toPrune := func(nd0 graph.Node) bool {
		nd, ok := nd0.(*GraphNode)
		if !ok {
			return false
		}
		preds := g.To(nd.ID())
		succs := g.From(nd.ID())
		if preds.Len() > 1 || succs.Len() > 1 {
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

	newgrph := simple.NewWeightedDirectedGraph(0, math.Inf(1))

	nds := g.Nodes()
	for nds.Next() {
		nd := nds.Node()
		if toPrune(nd) {
			continue
		}
		newgrph.AddNode(nd)
	}

	edgeContract = func(start, end graph.Node) {
		if !toPrune(start) && !toPrune(end) {
			newgrph.SetWeightedEdge(&GraphEdge{
				WeightedEdge: simple.WeightedEdge{
					F: start,
					T: end,
					W: 1,
				},
			})
		}
		if toPrune(end) {
			succs := g.From(end.ID())
			for succs.Next() {
				succ := succs.Node()
				edgeContract(start, succ)
			}
		}
		if toPrune(start) {
			preds := g.To(start.ID())
			for preds.Next() {
				pred := preds.Node()
				edgeContract(pred, end)
			}
		}
	}

	grphNodes := g.Nodes()
	for grphNodes.Next() {
		nd := grphNodes.Node()
		if toPrune(nd) {
			preds := g.To(nd.ID())
			succs := g.From(nd.ID())
			for preds.Next() {
				pred := preds.Node()
				for succs.Next() {
					succ := succs.Node()
					edgeContract(pred, succ)
				}
			}
			continue
		}
		preds := g.To(nd.ID())
		succs := g.From(nd.ID())
		for preds.Next() {
			pred := preds.Node()
			if toPrune(pred) {
				continue
			}
			newgrph.SetWeightedEdge(&GraphEdge{
				WeightedEdge: simple.WeightedEdge{
					F: pred,
					T: nd,
					W: 1,
				},
			})
		}
		for succs.Next() {
			succ := succs.Node()
			if toPrune(succ) {
				continue
			}
			newgrph.SetWeightedEdge(&GraphEdge{
				WeightedEdge: simple.WeightedEdge{
					F: nd,
					T: succ,
					W: 1,
				},
			})
		}
	}
	return &Graph{
		Root:                  g.Root,
		WeightedDirectedGraph: newgrph,
	}
}
