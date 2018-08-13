package onnx

import (
	"sort"

	"github.com/pkg/errors"
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/path"
	"gonum.org/v1/gonum/graph/simple"
	"gonum.org/v1/gonum/graph/topo"
)

// A DominatorTree represents a dominator tree.
type DominatorTree struct {
	path.DominatorTree
}

func (o Onnx) Dominators() DominatorTree {
	grph := o.ToGraph()
	return DominatorTree{
		path.DominatorsSLT(grph.(Graph).Root, grph),
	}
}

// Dominates reports whether A dominates B.
func (dt DominatorTree) Dominates(a, b graph.Node) bool {
	return a.ID() == dt.DominatorOf(b).ID()
}

func (dt DominatorTree) Dominated(a, b graph.Node) bool {
	doms := dt.DominatedBy(a)
	for _, dom := range doms {
		if dom.ID() == b.ID() {
			return true
		}
	}
	return false
}

// byID implements the sort.Interface sorting a slice of graph.Node by reverse ID.
type byID []graph.Node

func (n byID) Len() int           { return len(n) }
func (n byID) Less(i, j int) bool { return n[i].ID() < n[j].ID() }
func (n byID) Swap(i, j int)      { n[i], n[j] = n[j], n[i] }

// byReverseID implements the sort.Interface sorting a slice of graph.Node by reverse ID.
type byReverseID []graph.Node

func (n byReverseID) Len() int           { return len(n) }
func (n byReverseID) Less(i, j int) bool { return n[i].ID() > n[j].ID() }
func (n byReverseID) Swap(i, j int)      { n[i], n[j] = n[j], n[i] }

func (o Onnx) FindGraphGroups() ([]graph.Directed, error) {
	visited := map[int64]bool{}
	res := []graph.Directed{}
	grph := o.ToGraph()
	dt := o.Dominators()
	nds, err := topo.SortStabilized(grph, func(nodes []graph.Node) { sort.Sort(byID(nodes)) })
	if err != nil {
		return nil, errors.Wrap(err, "failed to topologically sort graph")
	}

	captureGroup := func(root graph.Node) {
		var visit func(graph.Node)

		subgrph := simple.NewDirectedGraph()
		// visited[root.ID()] = true

		visit = func(nd graph.Node) {
			if _, ok := visited[nd.ID()]; ok {
				return
			}
			visited[nd.ID()] = true
			subgrph.AddNode(nd)

			for _, pred := range grph.To(nd.ID()) {
				if dt.Dominates(pred, nd) {
					visit(pred)
					edge := subgrph.NewEdge(nd, pred)
					subgrph.SetEdge(edge)
				}
			}
		}

		visit(root)

		succs := grph.From(root.ID())
		for _, succ := range succs {
			visit(succ)
		}

		if len(subgrph.Nodes()) != 0 {
			res = append(res, subgrph)
		}
	}

	for _, nd := range nds {
		// if _, ok := visited[nd.ID()]; ok {
		// 	continue
		// }
		preds := grph.To(nd.ID())
		for _, pred := range preds {
			if dt.Dominates(pred, nd) {
				continue
			}
			// pp.Println(pred, "   ", nd)
			captureGroup(nd)
			break
		}
	}
	return res, nil
}
