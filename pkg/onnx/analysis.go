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

func (grph Graph) Dominators() DominatorTree {
	return DominatorTree{
		path.DominatorsSLT(grph.Root, grph),
	}
}

func (o Onnx) Dominators() DominatorTree {
	grph := o.ToGraph()
	return grph.Dominators()
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

func sortById(nodes []graph.Node) {
	sort.Sort(byID(nodes))
}

func (o Onnx) FindSubGraphs() ([]Graph, error) {
	grph := o.ToGraph()
	return grph.FindSubGraphs()
}

func (grph Graph) FindSubGraphs() ([]Graph, error) {
	visited := map[int64]bool{}
	res := []Graph{}
	dt := grph.Dominators()
	nds, err := topo.SortStabilized(grph, sortById)
	if err != nil {
		return nil, errors.Wrap(err, "failed to topologically sort graph")
	}

	captureGroup := func(root, sink graph.Node) {
		var visit func(graph.Node)

		if root == nil {
			return
		}

		subgrph := simple.NewDirectedGraph()

		visit = func(nd graph.Node) {
			if nd == nil {
				return
			}
			if _, ok := visited[nd.ID()]; ok {
				return
			}
			visited[nd.ID()] = true
			subgrph.AddNode(nd)

			for _, succ := range grph.From(nd.ID()) {
				if dt.Dominates(nd, succ) {
					visit(succ)
					edge := subgrph.NewEdge(nd, succ)
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
			res = append(res, Graph{
				Root:          root.(GraphNode),
				DirectedGraph: subgrph,
			})
		}
	}

	for _, nd := range nds {
		if _, ok := visited[nd.ID()]; ok {
			continue
		}
		var start, elem graph.Node
		wrkgrp := grph.To(nd.ID())
		for len(wrkgrp) > 0 {
			elem, wrkgrp = wrkgrp[0], wrkgrp[1:]
			if _, ok := visited[elem.ID()]; ok {
				continue
			}
			if dt.Dominates(elem, nd) {
				start = elem
				break
			}
			wrkgrp = append(wrkgrp, grph.To(elem.ID())...)
		}
		if start == nil {
			continue
		}
		if len(grph.From(start.ID())) > 1 {
			captureGroup(start, nd)
		}
	}
	return res, nil
}
