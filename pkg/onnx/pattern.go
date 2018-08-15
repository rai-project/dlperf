package onnx

import (
	"sort"
	"strings"

	"github.com/cevaris/ordered_map"
	"github.com/pkg/errors"
	"gonum.org/v1/gonum/graph/topo"
)

type Pattern struct {
	Nodes       GraphNodes
	Occurrences int64
}

type Patterns []Pattern

func (p Pattern) HashKey() string {
	nodeNames := make([]string, len(p.Nodes))
	for ii, nd := range p.Nodes {
		nodeNames[ii] = nd.GetOpType()
	}
	return strings.Join(nodeNames, ">")
}

func (pats Patterns) Counts() Patterns {
	patMap := ordered_map.NewOrderedMap()
	for _, pat := range pats {
		pp0, ok := patMap.Get(pat.HashKey())
		if !ok {
			patMap.Set(pat.HashKey(), pat)
			continue
		}
		pp, ok := pp0.(Pattern)
		if !ok {
			panic("expecting a pattern when finding counts")
		}
		pp.Occurrences += pat.Occurrences
		patMap.Set(pat.HashKey(), pp)
	}

	res := []Pattern{}
	iter := patMap.IterFunc()
	for kv, ok := iter(); ok; kv, ok = iter() {
		res = append(res, kv.Value.(Pattern))
	}
	sort.SliceStable(res, func(ii, jj int) bool {
		return res[ii].Occurrences < res[jj].Occurrences
	})
	return res
}

func (o Onnx) NodeSubsequences(length int) (Patterns, error) {
	grph := o.ToGraph()
	nds, err := topo.SortStabilized(grph, sortById)
	if err != nil {
		return nil, errors.Wrap(err, "failed to topologically sort graph when finding subsequences")
	}

	subsetsLength := len(nds) - length + 1
	result := make([]Pattern, subsetsLength)
	for ii := 0; ii < subsetsLength; ii++ {
		inds := make([]GraphNode, length)
		for jj, nd := range nds[ii : ii+length] {
			inds[jj] = nd.(GraphNode)
		}
		result[ii] = Pattern{
			Nodes:       inds,
			Occurrences: 1,
		}
	}
	return result, nil
}

func NodeSubsequences(length int, models ...Onnx) ([]Pattern, error) {
	pats := Patterns{}
	for _, o := range models {
		ipats, err := o.NodeSubsequences(length)
		if err != nil {
			return nil, errors.Wrap(err, "failed to get Node subsequence for onnx model")
		}
		pats = append(pats, ipats...)
	}
	return pats, nil
}
