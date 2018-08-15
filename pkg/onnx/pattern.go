package onnx

import (
	"strings"

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
	patMap := map[string]*Pattern{}
	for _, pat := range pats {
		if _, ok := patMap[pat.HashKey()]; !ok {
			patMap[pat.HashKey()] = &pat
			continue
		}
		patMap[pat.HashKey()].Occurrences += pat.Occurrences
	}
	res := []Pattern{}
	for _, pat := range pats {
		res = append(res, pat)
	}
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
