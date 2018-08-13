package onnx

import (
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/path"
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
	return a == dt.DominatorTree.DominatorOf(b)
}

// Algorithm in Fig. 10 from Cytron's classic paper:
//
// Cytron R., Ferrante J., Rosen B. K., and Wegman M. N. "Efficiently Computing
// Static Single Assignment Form and the Control Dependence Graph." ACM TOPLAS.
// https://doi.org/10.1145/115372.1
// func (t *DominatorTree) Frontier() *DominatorFrontier {
// 	if t.frontier != nil {
// 		return t.frontier
// 	}
// 	frontier := make(map[*Block]map[*Block]bool)
// 	var postfix func(*Block)
// 	postfix = func(blk *Block) {
// 		for _, kid := range t.Children(blk) {
// 			postfix(kid)
// 		}
// 		frontier[blk] = make(map[*Block]bool)
// 		for _, y := range t.succ(blk) {
// 			if t.IDom(y) != blk {
// 				frontier[blk][y] = true
// 			}
// 		}
// 		for _, kid := range t.Children(blk) {
// 			for y := range frontier[kid] {
// 				if t.IDom(y) != blk {
// 					frontier[blk][y] = true
// 				}
// 			}
// 		}
// 	}
// 	for _, r := range t.roots {
// 		postfix(r)
// 	}
// 	t.frontier = &DominatorFrontier{frontier}
// 	return t.frontier
// }

// func (t *DominatorTree) ImmediateDominators() []int {
// 	idom := make([]int, len(t.parent))
// 	for child, parent := range t.parent {
// 		if parent != nil {
// 			idom[child.Id] = parent.Id
// 		} else {
// 			idom[child.Id] = child.Id
// 		}
// 	}
// 	return idom
// }
