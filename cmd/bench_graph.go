package cmd

import (
	"image/color"
	"math"
	"time"

	"github.com/k0kubun/pp"
	"github.com/muesli/gamut"
	"github.com/rai-project/dlperf/pkg/onnx"
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/encoding"
	"gonum.org/v1/gonum/graph/simple"
)

type benchmarkGraph struct {
	Root   *benchmarkGraphNode
	nodes  []benchmarkGraphNode
	colors map[string]color.Color
	*simple.WeightedDirectedGraph
}

type benchmarkGraphNode struct {
	*onnx.GraphNode
	Benchmarks []*bench `json:"benchmarks,omitempty"`
}

type benchmarkGraphEdge struct {
	simple.WeightedEdge
}

type benchmarkGraphNodes []*benchmarkGraphNode

type gAttributer struct{}
type nAttributer struct{}
type eAttributer struct{}

func (gAttributer) Attributes() []encoding.Attribute {
	return []encoding.Attribute{
		encoding.Attribute{
			Key:   "rankdir",
			Value: "\"TB\"",
		},
		encoding.Attribute{
			Key:   "overlap",
			Value: "prism",
		},
		encoding.Attribute{
			Key:   "overlap_shrink",
			Value: "true",
		},
		encoding.Attribute{
			Key:   "splines",
			Value: "curved", //"polyline",
		},
	}
}

func (nAttributer) Attributes() []encoding.Attribute {
	return []encoding.Attribute{
		encoding.Attribute{
			Key:   "shape",
			Value: "Mrecord",
		},
	}
}

func (eAttributer) Attributes() []encoding.Attribute {
	return []encoding.Attribute{
		encoding.Attribute{
			Key:   "penwidth",
			Value: "3",
		},
	}
}

func (b benchmarkGraphNode) forwardRuntime() time.Duration {
	accum := time.Duration(0)
	for _, bench := range b.Benchmarks {
		if bench.Type == "forward" || bench.Type == "bias" {
			accum += bench.Benchmark.RealTime
		}
	}
	return accum
}

func makeBenchmarkGraph(model *onnx.Onnx, nds []benchmarkGraphNode) *benchmarkGraph {
	graphIds := map[string]int64{}

	colors := mkColors(nds)
	grph := simple.NewWeightedDirectedGraph(0, math.Inf(1))
	onnxGrph := model.ToGraph()
	ndIter := onnxGrph.Nodes()
	for ndIter.Next() {
		onnxNode, ok := ndIter.Node().(*onnx.GraphNode)
		if !ok {
			pp.Println(onnxNode)
			panic("invalid node. expecting an onnx node")
		}
		grph.AddNode(onnxNode)
		graphIds[onnxNode.Name] = onnxNode.ID()
	}

	getWeight := func(grNode graph.Node) float64 {
		onnxNode, ok := grNode.(*onnx.GraphNode)
		if !ok {
			pp.Println(grNode)
			panic(grNode)
		}
		for _, nd := range nds {
			if nd.Name == onnxNode.Name {
				t := nd.forwardRuntime()
				onnxNode.SetRuntime(nd.forwardRuntime())
				return toMicroSeconds(t)
			}
		}
		return math.Inf(1)
	}

	edgeIter := onnxGrph.Edges()
	for edgeIter.Next() {
		onnxEdge := edgeIter.Edge()
		weight := getWeight(onnxEdge.From())
		grph.SetWeightedEdge(grph.NewWeightedEdge(onnxEdge.From(), onnxEdge.To(), weight))
	}

	return &benchmarkGraph{
		nodes:                 nds,
		colors:                colors,
		WeightedDirectedGraph: grph,
	}
}

func toMicroSeconds(t time.Duration) float64 {
	return float64(t) / float64(time.Microsecond)
}

func (g benchmarkGraph) DOTAttributers() (graph, node, edge encoding.Attributer) {
	graph = gAttributer{}
	node = nAttributer{}
	edge = eAttributer{}
	return
}

func (nd benchmarkGraphNode) OperatorType() string {
	return nd.Layer().OperatorType()
}

func (nd benchmarkGraphEdge) Attributes() []encoding.Attribute {
	return []encoding.Attribute{}
}

func (o *benchmarkGraph) mkColors() map[string]color.Color {
	return mkColors(o.nodes)
}

func mkColors(nds []benchmarkGraphNode) map[string]color.Color {
	ndColors := map[string]color.Color{}
	for _, nd := range nds {
		ndColors[nd.Layer().OperatorType()] = nil
	}
	colors, err := gamut.Generate(len(ndColors), gamut.PastelGenerator{})
	if err != nil {
		return ndColors
	}
	ii := 0
	for k := range ndColors {
		color := colors[ii]
		ndColors[k] = color
		ii++
	}

	return ndColors
}
