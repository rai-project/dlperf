package cmd

import (
	"fmt"
	"image/color"
	"math"
	"time"

	"github.com/spf13/cast"

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
	*onnx.Graph
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
			Value: "\"LR\"", // or "\"TB\"" for top to bottom view
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

func makeBenchmarkGraph(model *onnx.Onnx, net *onnx.Graph, nds []benchmarkGraphNode, timeTransformFunction func(time.Duration) float64) *benchmarkGraph {
	// model := &onnx.Onnx{}
	// // deepcopy.Copy(&model, model0)
	// model = model0

	getWeight := func(grNode graph.Node) float64 {
		onnxNode, ok := grNode.(*onnx.GraphNode)
		if !ok {
			pp.Println(grNode)
			panic(grNode)
		}
		for _, nd := range nds {
			if nd.Name == onnxNode.Name {
				t := nd.forwardRuntime()
				onnxNode.SetRuntime(t)
				return timeTransformFunction(t)
			}
		}
		return math.Inf(1)
	}

	colors := mkColors(nds)
	// net := model.ToGraph(onnx.GraphPruneInputs(true))

	for _, edge := range graph.WeightedEdgesOf(net.WeightedEdges()) {
		onnxEdge := edge.(*onnx.GraphEdge)

		to := onnxEdge.To().(*onnx.GraphNode)
		from := onnxEdge.From().(*onnx.GraphNode)

		weight := getWeight(from)

		if from.ID() == to.ID() {
			continue
		}

		if false {
			pp.Println(from.Name + idString(from) + " -> " + to.Name + idString(to) + "  " + cast.ToString(weight))
		}
		net.SetWeightedEdge(&onnx.GraphEdge{
			WeightedEdge: net.NewWeightedEdge(from, to, weight),
		})
	}

	return &benchmarkGraph{
		nodes:  nds,
		colors: colors,
		Graph:  net,
	}
}

func idString(nd graph.Node) string {
	return fmt.Sprintf("(%d)", nd.ID())
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
