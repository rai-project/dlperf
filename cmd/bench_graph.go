package cmd

import (
	"image/color"
	"math"
	"time"

	"github.com/muesli/gamut"
	"github.com/rai-project/dlperf/pkg/onnx"
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
	// for _, onnxNode := range model.GetNode() {
	// 	for _, inputNode := range onnxNode.GetInput() {
	// 		grph.AddNode(inputNode)
	// 		graphIds[inputNode.Name] = inputNode.ID()
	// 	}
	// 	for _, outputNode := range onnxNode.GetOutput() {
	// 		grph.AddNode(outputNode)
	// 		graphIds[outputNode.Name] = outputNode.ID()
	// 	}
	// }

	for _, nd := range nds {
		grph.AddNode(nd)
		graphIds[nd.Name] = nd.ID()
	}

	for _, nd := range nds {
		for _, inputNode := range nd.GetInput() {
			for _, outputNode := range nd.GetOutput() {
				inId, ok := graphIds[inputNode]
				if !ok {
					continue
				}
				outId, ok := graphIds[outputNode]
				if !ok {
					continue
				}
				if inId == outId {
					continue
				}
				inNd := grph.Node(inId)
				outNd := grph.Node(outId)

				grph.SetWeightedEdge(grph.NewWeightedEdge(inNd, outNd, toMicroSeconds(nd.forwardRuntime())))
			}
		}
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
