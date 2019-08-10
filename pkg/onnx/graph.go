package onnx

import (
	"encoding/json"
	"fmt"
	"image/color"
	"math"
	"strings"
	"time"

	"github.com/fatih/set"
	"github.com/kljensen/snowball"
	colorful "github.com/lucasb-eyer/go-colorful"
	"github.com/muesli/gamut"
	"github.com/pkg/errors"
	dlperf "github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/dlperf/pkg/benchmark"
	"github.com/rai-project/onnx"
	"github.com/spf13/cast"
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/encoding"
	"gonum.org/v1/gonum/graph/simple"
	"gonum.org/v1/gonum/graph/topo"
)

type Graph struct {
	Root   *GraphNode
	colors map[string]color.Color
	*simple.WeightedDirectedGraph
}

type GraphNode struct {
	id         int64
	name       string
	benchmarks benchmark.Benchmarks
	layer      dlperf.Layer
	color      color.Color
	runtime    *time.Duration
	*onnx.NodeProto
}

type GraphEdge struct {
	graph.WeightedEdge
	color color.Color
}

type GraphNodes []*GraphNode

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
		// encoding.Attribute{
		// 	Key:   "pencolor",
		// 	Value: "black",
		// },
	}
}

func (g Graph) DOTAttributers() (graph, node, edge encoding.Attributer) {
	graph = gAttributer{}
	node = nAttributer{}
	edge = eAttributer{}
	return
}

func (nd *GraphNode) SetID(id int64) {
	nd.id = id
}

func (nd GraphNode) ID() int64 {
	return nd.id
}

func (nd GraphNode) Layer() dlperf.Layer {
	return nd.layer
}

func (nd GraphNode) GetRuntime() *time.Duration {
	return nd.runtime
}

func (nd *GraphNode) SetRuntime(t time.Duration) {
	nd.runtime = &t
}

func (nd GraphNode) DOTID() string {
	return fmt.Sprintf("\"%s\"", nd.name)
}

func (nd *GraphNode) SetBenchmarks(bs benchmark.Benchmarks) {
	nd.benchmarks = bs
}

func (nd *GraphNode) Benchmarks() benchmark.Benchmarks {
	return nd.benchmarks
}

func (nd GraphNode) Attributes() []encoding.Attribute {
	var lbl string
	extraAttrs := []encoding.Attribute{}
	if strings.ToLower(nd.OpType) == "constant_input" || strings.ToLower(nd.OpType) == "constantinput" {
		lbl = fmt.Sprintf("\"%s\"", nd.Name)
		extraAttrs = append(extraAttrs,
			[]encoding.Attribute{
				encoding.Attribute{
					Key:   "shape",
					Value: "box",
				},
				encoding.Attribute{
					Key:   "style",
					Value: `"filled,dashed"`,
				},
				encoding.Attribute{
					Key:   "fillcolor",
					Value: `"white"`,
				},
			}...,
		)
	} else {
		lbl = fmt.Sprintf("\"{%s}  | {%s}\"", nd.Name, shortenOpType(nd.OpType))
		if nd.layer != nil {
			var outputShapesBuf []byte
			var err error
			outShapes := nd.layer.OutputShapes()
			if len(outShapes) == 1 {
				outputShapesBuf, err = json.Marshal(outShapes[0])
			} else {
				outputShapesBuf, err = json.Marshal(outShapes)
			}
			if err == nil {
				lbl = fmt.Sprintf(`"{ {%s  | %s} | %s }"`, nd.Name, shortenOpType(nd.OpType), string(outputShapesBuf))
			}
		}
		if nd.runtime != nil {
			lbl = fmt.Sprintf(`"{ {%s  | %s} | %s }"`, nd.Name, shortenOpType(nd.OpType), cast.ToString(*nd.runtime))
		}
	}
	attrs := []encoding.Attribute{
		encoding.Attribute{
			Key:   "id",
			Value: fmt.Sprintf("%v", nd.ID()),
		},
		encoding.Attribute{
			Key:   "name",
			Value: fmt.Sprintf("\"%s\"", nd.name),
		},
		encoding.Attribute{
			Key:   "type",
			Value: nd.OpType,
		},
		encoding.Attribute{
			Key:   "label",
			Value: lbl,
		},
		encoding.Attribute{
			Key:   "inputs",
			Value: fmt.Sprintf("\"%s\"", strings.Join(nd.GetInput(), ";")),
		},
		encoding.Attribute{
			Key:   "outputs",
			Value: fmt.Sprintf("\"%s\"", strings.Join(nd.GetOutput(), ";")),
		},
	}
	if nd.runtime != nil {
		attrs = append(attrs,
			encoding.Attribute{
				Key:   "runtime",
				Value: cast.ToString(*nd.runtime),
			},
		)
	}
	for _, bench := range nd.benchmarks {
		benchAttr := encoding.Attribute{
			Key:   bench.Name,
			Value: cast.ToString(bench.RealTime),
		}
		attrs = append(attrs, benchAttr)
	}
	attrs = append(attrs, extraAttrs...)
	if nd.color != nil {

		color := nd.color
		fillColor := color
		// if nd.layer != nil && len(nd.layer.OutputShapes()) != 0 {
		// 	outputShape := nd.layer.OutputShapes()[0]
		// 	bytes := int64(1)
		// 	for _, b := range outputShape {
		// 		bytes *= b
		// 	}
		// 	println(float64(bytes) / float64(1024))
		// 	fillColor = gamut.Lighter(fillColor, float64(bytes)/float64(1024))
		// }
		attrs = append(
			attrs,
			[]encoding.Attribute{
				encoding.Attribute{
					Key:   "penwidth",
					Value: "3",
				},
				encoding.Attribute{
					Key:   "style",
					Value: "filled",
				},
				encoding.Attribute{
					Key:   "color",
					Value: colorToHex(gamut.Darker(color, 0.1)),
				},
				encoding.Attribute{
					Key:   "fontcolor",
					Value: colorToHex(gamut.Contrast(color)),
				},
				encoding.Attribute{
					Key:   "fillcolor",
					Value: colorToHex(fillColor),
				},
			}...,
		)
	}
	return attrs
}

func colorToHex(clr0 color.Color) string {
	clr, ok := colorful.MakeColor(clr0)
	if ok {
		return fmt.Sprintf(`"%s"`, clr.Hex())
	}
	return ""
}

func shortenOpType(ty string) string {
	stemmed, err := snowball.Stem(ty, "english", true)
	if err == nil {
		return iShortenOpType(stemmed)
	}
	return iShortenOpType(ty)
}

func iShortenOpType(ty0 string) string {
	ty := strings.ToLower(ty0)
	switch ty {
	case "batchnorm", "bn":
		return "BN"
	case "conv", "convolution":
		return "CONV"
	case "reshap":
		return "RSHP"
	case "add":
		return "ADD"
	case "sub":
		return "SUB"
	case "mul":
		return "MUL"
	case "relu":
		return "RELU"
	case "prelu":
		return "PRELU"
	case "unsqueez":
		return "UNSQ"
	case "concat":
		return "CONC"
	case "averagepool":
		return "AVGPL"
	case "globalaveragepool":
		return "GLBPL"
	case "lrn":
		return "LRN"
	case "softmax":
		return "SFT"
	case "maxpool":
		return "MXPL"
	case "flatten":
		return "FLT"
	case "gemm":
		return "GEMM"
	case "matmul":
		return "MM"
	case "ident":
		return "IDNT"
	case "dropout":
		return "DRP"
	default:
		fmt.Println(ty0)
		return ty0
	}
}

func (nd GraphEdge) Color() color.Color {
	return nd.color
}

func (nd *GraphEdge) SetColor(clr color.Color) {
	nd.color = clr
}

func (nd *GraphEdge) Highlight() {
	clr, err := colorful.Hex("#C65840")
	if err != nil {
		clr, _ = colorful.Hex("#FF0000")
	}
	nd.color = &clr
}

func (nd GraphEdge) Attributes() []encoding.Attribute {
	attrs := []encoding.Attribute{}
	if nd.color != nil {
		attrs = append(attrs,
			encoding.Attribute{
				Key:   "penwidth",
				Value: "5",
			},
			encoding.Attribute{
				Key:   "pencolor",
				Value: colorToHex(nd.color),
			},
			encoding.Attribute{
				Key:   "color",
				Value: colorToHex(nd.color),
			},
		)
	}
	return attrs
}

func (o *Onnx) mkColors() map[string]color.Color {
	onnxGraph := o.GetGraph()
	onnxNodes := onnxGraph.GetNode()

	s := set.New(set.ThreadSafe)
	for _, nd := range onnxNodes {
		s.Add(nd.GetOpType())
	}
	colors, err := gamut.Generate(s.Size(), gamut.PastelGenerator{})
	if err != nil {
		return map[string]color.Color{}
	}
	ndColors := map[string]color.Color{}
	idx := 0
	for _, nd := range onnxNodes {
		if _, ok := ndColors[nd.GetOpType()]; ok {
			continue
		}
		ndColors[nd.GetOpType()] = colors[idx]
		idx++
	}

	return ndColors
}

func (o *Onnx) ToGraph(oo ...GraphOption) *Graph {
	opts := NewGraphOptions(oo...)
	onnxGraph := o.GetGraph()
	graphIds := map[string]int64{}

	grph := simple.NewWeightedDirectedGraph(0, math.Inf(1))

	colors := o.mkColors()

	isInputNode := func(name string) bool {
		inputs := onnxGraph.GetInput()
		if !opts.InputsAsConstantNodes {
			inputs = inputs[1:]
		}
		if len(inputs) == 0 {
			return false
		}
		for _, input := range inputs {
			if input.GetName() == name {
				return true
			}
		}
		return false
	}

	isOutputNode := func(name string) bool {
		outputs := onnxGraph.GetOutput()
		if len(outputs) <= 1 {
			return false
		}
		for _, output := range outputs {
			if output.GetName() == name {
				return true
			}
		}
		return false
	}
	_ = isOutputNode

	mkOnnxConstantInputNode := func(name string, src *onnx.NodeProto) *onnx.NodeProto {
		return &onnx.NodeProto{
			Input:  []string{},
			Name:   name,
			OpType: "constant_input",
		}
	}

	mkOnnxConstantOutputNode := func(name string, src *onnx.NodeProto) *onnx.NodeProto {
		return &onnx.NodeProto{
			Input:  []string{src.Name},
			Name:   name,
			OpType: "constant_output",
		}
	}

	_ = mkOnnxConstantOutputNode

	skipNode := func(name string) bool {
		if !opts.PruneInputs {
			return false
		}
		return isInputNode(name)
	}

	addNode := func(onnxNode *onnx.NodeProto, name string) {
		if _, ok := graphIds[name]; ok {
			return
		}
		if skipNode(name) {
			return
		}
		id := grph.NewNode().ID()
		if opts.InputsAsConstantNodes && isInputNode(name) {
			onnxNode = mkOnnxConstantInputNode(name, onnxNode)
		} else {
			onnxNode.Attribute = append(onnxNode.Attribute, &onnx.AttributeProto{
				Name: "edge_name",
				S:    []byte(name),
			})
		}
		color, ok := colors[onnxNode.GetOpType()]
		if !ok {
			color = nil
		}
		nd := &GraphNode{
			id:        id,
			name:      name,
			color:     color,
			NodeProto: onnxNode,
		}
		grph.AddNode(nd)
		graphIds[name] = nd.ID()
	}

	for _, onnxNode := range onnxGraph.Node {
		for _, inputNode := range onnxNode.GetInput() {
			addNode(onnxNode, inputNode)
		}
		for _, outputNode := range onnxNode.GetOutput() {
			addNode(onnxNode, outputNode)
		}
	}

	for _, nd := range onnxGraph.GetNode() {
		for _, inputNode := range nd.GetInput() {
			if skipNode(inputNode) {
				continue
			}
			for _, outputNode := range nd.GetOutput() {
				if skipNode(outputNode) {
					continue
				}
				inId := graphIds[inputNode]
				outId := graphIds[outputNode]
				if inId == outId {
					continue
				}
				inNd := grph.Node(inId)
				outNd := grph.Node(outId)
				grph.SetWeightedEdge(&GraphEdge{
					WeightedEdge: simple.WeightedEdge{
						F: inNd,
						T: outNd,
						W: 1,
					},
				})
			}
		}
	}

	input := onnxGraph.GetInput()[0]
	inId := graphIds[input.GetName()]

	network := &Graph{
		Root:                  grph.Node(inId).(*GraphNode),
		WeightedDirectedGraph: grph,
	}

	o.network = network

	return network
}

func (o Onnx) TopologicallyOrderedNodes(oo ...GraphOption) (GraphNodes, error) {
	g := o.ToGraph(oo...)
	return g.TopologicallyOrderedNodes()
}

func (g Graph) TopologicallyOrderedNodes() (GraphNodes, error) {
	nds, err := topo.SortStabilized(g, sortById)
	if err != nil {
		return nil, errors.Wrap(err, "failed to topologically sort graph")
	}
	res := make([]*GraphNode, len(nds))
	for ii, nd := range nds {
		e, ok := nd.(*GraphNode)
		if !ok {
			panic("expecting a *GraphNode")
		}
		res[ii] = e
	}
	return res, nil
}
