package onnx

import (
	"sort"
	"strings"

	"github.com/cevaris/ordered_map"
	"github.com/k0kubun/pp"
	"github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/dlperf/pkg/layer"
	"github.com/rai-project/onnx"
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/topo"
)

func sortByOrder(nds []graph.Node, layers []dlperf.Layer) []dlperf.Layer {

	if len(layers) == 1 {
		return layers
	}

	findNodePosition := func(name string) int {
		for ii, n0 := range nds {
			n := n0.(GraphNode)
			if n.name == name || n.Name == name {
				return ii
			}
		}
		panic("unable to find node " + name)

		return 0
	}

	order := func(ii, jj int) bool {
		a := layers[ii]
		b := layers[jj]
		apos := findNodePosition(a.Name())
		bpos := findNodePosition(b.Name())
		return apos < bpos
	}

	sort.Slice(layers, order)

	return layers
}

func sortByDimension(layers []dlperf.Layer) []dlperf.Layer {
	if len(layers) == 1 {
		return layers
	}

	rest := layers[1:]

	isGreater := func(ii, jj int) bool {
		a := rest[ii]
		b := rest[jj]
		sa := a.OutputShapes()[0]
		sb := b.OutputShapes()[0]
		if len(sa) > len(sb) {
			return true
		}
		for ii := range sa {
			if sa[ii] > sb[ii] {
				return true
			}
		}
		return false
	}

	sort.Slice(rest, isGreater)

	return append([]dlperf.Layer{layers[0]}, rest...)
}

func (o Onnx) ModelInformation() ([]dlperf.LayerInformation, error) {
	ret := []dlperf.LayerInformation{}

	grph := o.ToGraph(GraphPruneInputs(false), GraphInputsAsConstantNodes(true))
	nds, err := topo.SortStabilized(grph, sortById)
	if err != nil {
		return nil, err
	}

	findNode := func(name string) *GraphNode {
		for _, n0 := range nds {
			n := n0.(GraphNode)
			if n.name == name {
				return &n
			}
		}
		panic("unable to find node " + name)

		return nil
	}

	layers := ordered_map.NewOrderedMap()

	for _, nd0 := range nds {
		nd, ok := nd0.(GraphNode)
		if !ok {
			panic("invalid type for " + pp.Sprint(nd0))
		}
		if nd.name == "OC2_DUMMY_1" {
			// pp.Println(nd.Attribute)
		}
		layer := o.mkLayer(nd.NodeProto)

		if layer == nil {
			continue
		}

		layers.Set(nd.name, layer)
	}

	iter := layers.IterFunc()
	for kv, ok := iter(); ok; kv, ok = iter() {
		layer, ok := kv.Value.(dlperf.Layer)
		if !ok {
			panic("invalid layer type for " + pp.Sprint(kv.Value))
		}

		// if layer.OperatorType() == "ConstantInput" {
		// 	continue
		// }

		nd := findNode(kv.Key.(string))
		inputLayers := []dlperf.Layer{}
		for _, input0 := range grph.To(nd.ID()) {
			input, ok := input0.(GraphNode)
			if !ok {
				panic("invalid type for " + pp.Sprint(input0))
			}

			inputLayer, ok := layers.Get(input.name)
			if !ok {
				panic("unable to find input layer " + pp.Sprint(input))
			}

			inputLayers = append(inputLayers, inputLayer.(dlperf.Layer))
		}

		inputLayers = sortByOrder(nds, inputLayers)
		layer.SetInputs(inputLayers)

		// pp.Println(inputLayers)

		outputLayers := []dlperf.Layer{}
		for _, output0 := range grph.From(nd.ID()) {
			output, ok := output0.(GraphNode)
			if !ok {
				panic("invalid type for " + pp.Sprint(output0))
			}

			outputLayer, ok := layers.Get(output.name)
			if !ok {
				panic("unable to find input layer " + pp.Sprint(output))
			}

			outputLayers = append(outputLayers, outputLayer.(dlperf.Layer))
		}
		outputLayers = sortByOrder(nds, outputLayers)
		layer.SetOutputs(outputLayers)

		// if layer.Name() == "conv_1" {
		// 	pp.Println("Infering Shape on ", layer.Name())
		// 	pp.Println(inputLayers)
		// }

		// defer func() {
		// 	if r := recover(); r != nil {
		// 		// pp.Println(layer.Name())
		// 		panic(r)
		// 	}

		// }()

		layer.SetInputShapes(getOutputShapes(inputLayers))
		layer.InferShape(inputLayers)

		// pp.Println(layer.Name())
		// pp.Println(layer.OutputShapes())

		info := layer.Information()
		ret = append(ret, info)
	}

	// dotEnc, err := dot.Marshal(grph, "", "", "  ", true)
	// if err == nil {
	// 	println(string(dotEnc))
	// }

	return ret, nil
}

func (o Onnx) FlopsInformation() dlperf.FlopsInformation {
	flops := dlperf.FlopsInformation{}
	infos, err := o.ModelInformation()
	if err != nil {
		panic(err)
	}
	for _, info := range infos {
		flops = flops.Add(info.Flops())
	}
	return flops
}

func (o Onnx) MemoryInformation() dlperf.MemoryInformation {
	memory := dlperf.MemoryInformation{}
	infos, err := o.ModelInformation()
	if err != nil {
		panic(err)
	}
	for _, info := range infos {
		memory = memory.Add(info.Memory())
	}
	return memory
}

func (o Onnx) GetValueInfoDimensions(names []string) []dlperf.Shape {
	var ret []dlperf.Shape
	// for k, _ := range o.initializers {
	// 	pp.Println(k)
	// }
	for _, name := range names {
		init, ok := o.initializers[name]
		if ok {
			shp := getTensorProtoDimensions(init)
			ret = append(ret, shp)
			continue
		}

		val, ok := o.valueInfo[name]
		if ok {
			ret = append(ret, getValueInfoDimensions(val))
			continue
		}

		val, ok = o.inputs[name]
		if ok {
			ret = append(ret, getValueInfoDimensions(val))
			continue
		}

		val, ok = o.outputs[name]
		if ok {
			ret = append(ret, getValueInfoDimensions(val))
		}
	}

	return ret
}

func (o Onnx) mkLayer(node *onnx.NodeProto) dlperf.Layer {
	var ret dlperf.Layer
	operatorType := strings.ToLower(node.GetOpType())

	switch operatorType {
	case "batchnorm", "batchnormalization":
		ret = o.mkBatchNorm(node)
	case "concat":
		ret = o.mkConcat(node)
	case "conv":
		ret = o.mkConv(node)
	case "dropout":
		ret = o.mkDropout(node)
	case "add", "sum", "sub", "mul", "div", "max,", "min":
		ret = o.mkElementWise(node)
	case "gemm":
		ret = o.mkGemm(node)
	case "lrn":
		ret = o.mkLRN(node)
	case "matmul":
		ret = o.mkMatMul(node)
	case "maxpool", "averagepool":
		ret = o.mkPooling(node)
	case "globalmaxpool", "globalaveragepool":
		ret = o.mkGlobalPooling(node)
	case "relu", "leakyrelu", "prelu":
		ret = o.mkReLU(node)
	case "reshape", "transpose", "unsqueeze":
		ret = o.mkReshape(node)
	case "scale", "imagescaler":
		ret = o.mkScale(node)
	case "softmax":
		ret = o.mkSoftMax(node)
	case "constant":
		ret = o.mkConstant(node)
	case "constant_input":
		ret = o.mkConstantInput(node)
	default:
		pp.Println("unhandeled", operatorType)
	}

	return ret
}

func (o Onnx) mkBase(node *onnx.NodeProto) *layer.Base {
	inputs := node.GetInput()
	outputs := node.GetOutput()

	base := &layer.Base{}
	base.SetNode(node)
	base.SetName(node.GetName())
	base.SetOperatorType(strings.ToLower(node.GetOpType()))
	base.SetInputNames(inputs)
	base.SetOutputNames(outputs)
	base.SetInputShapes(o.GetValueInfoDimensions(inputs))
	// base.SetOutputShapes(o.GetValueInfoDimensions(outputs))

	return base
}

func (o Onnx) mkBatchNorm(node *onnx.NodeProto) dlperf.Layer {
	return &layer.BatchNorm{
		Base: o.mkBase(node),
	}
}

func (o Onnx) mkConcat(node *onnx.NodeProto) dlperf.Layer {
	return &layer.Concat{
		Base: o.mkBase(node),
	}
}

func (o Onnx) mkConv(node *onnx.NodeProto) dlperf.Layer {
	autoPad := "VALID"
	autoPadAttr := getNodeAttributeFromName(node, "auto_pad")
	if autoPadAttr.GetStrings() != nil {
		autoPad = string(autoPadAttr.GetStrings()[0])
	}

	dilations := []int64{1, 1}
	dilationsAttr := getNodeAttributeFromName(node, "dilations")
	if dilationsAttr.GetInts() != nil {
		dilations = dilationsAttr.GetInts()
	}

	group := int64(1)
	groupAttr := getNodeAttributeFromName(node, "group")
	if groupAttr.GetInts() != nil {
		group = groupAttr.GetInts()[0]
	}

	kernelShapeAttr := getNodeAttributeFromName(node, "kernel_shape")
	if kernelShapeAttr.GetInts() == nil {
		log.WithField("layer", "conv").Error("unknown kernel_shapel, model must be shape inferred")

		return nil
	}

	pads := []int64{0, 0, 0, 0}
	padsAttr := getNodeAttributeFromName(node, "pads")
	if padsAttr.GetInts() != nil {
		pads = padsAttr.GetInts()
	}

	strides := []int64{0, 0}
	stridesAttr := getNodeAttributeFromName(node, "strides")
	if stridesAttr.GetInts() != nil {
		strides = stridesAttr.GetInts()
	}

	return &layer.Conv{
		Base:        o.mkBase(node),
		AutoPad:     autoPad,
		Dilations:   dilations,
		Group:       group,
		KernelShape: kernelShapeAttr.GetInts(),
		Pads:        pads,
		Strides:     strides,
	}
}

func (o Onnx) mkDropout(node *onnx.NodeProto) dlperf.Layer {
	return &layer.Dropout{
		Base: o.mkBase(node),
	}
}

func (o Onnx) mkElementWise(node *onnx.NodeProto) dlperf.Layer {
	return &layer.ElementWise{
		Base: o.mkBase(node),
	}
}

func (o Onnx) mkGemm(node *onnx.NodeProto) dlperf.Layer {
	return &layer.Gemm{
		Base: o.mkBase(node),
	}
}

func (o Onnx) mkMatMul(node *onnx.NodeProto) dlperf.Layer {

	return &layer.MatMul{
		Base: o.mkBase(node),
	}
}

func (o Onnx) mkPooling(node *onnx.NodeProto) dlperf.Layer {
	kernelShapeAttr := getNodeAttributeFromName(node, "kernel_shape")
	padsShapeAttr := getNodeAttributeFromName(node, "pads")
	stridesShapeAttr := getNodeAttributeFromName(node, "strides")

	return &layer.Pooling{
		Base:        o.mkBase(node),
		KernelShape: kernelShapeAttr.GetInts(),
		Pads:        padsShapeAttr.GetInts(),
		Strides:     stridesShapeAttr.GetInts(),
	}
}

func (o Onnx) mkGlobalPooling(node *onnx.NodeProto) dlperf.Layer {
	return &layer.GlobalPooling{
		Base: o.mkBase(node),
	}
}

func (o Onnx) mkLRN(node *onnx.NodeProto) dlperf.Layer {
	size := int64(1)
	sizeAttr := getNodeAttributeFromName(node, "size")
	if sizeAttr.GetInts() != nil {
		size = sizeAttr.GetInts()[0]
	}

	return &layer.LRN{
		Base: o.mkBase(node),
		Size: size,
	}
}

func (o Onnx) mkReLU(node *onnx.NodeProto) dlperf.Layer {
	return &layer.ReLU{
		Base: o.mkBase(node),
	}
}

func (o Onnx) mkReshape(node *onnx.NodeProto) dlperf.Layer {
	return &layer.Reshape{
		Base: o.mkBase(node),
	}
}

func (o Onnx) mkScale(node *onnx.NodeProto) dlperf.Layer {
	return &layer.Scale{
		Base: o.mkBase(node),
	}
}

func (o Onnx) mkSoftMax(node *onnx.NodeProto) dlperf.Layer {
	return &layer.SoftMax{
		Base: o.mkBase(node),
	}
}

func (o Onnx) mkConstant(node *onnx.NodeProto) dlperf.Layer {
	return &layer.Constant{
		Base: o.mkBase(node),
	}
}

func (o Onnx) mkConstantInput(node *onnx.NodeProto) dlperf.Layer {
	base := o.mkBase(node)

	base.SetInputShapes(o.GetValueInfoDimensions([]string{node.Name}))
	base.SetOutputShapes(o.GetValueInfoDimensions([]string{node.Name}))

	return &layer.ConstantInput{
		Base: base,
	}
}
