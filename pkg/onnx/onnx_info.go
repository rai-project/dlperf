package onnx

import (
	"strings"

	"github.com/cevaris/ordered_map"
	"github.com/k0kubun/pp"
	"github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/dlperf/pkg/layer"
	"github.com/rai-project/onnx"
	"gonum.org/v1/gonum/graph/topo"
)

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

		layer.InferShape(inputLayers...)

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

func (o Onnx) GetValueInfoDimensions(names []string) [][]int64 {
	ret := [][]int64{}
	for _, name := range names {
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
	case "maxpool", "averagepool", "globalmaxpool", "globalaveragepool":
		ret = o.mkPooling(node)
	case "relu", "leakyrelu", "prelu":
		ret = o.mkReLU(node)
	case "reshape", "transpose", "unsqueeze", "identity":
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

	if ret != nil {
		ret.SetName(node.Name)
	}

	return ret
}

func (o Onnx) mkBase(node *onnx.NodeProto) layer.Base {
	inputs := node.GetInput()
	outputs := node.GetOutput()

	base := &layer.Base{}
	base.SetNode(node)
	base.SetName(node.GetName())
	base.SetOperatorType(strings.ToLower(node.GetOpType()))
	base.SetInputs(inputs)
	base.SetOutputs(outputs)
	base.SetInputsDimensions(o.GetValueInfoDimensions(inputs))
	// base.SetOutputsDimensions(o.GetValueInfoDimensions(outputs))

	return *base
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

	pads := []int64{0, 0}
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

	return &layer.Pooling{
		Base:        o.mkBase(node),
		KernelShape: kernelShapeAttr.GetInts(),
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
	val, ok := o.inputs[node.Name]
	if !ok {
		return nil
	}
	base.SetInputsDimensions([][]int64{getValueInfoDimensions(val)})
	base.SetOutputsDimensions([][]int64{getValueInfoDimensions(val)})

	return &layer.ConstantInput{
		Base: base,
	}
}
