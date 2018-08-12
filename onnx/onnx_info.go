package onnx

import (
	"strings"

	"github.com/k0kubun/pp"
	"github.com/rai-project/dlperf"
	"github.com/rai-project/dlperf/layer"
	"github.com/rai-project/onnx"
)

func (o Onnx) Information() (dlperf.FlopsInformation, dlperf.MemoryInformation) {
	modelinfo := o.ModelInformation()
	flops := dlperf.FlopsInformation{}
	memory := dlperf.MemoryInformation{}
	for _, info := range modelinfo {
		flops = flops.Add(info.Flops())
		memory = memory.Add(info.Memory())
	}

	return flops, memory
}

func (o Onnx) FlopsInformation() dlperf.FlopsInformation {
	flops, _ := o.Information()
	return flops
}

func (o Onnx) MemoryInformation() dlperf.MemoryInformation {
	_, memory := o.Information()
	return memory
}

func (o Onnx) ModelInformation() []dlperf.LayerInformation {
	ret := []dlperf.LayerInformation{}

	// the nodes in the graph are sorted topologically
	iter := o.nodes.IterFunc()
	for kv, ok := iter(); ok; kv, ok = iter() {
		node := kv.Value.(*onnx.NodeProto)
		name := node.GetName()
		layer := o.mkLayer(node)

		// pp.Println("layer ", name)

		if layer == nil {
			continue
		}

		info := layer.Information()
		ret = append(ret, info)
	}

	return ret
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
	layerType := strings.ToLower(node.GetOpType())

	switch layerType {
	case "batchnorm":
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
	case "relu":
		ret = o.mkReLU(node)
	case "reshape":
		ret = o.mkReshape(node)
	case "scale":
		ret = o.mkScale(node)
	case "softmax":
		ret = o.mkSoftMax(node)
	case "constant":
		ret = o.mkConstant(node)
	default:
		pp.Println("unhandeled", layerType)
	}

	if ret != nil {
		ret.SetName(node.Name)
	}

	return ret
}

func (o Onnx) mkBatchNorm(node *onnx.NodeProto) dlperf.Layer {
	return &layer.BatchNorm{
		inputs:            node.GetInput(),
		outputs:           node.GetOutput(),
		InputsDimensions:  o.GetValueInfoDimensions(node.GetInput()),
		OutputsDimensions: o.GetValueInfoDimensions(node.GetOutput()),
	}
}

func (o Onnx) mkConcat(node *onnx.NodeProto) dlperf.Layer {
	return &layer.Concat{
		InputsDimensions:  o.GetValueInfoDimensions(node.GetInput()),
		OutputsDimensions: o.GetValueInfoDimensions(node.GetOutput()),
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
		AutoPad:           autoPad,
		Dilations:         dilations,
		Group:             group,
		KernelShape:       kernelShapeAttr.GetInts(),
		Pads:              pads,
		Strides:           strides,
		InputsDimensions:  o.GetValueInfoDimensions(node.GetInput()),
		OutputsDimensions: o.GetValueInfoDimensions(node.GetOutput()),
	}
}

func (o Onnx) mkDropout(node *onnx.NodeProto) dlperf.Layer {
	return &layer.Dropout{
		InputsDimensions:  o.GetValueInfoDimensions(node.GetInput()),
		OutputsDimensions: o.GetValueInfoDimensions(node.GetOutput()),
	}
}

func (o Onnx) mkElementWise(node *onnx.NodeProto) dlperf.Layer {

	return &layer.ElementWise{
		Operator:          node.GetOpType(),
		InputsDimensions:  o.GetValueInfoDimensions(node.GetInput()),
		OutputsDimensions: o.GetValueInfoDimensions(node.GetOutput()),
	}
}

func (o Onnx) mkGemm(node *onnx.NodeProto) dlperf.Layer {
	return &layer.Gemm{
		InputsDimensions:  o.GetValueInfoDimensions(node.GetInput()),
		OutputsDimensions: o.GetValueInfoDimensions(node.GetOutput()),
	}
}

func (o Onnx) mkMatMul(node *onnx.NodeProto) dlperf.Layer {
	return &layer.MatMul{
		InputsDimensions:  o.GetValueInfoDimensions(node.GetInput()),
		OutputsDimensions: o.GetValueInfoDimensions(node.GetOutput()),
	}
}

func (o Onnx) mkPooling(node *onnx.NodeProto) dlperf.Layer {
	kernelShapeAttr := getNodeAttributeFromName(node, "kernel_shape")

	return &layer.Pooling{
		Operator:          node.GetOpType(),
		KernelShape:       kernelShapeAttr.GetInts(),
		InputsDimensions:  o.GetValueInfoDimensions(node.GetInput()),
		OutputsDimensions: o.GetValueInfoDimensions(node.GetOutput()),
	}
}

func (o Onnx) mkLRN(node *onnx.NodeProto) dlperf.Layer {
	size := int64(1)
	sizeAttr := getNodeAttributeFromName(node, "size")
	if sizeAttr.GetInts() != nil {
		size = sizeAttr.GetInts()[0]
	}

	return &layer.LRN{
		Size:              size,
		InputsDimensions:  o.GetValueInfoDimensions(node.GetInput()),
		OutputsDimensions: o.GetValueInfoDimensions(node.GetOutput()),
	}
}

func (o Onnx) mkReLU(node *onnx.NodeProto) dlperf.Layer {
	return &layer.ReLU{
		InputsDimensions:  o.GetValueInfoDimensions(node.GetInput()),
		OutputsDimensions: o.GetValueInfoDimensions(node.GetOutput()),
	}
}

func (o Onnx) mkReshape(node *onnx.NodeProto) dlperf.Layer {
	return &layer.Reshape{
		InputsDimensions:  o.GetValueInfoDimensions(node.GetInput()),
		OutputsDimensions: o.GetValueInfoDimensions(node.GetOutput()),
	}
}

func (o Onnx) mkScale(node *onnx.NodeProto) dlperf.Layer {
	return &layer.Scale{
		InputsDimensions:  o.GetValueInfoDimensions(node.GetInput()),
		OutputsDimensions: o.GetValueInfoDimensions(node.GetOutput()),
	}
}

func (o Onnx) mkSoftMax(node *onnx.NodeProto) dlperf.Layer {
	return &layer.SoftMax{
		InputsDimensions:  o.GetValueInfoDimensions(node.GetInput()),
		OutputsDimensions: o.GetValueInfoDimensions(node.GetOutput()),
	}
}

func (o Onnx) mkConstant(node *onnx.NodeProto) dlperf.Layer {
	return &layer.Constant{
		InputsDimensions:  o.GetValueInfoDimensions(node.GetInput()),
		OutputsDimensions: o.GetValueInfoDimensions(node.GetOutput()),
	}
}
