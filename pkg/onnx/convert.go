package onnx

import (
	"strings"

	"github.com/k0kubun/pp"
	dlperf "github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/dlperf/pkg/layer"
	"github.com/rai-project/onnx"
)

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
	transAAttr := getNodeAttributeFromName(node, "transA")
	transBAttr := getNodeAttributeFromName(node, "transB")

	return &layer.Gemm{
		Base:   o.mkBase(node),
		TransA: int(transAAttr.GetI()),
		TransB: int(transBAttr.GetI()),
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

	dims := o.GetValueInfoDimensions([]string{node.Name})
	base.SetInputShapes(dims)
	base.SetOutputShapes(dims)

	return &layer.ConstantInput{
		Base: base,
	}
}
