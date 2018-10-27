package onnx

import (
	"strings"

	dlperf "github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/dlperf/pkg/layer"
	"github.com/rai-project/onnx"
)

func (o Onnx) mkLayer(node *onnx.NodeProto) dlperf.Layer {
	var ret dlperf.Layer
	operatorType := strings.ToLower(node.GetOpType())

	switch operatorType {
	case "identity":
		ret = o.mkIdentity(node)
	case "clip":
		ret = o.mkClip(node)
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
		ret = o.mkRelu(node)
	case "reshape":
		ret = o.mkReshape(node)
	case "flatten":
		ret = o.mkFlatten(node)
	case "transpose":
		ret = o.mkTranspose(node)
	case "unsqueeze":
		ret = o.mkUnsqueeze(node)
	case "scale", "imagescaler":
		ret = o.mkScale(node)
	case "softmax":
		ret = o.mkSoftMax(node)
	case "constant":
		ret = o.mkConstant(node)
	case "constant_input":
		ret = o.mkConstantInput(node)
	default:
		panic("unhandeled layer = " + operatorType + " when running the model at " + o.path)
	}

	return ret
}

func (o Onnx) mkBase(node *onnx.NodeProto, operatorTypeName string) *layer.Base {
	inputs := node.GetInput()
	outputs := node.GetOutput()

	base := &layer.Base{}
	base.SetNode(node)
	base.SetName(node.GetName())
	base.SetOnnxOperatorType(node.GetOpType())
	base.SetInputNames(inputs)
	base.SetOutputNames(outputs)
	base.SetInputShapes(o.GetValueInfoDimensions(inputs))
	base.SetOperatorType(operatorTypeName)
	// base.SetOutputShapes(o.GetValueInfoDimensions(outputs))

	return base
}

func (o Onnx) mkBatchNorm(node *onnx.NodeProto) dlperf.Layer {

	spatial := int64(1)
	spatialAttr := getNodeAttributeFromName(node, "spatial")
	spatial = spatialAttr.GetI()

	base := o.mkBase(node, "BatchNorm")
	return &layer.BatchNorm{
		Base:    base,
		Spatial: spatial,
	}
}

func (o Onnx) mkConcat(node *onnx.NodeProto) dlperf.Layer {
	axisAttr := getNodeAttributeFromName(node, "axis")
	return &layer.Concat{
		Base: o.mkBase(node, "Concat"),
		Axis: axisAttr.GetI(),
	}
}

func (o Onnx) mkConv(node *onnx.NodeProto) dlperf.Layer {
	autoPad := "VALID"
	autoPadAttr := getNodeAttributeFromName(node, "auto_pad")
	if autoPadAttr.GetS() != nil {
		autoPad = string(autoPadAttr.GetS())
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
		Base:        o.mkBase(node, "Conv"),
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
		Base: o.mkBase(node, "Dropout"),
	}
}

func (o Onnx) mkElementWise(node *onnx.NodeProto) dlperf.Layer {
	return &layer.ElementWise{
		Base: o.mkBase(node, "ElementWise"),
	}
}

func (o Onnx) mkGemm(node *onnx.NodeProto) dlperf.Layer {
	transAAttr := getNodeAttributeFromName(node, "transA")
	transBAttr := getNodeAttributeFromName(node, "transB")

	return &layer.Gemm{
		Base:   o.mkBase(node, "Gemm"),
		TransA: transAAttr.GetI(),
		TransB: transBAttr.GetI(),
	}
}

func (o Onnx) mkMatMul(node *onnx.NodeProto) dlperf.Layer {

	return &layer.MatMul{
		Base: o.mkBase(node, "MatMul"),
	}
}

func (o Onnx) mkPooling(node *onnx.NodeProto) dlperf.Layer {
	kernelShapeAttr := getNodeAttributeFromName(node, "kernel_shape")
	stridesShapeAttr := getNodeAttributeFromName(node, "strides")

	autoPadAttrVal := getNodeAttributeFromName(node, "auto_pad").GetS()
	autoPadAttr := ""
	if autoPadAttrVal != nil {
		autoPadAttr = string(autoPadAttrVal)
	}
	if autoPadAttr != "" && autoPadAttr != "SAME_UPPER" && autoPadAttr != "SAME_LOWER" && autoPadAttr != "NOTSET" {
		panic("autopad " + autoPadAttr + " is depricated for the maxpooling layer. " +
			"see https://github.com/onnx/onnx/blob/master/docs/Operators.md#maxpool")
	}

	pads := []int64{0, 0, 0, 0}
	padsAttr := getNodeAttributeFromName(node, "pads")
	if padsAttr.GetInts() != nil {
		pads = padsAttr.GetInts()
	}

	return &layer.Pooling{
		Base:        o.mkBase(node, "Pooling"),
		KernelShape: kernelShapeAttr.GetInts(),
		Pads:        pads,
		Strides:     stridesShapeAttr.GetInts(),
	}
}

func (o Onnx) mkGlobalPooling(node *onnx.NodeProto) dlperf.Layer {
	return &layer.GlobalPooling{
		Base: o.mkBase(node, "GlobalPooling"),
	}
}

func (o Onnx) mkLRN(node *onnx.NodeProto) dlperf.Layer {
	size := int64(1)
	sizeAttr := getNodeAttributeFromName(node, "size")
	if sizeAttr.GetInts() != nil {
		size = sizeAttr.GetInts()[0]
	}

	return &layer.LRN{
		Base: o.mkBase(node, "LRN"),
		Size: size,
	}
}

func (o Onnx) mkRelu(node *onnx.NodeProto) dlperf.Layer {
	return &layer.Relu{
		Base: o.mkBase(node, "Relu"),
	}
}

func (o Onnx) mkReshape(node *onnx.NodeProto) dlperf.Layer {
	return &layer.Reshape{
		Base: o.mkBase(node, "Reshape"),
	}
}

func (o Onnx) mkFlatten(node *onnx.NodeProto) dlperf.Layer {
	return &layer.Reshape{
		Base: o.mkBase(node, "Flatten"),
	}
}

func (o Onnx) mkTranspose(node *onnx.NodeProto) dlperf.Layer {
	permAttr := getNodeAttributeFromName(node, "perm")
	perm := permAttr.GetInts()
	return &layer.Transpose{
		Base:        o.mkBase(node, "Transpose"),
		Permutation: perm,
	}
}

func (o Onnx) mkIdentity(node *onnx.NodeProto) dlperf.Layer {
	return &layer.Identity{
		Base: o.mkBase(node, "Identity"),
	}
}

func (o Onnx) mkClip(node *onnx.NodeProto) dlperf.Layer {
	minAttr := getNodeAttributeFromName(node, "min")
	maxAttr := getNodeAttributeFromName(node, "min")
	return &layer.Identity{
		Base: o.mkBase(node, "Clip"),
		Min:  minAttr.GetF(),
		Max:  maxAttr.GetF(),
	}
}

func (o Onnx) mkScale(node *onnx.NodeProto) dlperf.Layer {
	return &layer.Scale{
		Base: o.mkBase(node, "Scale"),
	}
}

func (o Onnx) mkSoftMax(node *onnx.NodeProto) dlperf.Layer {
	return &layer.Softmax{
		Base: o.mkBase(node, "Softmax"),
	}
}

func (o Onnx) mkUnsqueeze(node *onnx.NodeProto) dlperf.Layer {
	axesAttr := getNodeAttributeFromName(node, "axes")
	return &layer.Unsqueeze{
		Base: o.mkBase(node, "Unsqueeze"),
		Axes: axesAttr.GetInts(),
	}
}

func (o Onnx) mkConstant(node *onnx.NodeProto) dlperf.Layer {
	return &layer.Constant{
		Base: o.mkBase(node, "Constant"),
	}
}

func (o Onnx) mkConstantInput(node *onnx.NodeProto) dlperf.Layer {
	base := o.mkBase(node, "ConstantInput")

	dims := o.GetValueInfoDimensions([]string{node.Name})
	base.SetInputShapes(dims)
	base.SetOutputShapes(dims)

	return &layer.ConstantInput{
		Base: base,
	}
}
