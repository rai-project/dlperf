package onnx

import (
	"strings"

	"github.com/k0kubun/pp"

	dlperf "github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/dlperf/pkg/layer"
	"github.com/rai-project/onnx"
)

func (o Onnx) mkLayer(node *onnx.NodeProto) []dlperf.Layer {
	var ret []dlperf.Layer
	operatorType := strings.ToLower(node.GetOpType())

	switch operatorType {
	case "identity":
		ret = o.mkIdentity(node)
	case "cast":
		ret = o.mkCast(node)
	case "clip":
		ret = o.mkClip(node)
	case "exp":
		ret = o.mkExp(node)
	case "batchnorm", "batchnormalization":
		ret = o.mkBatchNorm(node)
	case "concat":
		ret = o.mkConcat(node)
	case "conv":
		ret = o.mkConv(node)
	case "nonmaxsuppression":
		ret = o.mkNonMaxSuppression(node)
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
	case "reducemin":
		ret = o.mkReduceMin(node)
	case "topk":
		ret = o.mkTopK(node)
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
	case "squeeze":
		ret = o.mkSqueeze(node)
	case "unsqueeze":
		ret = o.mkUnsqueeze(node)
	case "scale", "imagescaler":
		ret = o.mkScale(node)
	case "softmax":
		ret = o.mkSoftMax(node)
	case "constant":
		ret = o.mkConstant(node)
	case "shape":
		ret = o.mkShape(node)
	case "gather":
		ret = o.mkGather(node)
	case "slice":
		ret = o.mkSlice(node)
	case "constantofshape":
		ret = o.mkConstantOfShape(node)
	case "constant_input", "constantinput":
		ret = o.mkConstantInput(node)
	default:
		panic("unhandeled layer = " + operatorType + " when running the model at " + o.path)
	}

	return ret
}

func (o *Onnx) getTensorProtoByName(name string) *onnx.TensorProto {
	ret, ok := o.initializers[name]
	if ok != true {
		return nil
	}
	return ret
}

func (o Onnx) mkBase(node *onnx.NodeProto, operatorTypeName string) *layer.Base {
	inputs := node.GetInput()
	outputs := node.GetOutput()

	var tensors []*onnx.TensorProto
	for _, input := range inputs {
		t := o.getTensorProtoByName(input)
		if t != nil {
			tensors = append(tensors, t)
		}
	}

	base := &layer.Base{}
	base.SetNode(node)
	base.SetWeightTensors(tensors)
	base.SetName(node.GetName())
	base.SetOnnxOperatorType(node.GetOpType())
	base.SetInputNames(inputs)
	base.SetOutputNames(outputs)
	base.SetInputShapes(o.GetValueInfoDimensions(inputs))
	base.SetOperatorType(operatorTypeName)
	// base.SetOutputShapes(o.GetValueInfoDimensions(outputs))

	return base
}

func (o Onnx) mkBatchNorm(node *onnx.NodeProto) []dlperf.Layer {

	spatial := int64(1)
	spatialAttr := getNodeAttributeFromName(node, "spatial")
	spatial = spatialAttr.GetI()

	base := o.mkBase(node, "BatchNorm")
	return []dlperf.Layer{
		&layer.BatchNorm{
			Base:    base,
			Spatial: spatial,
		},
	}
}

func (o Onnx) mkConcat(node *onnx.NodeProto) []dlperf.Layer {
	axisAttr := getNodeAttributeFromName(node, "axis")
	return []dlperf.Layer{
		&layer.Concat{
			Base: o.mkBase(node, "Concat"),
			Axis: axisAttr.GetI(),
		},
	}
}

func (o Onnx) mkNonMaxSuppression(node *onnx.NodeProto) []dlperf.Layer {
	return []dlperf.Layer{
		&layer.NonMaxSuppression{
			Base: o.mkBase(node, "NonMaxSuppression"),
		},
	}
}

func (o Onnx) mkConv(node *onnx.NodeProto) []dlperf.Layer {
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
	if groupAttr.GetI() != 0 {
		group = groupAttr.GetI()
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

	return []dlperf.Layer{
		&layer.Conv{
			Base:        o.mkBase(node, "Conv"),
			AutoPad:     autoPad,
			Dilations:   dilations,
			Group:       group,
			KernelShape: kernelShapeAttr.GetInts(),
			Pads:        pads,
			Strides:     strides,
		},
	}
}

func (o Onnx) mkDropout(node *onnx.NodeProto) []dlperf.Layer {
	return []dlperf.Layer{
		&layer.Dropout{
			Base: o.mkBase(node, "Dropout"),
		},
	}
}

func (o Onnx) mkElementWise(node *onnx.NodeProto) []dlperf.Layer {
	boardcastAttr := getNodeAttributeFromName(node, "broadcast")
	axisAttr := getNodeAttributeFromName(node, "axis")

	return []dlperf.Layer{
		&layer.ElementWise{
			Base:      o.mkBase(node, "ElementWise"),
			Broadcast: boardcastAttr.GetI(),
			Axis:      axisAttr.GetI(),
		},
	}
}

func (o Onnx) mkGemm(node *onnx.NodeProto) []dlperf.Layer {
	transAAttr := getNodeAttributeFromName(node, "transA")
	transBAttr := getNodeAttributeFromName(node, "transB")

	alphaAttr := getNodeAttributeFromName(node, "alpha")
	alpha := alphaAttr.GetF()
	if alphaAttr == nil {
		alpha = 1.0
	}

	betaAttr := getNodeAttributeFromName(node, "beta")
	beta := betaAttr.GetF()
	if betaAttr == nil {
		beta = 1.0
	}

	return []dlperf.Layer{
		&layer.Gemm{
			Base:   o.mkBase(node, "Gemm"),
			Alpha:  float64(alpha),
			Beta:   float64(beta),
			TransA: transAAttr.GetI(),
			TransB: transBAttr.GetI(),
		},
	}
}

func (o Onnx) mkMatMul(node *onnx.NodeProto) []dlperf.Layer {
	return []dlperf.Layer{
		&layer.Gemm{
			Base:   o.mkBase(node, "Gemm"),
			Alpha:  float64(1.0),
			Beta:   float64(0.0),
			TransA: 0,
			TransB: 0,
		},
	}
}

func (o Onnx) mkReduceMin(node *onnx.NodeProto) []dlperf.Layer {
	axesAttr := getNodeAttributeFromName(node, "axes")
	keepDimsAttr := getNodeAttributeFromName(node, "keepdims")
	return []dlperf.Layer{
		&layer.Reduce{
			Base:     o.mkBase(node, "ReduceMin"),
			Axes:     axesAttr.GetInts(),
			KeepDims: keepDimsAttr.GetI() == 1,
		},
	}
}

func (o Onnx) mkTopK(node *onnx.NodeProto) []dlperf.Layer {
	axis := int64(-1)
	axisAttr := getNodeAttributeFromName(node, "axis")
	if axisAttr.GetI() != 0 {
		axis = axisAttr.GetI()
	} else if axisAttr.GetInts() != nil {
		axis = axisAttr.GetInts()[0]
	}

	// k := o.getTensorProtoByName(node.Input[1])
	pp.Println(node.Input)

	return []dlperf.Layer{
		&layer.TopK{
			Base: o.mkBase(node, "TopK"),
			Axis: axis,
		},
	}
}

func (o Onnx) mkPooling(node *onnx.NodeProto) []dlperf.Layer {
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

	return []dlperf.Layer{
		&layer.Pooling{
			Base:        o.mkBase(node, "Pooling"),
			KernelShape: kernelShapeAttr.GetInts(),
			Pads:        pads,
			Strides:     stridesShapeAttr.GetInts(),
		},
	}
}

func (o Onnx) mkGlobalPooling(node *onnx.NodeProto) []dlperf.Layer {
	return []dlperf.Layer{
		&layer.GlobalPooling{
			Base: o.mkBase(node, "GlobalPooling"),
		},
	}
}

func (o Onnx) mkLRN(node *onnx.NodeProto) []dlperf.Layer {
	size := int64(1)
	sizeAttr := getNodeAttributeFromName(node, "size")
	if sizeAttr.GetInts() != nil {
		size = sizeAttr.GetInts()[0]
	}

	return []dlperf.Layer{
		&layer.LRN{
			Base: o.mkBase(node, "LRN"),
			Size: size,
		},
	}
}

func (o Onnx) mkRelu(node *onnx.NodeProto) []dlperf.Layer {
	return []dlperf.Layer{
		&layer.Relu{
			Base: o.mkBase(node, "Relu"),
		},
	}
}

func (o Onnx) mkReshape(node *onnx.NodeProto) []dlperf.Layer {
	return []dlperf.Layer{
		&layer.Reshape{
			Base: o.mkBase(node, "Reshape"),
		},
	}
}

func (o Onnx) mkFlatten(node *onnx.NodeProto) []dlperf.Layer {
	axis := int64(1)
	axisAttr := getNodeAttributeFromName(node, "size")
	if axisAttr.GetInts() != nil {
		axis = axisAttr.GetInts()[0]
	}
	return []dlperf.Layer{
		&layer.Flatten{
			Base: o.mkBase(node, "Flatten"),
			Axis: axis,
		},
	}
}

func (o Onnx) mkTranspose(node *onnx.NodeProto) []dlperf.Layer {
	permAttr := getNodeAttributeFromName(node, "perm")
	perm := permAttr.GetInts()
	return []dlperf.Layer{
		&layer.Transpose{
			Base:        o.mkBase(node, "Transpose"),
			Permutation: perm,
		},
	}
}

func (o Onnx) mkIdentity(node *onnx.NodeProto) []dlperf.Layer {
	return []dlperf.Layer{
		&layer.Identity{
			Base: o.mkBase(node, "Identity"),
		},
	}
}

func (o Onnx) mkCast(node *onnx.NodeProto) []dlperf.Layer {
	return []dlperf.Layer{
		&layer.Cast{
			Base: o.mkBase(node, "Cast"),
		},
	}
}

func (o Onnx) mkExp(node *onnx.NodeProto) []dlperf.Layer {
	return []dlperf.Layer{
		&layer.Exp{
			Base: o.mkBase(node, "Exp"),
		},
	}
}

func (o Onnx) mkClip(node *onnx.NodeProto) []dlperf.Layer {
	minAttr := getNodeAttributeFromName(node, "min")
	maxAttr := getNodeAttributeFromName(node, "min")
	return []dlperf.Layer{
		&layer.Clip{
			Base: o.mkBase(node, "Clip"),
			Min:  minAttr.GetF(),
			Max:  maxAttr.GetF(),
		},
	}
}

func (o Onnx) mkScale(node *onnx.NodeProto) []dlperf.Layer {
	return []dlperf.Layer{
		&layer.Scale{
			Base: o.mkBase(node, "Scale"),
		},
	}
}

func (o Onnx) mkSoftMax(node *onnx.NodeProto) []dlperf.Layer {
	return []dlperf.Layer{
		&layer.Softmax{
			Base: o.mkBase(node, "Softmax"),
		},
	}
}

func (o Onnx) mkSqueeze(node *onnx.NodeProto) []dlperf.Layer {
	axesAttr := getNodeAttributeFromName(node, "axes")
	return []dlperf.Layer{
		&layer.Squeeze{
			Base: o.mkBase(node, "Squeeze"),
			Axes: axesAttr.GetInts(),
		},
	}
}

func (o Onnx) mkUnsqueeze(node *onnx.NodeProto) []dlperf.Layer {
	axesAttr := getNodeAttributeFromName(node, "axes")
	return []dlperf.Layer{
		&layer.Unsqueeze{
			Base: o.mkBase(node, "Unsqueeze"),
			Axes: axesAttr.GetInts(),
		},
	}
}

func (o Onnx) mkShape(node *onnx.NodeProto) []dlperf.Layer {
	return []dlperf.Layer{
		&layer.Shape{
			Base: o.mkBase(node, "Shape"),
		},
	}
}

func (o Onnx) mkGather(node *onnx.NodeProto) []dlperf.Layer {
	base := o.mkBase(node, "Gather")

	axis := getNodeAttributeFromName(node, "axis").GetI()

	return []dlperf.Layer{
		&layer.Gather{
			Base: base,
			Axis: axis,
		},
	}
}

func (o Onnx) mkSlice(node *onnx.NodeProto) []dlperf.Layer {
	base := o.mkBase(node, "Slice")

	println("todo slice operator")
	return []dlperf.Layer{
		&layer.Gather{
			Base: base,
			// Axis: axis,
		},
	}
}

func (o Onnx) mkConstant(node *onnx.NodeProto) []dlperf.Layer {
	base := o.mkBase(node, "Constant")

	dims := o.GetValueInfoDimensions([]string{node.Name})
	base.SetInputShapes(dims)
	base.SetOutputShapes(dims)

	return []dlperf.Layer{
		&layer.Constant{
			Base: base,
		},
	}
}

func (o Onnx) mkConstantOfShape(node *onnx.NodeProto) []dlperf.Layer {
	base := o.mkBase(node, "ConstantOfShape")

	return []dlperf.Layer{
		&layer.ConstantOfShape{
			Base: base,
		},
	}
}

func (o Onnx) mkConstantInput(node *onnx.NodeProto) []dlperf.Layer {
	base := o.mkBase(node, "ConstantInput")

	dims := o.GetValueInfoDimensions([]string{node.Name})
	base.SetInputShapes(dims)
	base.SetOutputShapes(dims)

	return []dlperf.Layer{
		&layer.ConstantInput{
			Base: base,
		},
	}
}

func dummy() {
	pp.Println("dummy")
}
