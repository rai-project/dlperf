package onnx

import (
	"strings"

	"github.com/k0kubun/pp"
	"github.com/rai-project/dlperf"
	"github.com/rai-project/onnx"
)

func (o Onnx) Information() (dlperf.FlopsInformation, dlperf.MemoryInformation) {
	infos := o.LayerInformations()
	flops := dlperf.FlopsInformation{}
	memory := dlperf.MemoryInformation{}
	for _, info := range infos {
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
	_, mem := o.Information()
	return mem
}

func getDimValues(dims []*onnx.TensorShapeProto_Dimension) []int64 {
	ret := []int64{}
	for _, dim := range dims {
		ret = append(ret, dim.GetDimValue())
	}

	return ret
}

func (o Onnx) GetNodeInputs(node *onnx.NodeProto) []*onnx.ValueInfoProto {
	ret := []*onnx.ValueInfoProto{}
	for _, name := range node.GetInput() {
		val, ok := o.values[name]
		if ok {
			ret = append(ret, val)
		}
	}

	return ret
}

func (o Onnx) LayerInformations() []dlperf.LayerInfo {
	// use the first input
	// inputs := o.Graph.GetInput()
	// inputDims := inputs[0].GetType().GetTensorType().GetShape().GetDim()

	// if len(inputs) == 0 || inputs[0] == nil || len(inputDims) == 0 {
	// 	log.Info("no input info for graph", o.Graph.GetName())
	// 	return nil
	// }

	ret := []dlperf.LayerInfo{}

	// the nodes in the graph are sorted topologically
	for _, node := range o.Graph.GetNode() {
		name := node.GetName()
		layer := o.mkLayer(node)

		if layer == nil {
			pp.Println("failed to create ", name)
			return nil
		}

		info := layer.LayerInformation()
		o.layerInformation[name] = info
		ret = append(ret, info)
	}

	return ret
}

func (o Onnx) mkLayer(node *onnx.NodeProto) dlperf.Layer {
	var layer dlperf.Layer
	operatorType := strings.ToLower(node.GetOpType())

	switch operatorType {
	case "conv":
		layer = &layer.Convolution{Node: node}
	// case "reshape":
	// 	layer = mkReshape(node)
	// case "relu":
	// 	layer = mkReLU(node)
	// case "dropout":
	// 	layer = mkDropout(node)
	// case "add", "mul":
	// 	layer = mkElementWise(node)
	// case "maxpool":
	// 	layer = mkMaxPool(node)
	// case "averagepool":
	// 	layer = mkAveragePool(node)
	// case "batchnorm":
	// 	layer = mkBatchNorm(node)
	// case "lrn":
	// 	layer = mkLRN(node)
	// case "concat":
	// 	parentsInfo := c.getParentsInfo(node)
	// 	layer = mkConcat(parentsInfo, node)
	// case "softmax":
	// 	layer = mkSoftMax(node)
	// case "gemm":
	// 	layer = mkGemm(node)
	case "globalaveragepool":
		pp.Println("unhandeled", operatorType)
	default:
		pp.Println("unhandeled", operatorType)
	}

	if layer == nil {
		pp.Println(node)
		return nil
	}

	layer.SetName(node.Name)

	return layer
}

// func mkReLU(node *onnx.NodeProto) dllayer.Layer {
// 	return &layer.ReLU{}
// }

// func mkDropout(node *onnx.NodeProto) dllayer.Layer {
// 	return &layer.Dropout{}
// }

// func mkSoftMax(node *onnx.NodeProto) dllayer.Layer {
// 	return &layer.SoftMax{}
// }

// func mkBatchNorm(node *onnx.NodeProto) dllayer.Layer {
// 	return &layer.BatchNorm{}
// }

// func mkLRN(param *caffe.LRNParameter) dllayer.Layer {
// 	size := uint32(1)
// 	if param != nil && param.LocalSize != nil && *param.LocalSize != 0 {
// 		size = *param.LocalSize
// 	}
// 	region := "ACROSS_CHANNELS"
// 	if param != nil && param.NormRegion != nil && *param.NormRegion != 0 {
// 		region = "WITHIN_CHANNEL"
// 	}

// 	return &layer.LRN{
// 		Region: region,
// 		Size:   size,
// 	}
// }

// func mkPooling(param *caffe.PoolingParameter) dllayer.Layer {
// 	operator := "max"
// 	if param.Pool != nil && *param.Pool != 0 {
// 		operator = param.Pool.String()
// 	}
// 	global := false
// 	if param.GlobalPooling != nil {
// 		global = *param.GlobalPooling
// 	}
// 	return &layer.Pooling{
// 		Operator: strings.ToUpper(operator),
// 		PadH:     padSizeOfPtr(zeroIfNilUint32(param.PadH), param.Pad),
// 		PadW:     padSizeOfPtr(zeroIfNilUint32(param.PadW), param.Pad),
// 		KernelH:  kernelSizeOfPtr(param.KernelH, &param.KernelSize),
// 		KernelW:  kernelSizeOfPtr(param.KernelW, &param.KernelSize),
// 		StrideH:  zeroToOne(strideSizeOfPtr(param.StrideH, param.Stride)),
// 		StrideW:  zeroToOne(strideSizeOfPtr(param.StrideW, param.Stride)),
// 		Global:   global,
// 	}
// }

// func mkInnerProduct(param *caffe.InnerProductParameter) dllayer.Layer {
// 	return &layer.InnerProduct{
// 		NumOutput: param.NumOutput,
// 	}
// }

// func mkConcat(parentsInfo []dllayer.LayerInfo, param *caffe.ConcatParameter) dllayer.Layer {
// 	return &layer.Concat{
// 		ParentsInformation: parentsInfo,
// 	}
// }

// func mkElementWise(parentsInfo []dllayer.LayerInfo, param *caffe.EltwiseParameter) dllayer.Layer {
// 	op := "SUM"
// 	if param != nil && param.Operation != nil && param.Operation.String() != "" {
// 		op = param.Operation.String()
// 	}
// 	return &layer.ElementWise{
// 		Operation:          op,
// 		ParentsInformation: parentsInfo,
// 	}
// }
