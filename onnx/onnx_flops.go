package onnx

import (
	"strings"

	"github.com/k0kubun/pp"
	"github.com/rai-project/dlperf"
	"github.com/rai-project/dlperf/layer"
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
	inputs := o.Graph.GetInput()
	inputDims := inputs[0].GetType().GetTensorType().GetShape().GetDim()

	if len(inputs) == 0 || inputs[0] == nil || len(inputDims) == 0 {
		log.Info("no input info for graph", o.Graph.GetName())
		return nil
	}

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
	case "constant":
		layer = mkConstant(node)
	case "convolution":
		layer = mkConv(node)
	case "relu":
		layer = mkReLU(node)
	case "dropout":
		layer = mkDropout(node)
	case "innerproduct", "inner_product":
		layer = mkInnerProduct(node)
	case "pooling":
		layer = mkPooling(node)
	case "batchnorm", "bn":
		layer = mkBatchNorm(node)
	case "lrn":
		layer = mkLRN(node)
	case "normalize":
	case "concat":
		parentsInfo := c.getParentsInfo(node)
		layer = mkConcat(parentsInfo, node)
	case "eltwise":
		parentsInfo := c.getParentsInfo(node)
		layer = mkElementWise(parentsInfo, node)
	case "softmax", "softmaxwithloss", "softmax_loss":
		layer = mkSoftMax(node)
	case "flatten":
		pp.Println("unhandeled", operatorType)
	case "power":
		pp.Println("unhandeled", operatorType)
	case "deconvolution":
		pp.Println("unhandeled", operatorType)
	case "crop":
		pp.Println("unhandeled", operatorType)
	case "scale":
		layer = mkScale(node)
	case "implicit":
		pp.Println("unhandeled", operatorType)
	case "accuracy":
		pp.Println("unhandeled", operatorType)
	case "permute":
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

func mkConstant(node *onnx.NodeProto) dlperf.Layer {
	inputDimensions := dlperf.ToInt64Slice(param.Shape[0].Dim)
	return &layer.Input{
		N: inputDimensions[0],
		C: inputDimensions[1],
		W: inputDimensions[2],
		H: inputDimensions[3],
	}
}
