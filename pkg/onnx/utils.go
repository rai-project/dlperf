package onnx

import (
	"encoding/binary"
	"fmt"
	"reflect"

	dlperf "github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/onnx"
	"github.com/spf13/cast"
)

func getNodeAttributeFromName(node *onnx.NodeProto, attrName string) *onnx.AttributeProto {
	for _, attr := range node.GetAttribute() {
		if attr.GetName() == attrName {
			return attr
		}
	}

	return nil
}

func getTensorProtoDimensions(tensor *onnx.TensorProto) dlperf.Shape {
	var ret dlperf.Shape
	// if tensor.Name == "Parameter1367_reshape1_shape" {
	// 	pp.Println(tensor)
	// }
	if tensor.DataType == onnx.TensorProto_INT32 && len(tensor.GetInt32Data()) > 0 {
		return toInt64Slice(tensor.GetInt32Data())
	}
	if tensor.DataType == onnx.TensorProto_INT64 && len(tensor.GetInt64Data()) > 0 {
		return tensor.GetInt64Data()
	}
	if tensor.DataType == onnx.TensorProto_INT32 && len(tensor.GetRawData()) > 0 {
		dim := tensor.Dims[0]
		rawdata := tensor.GetRawData()
		for ii := int64(0); ii < dim; ii++ {
			val := int64(binary.LittleEndian.Uint32(rawdata[ii*4 : (ii+1)*4]))
			ret = append(ret, val)
		}
		return ret
	}
	if tensor.DataType == onnx.TensorProto_INT64 && len(tensor.GetRawData()) > 0 {
		dim := tensor.Dims[0]
		rawdata := tensor.GetRawData()
		for ii := int64(0); ii < dim; ii++ {
			val := int64(binary.LittleEndian.Uint64(rawdata[ii*8 : (ii+1)*8]))
			ret = append(ret, val)
		}
		return ret
	}
	return tensor.Dims
}

func getValueInfoDimensions(valueInfo *onnx.ValueInfoProto) dlperf.Shape {
	var ret dlperf.Shape
	for _, dim := range valueInfo.GetType().GetTensorType().GetShape().GetDim() {
		ret = append(ret, dim.GetDimValue())
	}
	return ret
}

// toIntSliceE casts an interface to a []int32 type.
func toInt32SliceE(i interface{}) ([]int32, error) {
	if i == nil {
		return []int32{}, fmt.Errorf("unable to cast %#v of type %T to []int32", i, i)
	}

	switch v := i.(type) {
	case []int32:
		return v, nil
	}

	kind := reflect.TypeOf(i).Kind()
	switch kind {
	case reflect.Slice, reflect.Array:
		s := reflect.ValueOf(i)
		a := make([]int32, s.Len())
		for j := 0; j < s.Len(); j++ {
			val, err := cast.ToInt32E(s.Index(j).Interface())
			if err != nil {
				return []int32{}, fmt.Errorf("unable to cast %#v of type %T to []int32", i, i)
			}
			a[j] = val
		}
		return a, nil
	default:
		return []int32{}, fmt.Errorf("unable to cast %#v of type %T to []int32", i, i)
	}
}

func toInt32Slice(i interface{}) []int32 {
	v, _ := toInt32SliceE(i)
	return v
}

// toIntSliceE casts an interface to a []int64 type.
func toInt64SliceE(i interface{}) ([]int64, error) {
	if i == nil {
		return []int64{}, fmt.Errorf("unable to cast %#v of type %T to []int64", i, i)
	}

	switch v := i.(type) {
	case []int64:
		return v, nil
	}

	kind := reflect.TypeOf(i).Kind()
	switch kind {
	case reflect.Slice, reflect.Array:
		s := reflect.ValueOf(i)
		a := make([]int64, s.Len())
		for j := 0; j < s.Len(); j++ {
			val, err := cast.ToInt64E(s.Index(j).Interface())
			if err != nil {
				return []int64{}, fmt.Errorf("unable to cast %#v of type %T to []int64", i, i)
			}
			a[j] = val
		}
		return a, nil
	default:
		return []int64{}, fmt.Errorf("unable to cast %#v of type %T to []int64", i, i)
	}
}

func toInt64Slice(i interface{}) []int64 {
	v, _ := toInt64SliceE(i)
	return v
}

func getOutputShapes(layers []dlperf.Layer) []dlperf.Shape {
	outputShapes := []dlperf.Shape{}
	for _, layer := range layers {
		// pp.Println(layer.Name())
		// pp.Println(layer.OperatorType())
		// pp.Println(layer.InputShapes())
		outputShapes = append(outputShapes, layer.OutputShapes()[0])
	}

	return outputShapes
}
