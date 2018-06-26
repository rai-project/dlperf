package onnx

import (
	"fmt"
	"reflect"

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

func getValueInfoDimensions(valueInfo *onnx.ValueInfoProto) []int64 {
	ret := []int64{}

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
