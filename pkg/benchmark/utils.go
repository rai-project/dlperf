package benchmark

import (
	"reflect"

	"github.com/google/go-cmp/cmp"
	"github.com/k0kubun/pp"
	"github.com/spf13/cast"
)



// func isSameScalar(a, b interface{}) bool {
// 	if cmp.Equal(a, b) {
// 		return true
// 	}

// 	a0, err := cast.ToFloat64E(a)
// 	if err != nil {
// 		pp.Println(a)
// 		return false
// 	}
// 	// if a0 < 0 {
// 	// 	// panic("a0 < 0")
// 	// 	a0 = float64(0)
// 	// }

// 	b0, err := cast.ToFloat64E(b)
// 	if err != nil {
// 		pp.Println(b)
// 		return false
// 	}
// 	// if b0 < 0 {
// 	// 	// panic("b0 < 0")
// 	// 	b0 = float64(0)
// 	// }

// 	floatEquals := func(a, b float64) bool {
// 		const EPSILON float64 = 0.0001
// 		return (a-b) < EPSILON && (b-a) < EPSILON
// 	}
// 	return floatEquals(a0, b0)
// }

func isSameScalar(a, b interface{}) bool {
	a = indirect(a)
	b = indirect(b)

	if cmp.Equal(a, b) {
		return true
	}

	switch reflect.TypeOf(a).Kind() {
	case reflect.Int:
		return isSameInt(int(reflect.ValueOf(a).Int()), b)
	case reflect.Uint:
		return isSameUint(a.(uint), b)
	case reflect.Int8:
		return isSameInt8(a.(int8), b)
	case reflect.Uint8:
		return isSameUint8(a.(uint8), b)
	case reflect.Int16:
		return isSameInt16(a.(int16), b)
	case reflect.Uint16:
		return isSameUint16(a.(uint16), b)
	case reflect.Int32:
		return isSameInt32(a.(int32), b)
	case reflect.Uint32:
		return isSameUint32(a.(uint32), b)
	case reflect.Int64:
		return isSameInt64(a.(int64), b)
	case reflect.Uint64:
		return isSameUint64(a.(uint64), b)
	case reflect.Float32:
		return isSameFloat32(a.(float32), b)
	case reflect.Float64:
		return isSameFloat64(a.(float64), b)
	case reflect.String:
		if s, err := cast.ToFloat64E(a); err == nil {
			return isSameFloat64(s, b)
		}
	}
	// switch a := a.(type) {
	// case dlperf.ConvBwdType:
	// 	return isSameInt(int(a), b)
	// case int:
	// 	return isSameInt(a, b)
	// case uint:
	// 	return isSameUint(a, b)
	// case int8:
	// 	return isSameInt8(a, b)
	// case uint8:
	// 	return isSameUint8(a, b)
	// case int16:
	// 	return isSameInt16(a, b)
	// case uint16:
	// 	return isSameUint16(a, b)
	// case int32:
	// 	return isSameInt32(a, b)
	// case uint64:
	// 	return isSameUint64(a, b)
	// case int64:
	// 	return isSameInt64(a, b)
	// case float32:
	// 	return isSameFloat32(a, b)
	// case float64:
	// 	return isSameFloat64(a, b)
	// case string:
	// 	if s, err := cast.ToFloat64E(a); err == nil {
	// 		return isSameFloat64(s, b)
	// 	}
	// }

	pp.Println(a)
	pp.Println("int = ", reflect.TypeOf(a).Kind() == reflect.Int)
	pp.Println("int32 = ", reflect.TypeOf(a).Kind() == reflect.Int32)
	pp.Println("int64 = ", reflect.TypeOf(a).Kind() == reflect.Int64)
	pp.Println("float32 = ", reflect.TypeOf(a).Kind() == reflect.Float32)

	return false
}

// From html/template/content.go
// Copyright 2011 The Go Authors. All rights reserved.
// indirect returns the value, after dereferencing as many times
// as necessary to reach the base type (or nil).
func indirect(a interface{}) interface{} {
	if a == nil {
		return nil
	}
	if t := reflect.TypeOf(a); t.Kind() != reflect.Ptr {
		// Avoid creating a reflect.Value if it's not a pointer.
		return a
	}
	v := reflect.ValueOf(a)
	for v.Kind() == reflect.Ptr && !v.IsNil() {
		v = v.Elem()
	}
	return v.Interface()
}
