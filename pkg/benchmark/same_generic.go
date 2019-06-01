//go:generate go get github.com/cheekybits/genny
//go:generate genny -in=$GOFILE -out=gen-$GOFILE gen "ElementType=int,uint,uintptr,uint8,uint16,uint32,uint64,int8,int16,int32,int64,float32,float64"

package benchmark

import (
	"reflect"

	"github.com/k0kubun/pp"
	"github.com/spf13/cast"

	"github.com/cheekybits/genny/generic"
)

type ElementType generic.Type

func isSameElementType(self ElementType, other0 interface{}) bool {

	float32Equals := func(a, b float32) bool {
		const EPSILON float32 = 0.0001
		return (a-b) < EPSILON && (b-a) < EPSILON
	}

	float64Equals := func(a, b float64) bool {
		const EPSILON float64 = 0.0001
		return (a-b) < EPSILON && (b-a) < EPSILON
	}

	switch other := other0.(type) {
	case int:
		if i, err := cast.ToIntE(self); err == nil {
			return i == other
		}
		return int(reflect.ValueOf(self).Int()) == other
	case uint:
		return cast.ToUint(self) == other
	case int8:
		return cast.ToInt8(self) == other
	case uint8:
		return cast.ToUint8(self) == other
	case int16:
		return cast.ToInt16(self) == other
	case uint16:
		return cast.ToUint16(self) == other
	case int32:
		return cast.ToInt32(self) == other
	case uint32:
		return cast.ToUint32(self) == other
	case int64:
		return cast.ToInt64(self) == other
	case uint64:
		return cast.ToUint64(self) == other
	case float32:
		return float32Equals(cast.ToFloat32(self), other)
	case float64:
		return float64Equals(cast.ToFloat64(self), other)
	case string:
		return cast.ToString(self) == other
	}
	pp.Println(other0)
	return false
}
