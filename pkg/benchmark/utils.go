package benchmark 


import (
	"reflect"
	"github.com/google/go-cmp/cmp"
	"github.com/k0kubun/pp"
	"github.com/spf13/cast"
)

func isSameScalar(a, b interface{}) bool {
	if cmp.Equal(a, b) {
		return true
	}

	
	switch a := a.(type) {
	case int:
		return isSameInt(a, b)
	case uint:
		return isSameUint(a, b)
	case int8:
		return isSameInt8(a, b)
	case uint8:
		return isSameUint8(a, b)
	case int16:
		return isSameInt16(a, b)
	case uint16:
		return isSameUint16(a, b)
	case int32:
		return isSameInt32(a, b)
	case uint64:
		return isSameUint64(a, b)
	case float32:
		return isSameFloat32(a, b)
	case float64:
		return isSameFloat64(a, b)
	case string:
		if s, err := cast.ToFloat64E(a); err == nil {
		return isSameFloat64(s, b)
		}
	}

    pp.Println(reflect.TypeOf(a).Kind())

	return false
}