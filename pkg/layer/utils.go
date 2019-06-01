package layer

import (
	"errors"
	"reflect"
	"sort"
	"strings"

	"github.com/fatih/structs"
	"github.com/getlantern/deepcopy"
	"github.com/k0kubun/pp"

	dlperf "github.com/rai-project/dlperf/pkg"
)

// Int64Slice attaches the methods of Interface to []int64, sorting in increasing order.
type Int64Slice []int64

func (p Int64Slice) Len() int           { return len(p) }
func (p Int64Slice) Less(i, j int) bool { return p[i] < p[j] }
func (p Int64Slice) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

// Sort is a convenience method.
func (p Int64Slice) Sort() { sort.Sort(p) }
func (p Int64Slice) Reverse() Int64Slice {
	cpy := []int64{}
	deepcopy.Copy(&cpy, p)
	return sort.Reverse(Int64Slice(p)).(Int64Slice)
}

func getOrMinus1(arry []int64, idx int) int64 {
	if len(arry) <= idx {
		return int64(-1)
	}
	return arry[idx]
}

func checkNumber(val interface{}, expected []int, layer string, name string) error {

	var valLen int
	v := reflect.ValueOf(val)

	switch v.Kind() {
	case reflect.Slice:
		valLen = v.Len()
	default:
		return errors.New("checkNumber takes a slice")
	}

	if valLen == 0 {
		log.WithField("layer", layer).WithField("name", name).Error("has zero input")
		return errors.New("checkNumber failed")
	}

	if expected == nil {
		return nil
	}

	for _, e := range expected {
		if valLen == e {
			return nil
		}
	}

	log.WithField("layer", layer).WithField(name, valLen).Errorf("should be in %v", expected)

	return errors.New("checkNumber failed")
}

func isEmpty(object interface{}) bool {

	// get nil case out of the way
	if object == nil {
		return true
	}

	objValue := reflect.ValueOf(object)

	switch objValue.Kind() {
	// collection types are empty when they have no element
	case reflect.Array, reflect.Chan, reflect.Map, reflect.Slice:
		if objValue.Len() == 0 {
			return true
		}
		for ii := 0; ii < objValue.Len(); ii++ {
			elem := objValue.Index(ii).Interface()
			if !isEmpty(elem) {
				return false
			}
		}
		return true
	// pointers are empty if nil or if the value they point to is empty
	case reflect.Ptr:
		if objValue.IsNil() {
			return true
		}
		deref := objValue.Elem().Interface()
		return isEmpty(deref)
		// for all other types, compare against the zero value
	default:
		return false
	}
}

func isAnyEmpty(object interface{}) bool {

	// get nil case out of the way
	if object == nil {
		return true
	}

	objValue := reflect.ValueOf(object)

	switch objValue.Kind() {
	// collection types are empty when they have no element
	case reflect.Array, reflect.Chan, reflect.Map, reflect.Slice:
		if objValue.Len() == 0 {
			return true
		}
		for ii := 0; ii < objValue.Len(); ii++ {
			elem := objValue.Index(ii).Interface()
			if isAnyEmpty(elem) {
				return true
			}
		}
		return false
	// pointers are empty if nil or if the value they point to is empty
	case reflect.Ptr:
		if objValue.IsNil() {
			return true
		}
		deref := objValue.Elem().Interface()
		return isAnyEmpty(deref)
		// for all other types, compare against the zero value
	default:
		return false
	}
}

func getOutputShapes(layers dlperf.Layers) []dlperf.Shape {
	outputShapes := []dlperf.Shape{}
	for _, layer := range layers {
		if len(layer.OutputShapes()) == 0 {
			continue
		}
		if isAnyEmpty(layer.OutputShapes()) {
			log.WithField("layer", layer.Name()).Error(" has empty OutputShapes")
		}
		outputShapes = append(outputShapes, layer.OutputShapes()[0])
	}

	return outputShapes
}

// multidirectionalBroadcastShapeInference
func multidirectionalBroadcastShapeInference(inputShapes []dlperf.Shape) []dlperf.Shape {
	resultShapeSize := 0
	for _, inputShape := range inputShapes {
		pp.Println(inputShape)

		if len(inputShape) > resultShapeSize {
			resultShapeSize = len(inputShape)
		}
	}
	resultShape := dlperf.Shape{}
	for ii := 0; ii < resultShapeSize; ii++ {
		dimValue := int64(1)
		symbolicDim := int64(0)
		numSymbolicDims := int64(0)
		for _, shape := range inputShapes {
			if ii < resultShapeSize-len(shape) {
				continue
			}
			l := ii - resultShapeSize + len(shape)
			dimIJ := int64(0)
			if l < len(shape) {
				dimIJ = shape[l]
			}
			if dimIJ != 0 {
				if dimIJ != 1 {
					if dimValue != dimIJ && dimValue != 1 {
						pp.Println("dimValue =", dimValue, "dimIJ =", dimIJ)
						panic("Incompatible dimensions")
					} else {
						dimValue = dimIJ
					}
				}
			} else {
				if numSymbolicDims == 0 {
					symbolicDim = dimIJ
				}
				numSymbolicDims++
			}
		}
		if dimValue != 0 || numSymbolicDims == 0 {
			resultShape = append(resultShape, int64(dimValue))
		} else if numSymbolicDims == 1 {
			resultShape = append(resultShape, int64(symbolicDim))
		} else {
			resultShape = append(resultShape, 0)
		}
	}

	return []dlperf.Shape{resultShape}
}

func mkFwdBenchmarkFilterName(layer dlperf.Layer, datatype, algorithm string, opts ...dlperf.FwdBenchmarkArgsOptionFunc) string {
	name := "^" + layer.FwdBenchmarkName(opts...) + "_" + strings.ToUpper(datatype) + `(__BatchSize_\d+)?` + `(__\d+)?`
	if algorithm != "" {
		name += `<(.*,\s*)*` + strings.ToUpper(algorithm) + `(\s*,.*)*>`
	}
	return name + ".*"
}

func mkBwdBenchmarkFilterName(layer dlperf.Layer, datatype, algorithm string, opts ...dlperf.BwdBenchmarkArgsOptionFunc) string {
	name := "^" + layer.BwdBenchmarkName(opts...) + "_" + strings.ToUpper(datatype) + `(__BatchSize_\d+)?` + `(__\d+)?`
	if algorithm != "" {
		name += `<(.*,\s*)*` + strings.ToUpper(algorithm) + `(\s*,.*)*>`
	}
	return name + ".*"
}

func benchmarkArgNames(st interface{}) []string {
	tags := []string{}
	for _, field := range structs.New(st).Fields() {
		if field.IsExported() && structs.IsStruct(field.Value()) {
			es := benchmarkArgNames(field.Value())
			for _, v := range es {
				tags = append(tags, v)
			}
			continue
		}
		tag := field.Tag("args")
		if tag == "" || tag == "-" {
			continue
		}
		tags = append(tags, "\""+tag+"\"")
	}
	return tags
}

func benchmarkAttributes(st interface{}) map[string]interface{} {
	attrs := map[string]interface{}{}
	for _, field := range structs.New(st).Fields() {
		if field.IsExported() && structs.IsStruct(field.Value()) {
			es := benchmarkAttributes(field.Value())
			for k, v := range es {
				attrs[k] = v
			}
			continue
		}
		tag := field.Tag("args")
		if tag == "" || tag == "-" {
			continue
		}
		attrs[tag] = field.Value()
	}
	return attrs
}
