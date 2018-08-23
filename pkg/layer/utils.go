package layer

import (
	"errors"
	"reflect"
	"sort"
	"strings"

	"github.com/fatih/structs"
	dlperf "github.com/rai-project/dlperf/pkg"
)

// Int64Slice attaches the methods of Interface to []int64, sorting in increasing order.
type Int64Slice []int64

func (p Int64Slice) Len() int           { return len(p) }
func (p Int64Slice) Less(i, j int) bool { return p[i] < p[j] }
func (p Int64Slice) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

// Sort is a convenience method.
func (p Int64Slice) Sort() { sort.Sort(p) }

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

	log.WithField("layer", layer).WithField(name, valLen).Errorf("should be in", expected)

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
		if isAnyEmpty(layer.OutputShapes()) {
			log.WithField("layer", layer.Name()).Error(" has empty OutputShapes")
		}
		outputShapes = append(outputShapes, layer.OutputShapes()[0])
	}

	return outputShapes
}

func mkBenchmarkFilterName(layer dlperf.Layer, datatype, algorithm string) string {
	name := "^" + layer.FwdBenchmarkName() + "_" + strings.ToUpper(datatype) + `(__\d+)?`
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
