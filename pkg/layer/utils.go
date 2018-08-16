package layer

import (
	"errors"
	"reflect"

	dlperf "github.com/rai-project/dlperf/pkg"
)

func checkNumber(val interface{}, expected []int, layer string, name string) error {

	var len int
	v := reflect.ValueOf(val)

	switch v.Kind() {
	case reflect.Slice:
		len = v.Len()
	default:
		return errors.New("checkNumber takes a slice")
	}

	for _, e := range expected {
		if len == e {
			return nil
		}
	}

	log.WithField("layer", layer).WithField(name, len).Errorf("should be in", expected)

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

func getOutputShapes(layers []dlperf.Layer) []dlperf.Shape {
	outputShapes := []dlperf.Shape{}
	for _, layer := range layers {
		if isAnyEmpty(layer.OutputShapes()) {
			log.WithField("layer", layer.Name()).Error(" has empty OutputShapes")
		}
		outputShapes = append(outputShapes, layer.OutputShapes()[0])
	}

	return outputShapes
}
