package layer

import (
	"errors"
	"reflect"
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
