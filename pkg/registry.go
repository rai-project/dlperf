package dlperf

import (
	"errors"
	"strings"

	"golang.org/x/sync/syncmap"
)

var layers syncmap.Map

func FromName(s string) (Layer, error) {
	s = strings.ToLower(s)
	val, ok := layers.Load(s)
	if !ok {
		log.WithField("layer", s).
			Warn("cannot find layer")
		return nil, errors.New("cannot find layer")
	}
	layer, ok := val.(Layer)
	if !ok {
		log.WithField("layer", s).
			Warn("invalid layer")
		return nil, errors.New("invalid layer")
	}
	return layer, nil
}

func Register(s Layer) {
	layers.Store(strings.ToLower(s.Name()), s)
}

func RegisteredLayers() []string {
	names := []string{}
	layers.Range(func(key, _ interface{}) bool {
		if name, ok := key.(string); ok {
			names = append(names, name)
		}
		return true
	})
	return names
}
