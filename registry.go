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
		log.WithField("layers", s).
			Warn("cannot find layers")
		return nil, errors.New("cannot find layers")
	}
	layers, ok := val.(Layer)
	if !ok {
		log.WithField("layers", s).
			Warn("invalid layers")
		return nil, errors.New("invalid layers")
	}
	return layers, nil
}

func Register(s Layer) {
	layers.Store(strings.ToLower(s.Name()), s)
}

func Layers() []string {
	names := []string{}
	layers.Range(func(key, _ interface{}) bool {
		if name, ok := key.(string); ok {
			names = append(names, name)
		}
		return true
	})
	return names
}
