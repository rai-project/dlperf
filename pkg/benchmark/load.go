package benchmark

import (
	"encoding/json"
	"io/ioutil"
	"path/filepath"

	"github.com/Unknwon/com"
	zglob "github.com/mattn/go-zglob"
	"github.com/pkg/errors"
)

func readDir(path string) (Suite, error) {
	benchmarkFiles, err := zglob.Glob(filepath.Join(path, "**", "*.json"))
	if err != nil {
		return Suite{}, err
	}
	var suite *Suite
	for _, file := range benchmarkFiles {
		s, err := Read(file)
		if err != nil {
			continue
		}
		if suite == nil {
			suite = &s
			continue
		}
		suite.Merge(s)
	}
	return *suite, nil
}

func Read(path string) (Suite, error) {
	if com.IsDir(path) {
		return readDir(path)
	}
	if !com.IsFile(path) {
		return Suite{}, errors.Errorf("benchmark file %s not found", path)
	}
	bts, err := ioutil.ReadFile(path)
	if err != nil {
		return Suite{}, errors.Wrapf(err, "unable to read benchmark file %s", path)
	}
	var suite Suite
	err = json.Unmarshal(bts, &suite)
	if err != nil {
		return Suite{}, errors.Wrapf(err, "unable to parse benchmark file %s", path)
	}
	return suite, nil
}
