package benchmark

import (
	"encoding/json"
	"encoding/xml"
	"io/ioutil"
	"path/filepath"
	"strings"

	"github.com/Unknwon/com"
	zglob "github.com/mattn/go-zglob"
	"github.com/pkg/errors"
	nvidiasmi "github.com/rai-project/nvidia-smi"
)

func readDir(path string) (Suite, error) {
	benchmarkFiles, err := zglob.Glob(filepath.Join(path, "**", "*.json"))
	if err != nil {
		return Suite{}, err
	}
	var suite *Suite
	for _, file := range benchmarkFiles {
		s, err := New(file)
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

func New(path string) (Suite, error) {
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
  // replace inf with 0
	btss := strings.ReplaceAll(string(bts), "inf,", "0,")
	var suite Suite
	err = json.Unmarshal([]byte(btss), &suite)
	if err != nil {
		return Suite{}, errors.Wrapf(err, "unable to parse benchmark file %s", path)
	}
	resBenchs := []Benchmark{}
	for _, b := range suite.Benchmarks {
		if b.Name == "" {
			continue
		}
		resBenchs = append(resBenchs, b)
	}
	var res Suite
	res.Context = suite.Context
	res.Benchmarks = resBenchs
	if com.IsFile(strings.TrimSuffix(path, ".json") + ".machine") {
		info := new(nvidiasmi.NvidiaSmi)
		xmlFilePath := strings.TrimSuffix(path, ".json") + ".machine"
		if bts, err := ioutil.ReadFile(xmlFilePath); err == nil {
			err := xml.Unmarshal(bts, info)
			if err == nil {
				res.GPUInformation = info
			}
		}
	}
	return res, nil
}
