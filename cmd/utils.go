package cmd

import (
	"context"
	"fmt"
	"path"
	"path/filepath"
	"strings"

	"github.com/Unknwon/com"
	zglob "github.com/mattn/go-zglob"
	"github.com/pkg/errors"
	"github.com/rai-project/dlperf/pkg/onnx"
	"github.com/rai-project/utils"
	"golang.org/x/sync/errgroup"
)

func readModels(modelPath string) ([]*onnx.Onnx, error) {

	if !com.IsFile(modelPath) && !com.IsDir(modelPath) {
		return nil, errors.Errorf("file %v does not exist", modelPath)
	}

	if com.IsFile(modelPath) {
		model, err := onnx.New(modelPath)
		if err != nil {
			return nil, err
		}
		return []*onnx.Onnx{model}, nil
	}

	// is a directory

	modelPaths, err := zglob.Glob(filepath.Join(modelPath, "**", "*.onnx"))
	if err != nil {
		return nil, err
	}
	models := make([]*onnx.Onnx, len(modelPaths))
	g, _ := errgroup.WithContext(context.Background())
	for ii := range modelPaths {
		idx := ii
		g.Go(func() error {
			path := modelPaths[idx]
			model, err := onnx.New(path)
			if err != nil {
				return err
			}
			models[idx] = model
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return nil, err
	}
	return models, nil

}

func getSrcPath(importPath string) (appPath string) {
	paths := com.GetGOPATHs()
	for _, p := range paths {
		d := filepath.Join(p, "src", importPath)
		if com.IsExist(d) {
			appPath = d
			break
		}
	}

	if len(appPath) == 0 {
		appPath = filepath.Join(goPath, "src", importPath)
	}

	return appPath
}

func flopsToString(e int64, humanFlops bool) string {
	if humanFlops {
		return utils.Flops(uint64(e))
	}
	return fmt.Sprintf("%v", e)
}

func getModelName(modelPath string) string {
	return iGetModelName(modelPath, "")
}

func iGetModelName(modelPath, suffix string) string {
	name := strings.TrimSuffix(filepath.Base(modelPath), ".onnx")
	if name != "model" && name != "model_inferred" {
		return name + suffix
	}
	if suffix == "" && name == "model_inferred" {
		suffix = "_inferred"
	}
	return iGetModelName(path.Dir(modelPath), suffix)
}
