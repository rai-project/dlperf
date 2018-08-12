package cmd

import (
	"fmt"
	"path"
	"path/filepath"
	"strings"

	"github.com/Unknwon/com"
	"github.com/rai-project/utils"
)

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
