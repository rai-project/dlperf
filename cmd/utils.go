package cmd

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/Unknwon/com"
	framework "github.com/rai-project/dlframework/framework/cmd"
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

func forallmodels(run func() error) error {

	if modelName != "all" {
		return run()
	}

	outputDirectory := outputFileName
	if !com.IsDir(outputDirectory) {
		os.MkdirAll(outputDirectory, os.ModePerm)
	}
	for _, model := range framework.DefaultEvaulationModels {
		modelName, modelVersion = framework.ParseModelName(model)
		outputFileName = filepath.Join(outputDirectory, model+"."+outputFileExtension)
		err := run()
		if err != nil {
			log.WithError(err).WithField("modelName", modelName).WithField(modelVersion, modelVersion).Error("failed to get flops information")
		}
	}
	return nil
}

func flopsToString(e int64, humanFlops bool) string {
	if humanFlops {
		return utils.Flops(uint64(e))
	}
	return fmt.Sprintf("%v", e)
}
