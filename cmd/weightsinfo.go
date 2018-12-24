package cmd

import (
	"os"
	"path/filepath"

	sourcepath "github.com/GeertJohan/go-sourcepath"
	"github.com/Unknwon/com"
	"github.com/k0kubun/pp"
	zglob "github.com/mattn/go-zglob"
	"github.com/pkg/errors"
	"github.com/rai-project/dlperf/pkg/onnx"
	"github.com/spf13/cobra"
)

func runWeightsCmd(cmd *cobra.Command, args []string) error {
	if com.IsDir(modelPath) {
		baseOutputFileName := outputFileName
		if !com.IsDir(baseOutputFileName) {
			os.MkdirAll(baseOutputFileName, os.ModePerm)
		}
		modelPaths, err := zglob.Glob(filepath.Join(modelPath, "**", "*.onnx"))
		if err != nil {
			return errors.Wrapf(err, "unable to glob %s", modelPath)
		}
		for _, path := range modelPaths {
			modelPath = path
			modelName := getModelName(modelPath)
			outputFileName = filepath.Join(baseOutputFileName, modelName+"."+outputFormat)
			pp.Println("processing " + modelName + " from " + modelPath + " to " + outputFileName)
			runWeightsCmd(cmd, args)
		}
		return nil
	}

	if modelPath == "" {
		modelPath = filepath.Join(sourcepath.MustAbsoluteDir(), "..", "assets", "onnx_models", "mnist.onnx")
	} else {
		s, err := filepath.Abs(modelPath)
		if err == nil {
			modelPath = s
		}
	}

	if !com.IsFile(modelPath) {
		return errors.Errorf("file %v does not exist", modelPath)
	}

	net, err := onnx.New(modelPath)
	if err != nil {
		return err
	}
	infos, err := net.Information()
	if err != nil {
		return err
	}

	writer := NewWriter(layerWeights{}, humanFlops)
	defer writer.Close()

	for _, info := range infos {
		if info.OperatorType() == "constant_input" || info.OperatorType() == "constant" {
			continue
		}
		writer.Row(
			layerWeights{
				Name:    info.Name(),
				Type:    info.OperatorType(),
				Weigths: info.Weigths(),
			},
		)
	}

	return nil
}

var weigthsinfoCmd = &cobra.Command{
	Use:     "weigthsinfo",
	Aliases: []string{"weights"},
	Short:   "Get weights information about the model",
	RunE:    runWeightsCmd,
}

func init() {
	rootCmd.AddCommand(weigthsinfoCmd)
}
