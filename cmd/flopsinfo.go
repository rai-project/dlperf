package cmd

import (
	"os"
	"path/filepath"

  "strings"
	sourcepath "github.com/GeertJohan/go-sourcepath"
	"github.com/Unknwon/com"
	"github.com/k0kubun/pp"
	"github.com/pkg/errors"
	"github.com/rai-project/dlperf/pkg/onnx"
	"github.com/rai-project/dlperf/pkg/writer"
	"github.com/spf13/cobra"
)

func runFlopsCmd(cmd *cobra.Command, args []string) error {
	if com.IsDir(modelPath) {
		baseOutputFileName := outputFileName
		if !com.IsDir(baseOutputFileName) {
			os.MkdirAll(baseOutputFileName, os.ModePerm)
		}
		modelPaths, err := getModelsIn(modelPath)
		if err != nil {
			return errors.Wrapf(err, "unable to glob %s", modelPath)
		}
		for _, path := range modelPaths {
			modelPath = path
			modelName := getModelName(modelPath)
			outputFileName = filepath.Join(baseOutputFileName, modelName+"."+outputFormat)
			pp.Println("processing " + modelName + " from " + modelPath + " to " + outputFileName)
			runFlopsCmd(cmd, args)
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

	net, err := onnx.New(modelPath, batchSize)
	if err != nil {
		return err
	}

	if fullInfo {
		infos, err := net.Information()
		if err != nil {
			return err
		}

		writer := NewWriter(
			layerFlops{},
			writer.ShowHumanFlops(humanFlops),
		)
		defer writer.Close()

		for _, info := range infos {
      opType := strings.ToLower(info.OperatorType())
			if opType == "constant_input" || opType == "constantinput" || opType == "constant" {
				continue
			}
			writer.Row(
				layerFlops{
					Name:             info.Name(),
					Type:             info.OperatorType(),
					FlopsInformation: info.Flops(),
					Total:            info.Flops().Total(),
				},
			)
		}

		return nil
	}

	info := net.FlopsInformation()

	writer := NewWriter(
		netFlopsSummary{},
		writer.ShowHumanFlops(humanFlops),
	)
	defer writer.Close()

	writer.Row(netFlopsSummary{Name: "MultipleAdds", Value: info.MultiplyAdds})
	writer.Row(netFlopsSummary{Name: "Additions", Value: info.Additions})
	writer.Row(netFlopsSummary{Name: "Divisions", Value: info.Divisions})
	writer.Row(netFlopsSummary{Name: "Exponentiations", Value: info.Exponentiations})
	writer.Row(netFlopsSummary{Name: "Comparisons", Value: info.Comparisons})
	writer.Row(netFlopsSummary{Name: "General", Value: info.General})
	writer.Row(netFlopsSummary{Name: "Total", Value: info.Total()})

	return nil
}

// flopsinfoCmd represents the flopsinfo command
var flopsinfoCmd = &cobra.Command{
	Use:     "flopsinfo",
	Aliases: []string{"flops"},
	Short:   "Get flops information about the model",
	RunE:    runFlopsCmd,
}

func init() {
	rootCmd.AddCommand(flopsinfoCmd)
}
