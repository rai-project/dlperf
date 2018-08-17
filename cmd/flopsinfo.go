// Copyright © 2018 NAME HERE <EMAIL ADDRESS>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

func runFlopsCmd(cmd *cobra.Command, args []string) error {

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

	net, err := onnx.New(modelPath)
	if err != nil {
		return err
	}

	if fullFlops {
		infos, err := net.Analyze()
		if err != nil {
			return err
		}

		writer := NewWriter(layer{}, humanFlops)
		defer writer.Close()

		for _, info := range infos {
			writer.Row(
				layer{
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

	writer := NewWriter(netSummary{}, humanFlops)
	defer writer.Close()

	writer.Row(netSummary{Name: "MultipleAdds", Value: info.MultiplyAdds})
	writer.Row(netSummary{Name: "Additions", Value: info.Additions})
	writer.Row(netSummary{Name: "Divisions", Value: info.Divisions})
	writer.Row(netSummary{Name: "Exponentiations", Value: info.Exponentiations})
	writer.Row(netSummary{Name: "Comparisons", Value: info.Comparisons})
	writer.Row(netSummary{Name: "General", Value: info.General})
	writer.Row(netSummary{Name: "Total", Value: info.Total()})

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
