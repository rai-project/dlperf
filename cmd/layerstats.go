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
	"gonum.org/v1/gonum/graph/encoding/dot"
)

func runLayerStats(cmd *cobra.Command, args []string) error {

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
			runLayerStats(cmd, args)
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

	infos, err := net.Analyze()
	if err != nil {
		return err
	}

	if outputFormat == "dot" {
		dotEnc, err := dot.Marshal(net.Network(), "", "", "  ", true)
		if err != nil {
			return err
		}

		img, err := dotToImage(dotEnc)
		if err != nil {
			return err
		}

		println(img)

		return nil
	}

	writer := NewWriter(stat{}, humanFlops)
	defer writer.Close()

	for _, info := range infos {
		writer.Row(
			stat{
				Name:             info.Name(),
				Type:             info.OperatorType(),
				InputNames:       info.InputNames(),
				OutputNames:      info.OutputNames(),
				ShapeInformation: info.Shape(),
			},
		)
	}

	return nil
}

// layerstatsCmd represents the layerstats command
var layerstatsCmd = &cobra.Command{
	Use:     "layerstats",
	Aliases: []string{"stats"},
	RunE:    runLayerStats,
}

func init() {
	rootCmd.AddCommand(layerstatsCmd)
}
