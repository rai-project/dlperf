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
	"path/filepath"

	sourcepath "github.com/GeertJohan/go-sourcepath"
	"github.com/Unknwon/com"
	"github.com/pkg/errors"
	"github.com/rai-project/dlperf/onnx"
	"github.com/spf13/cobra"
)

// layerstatsCmd represents the layerstats command
var layerstatsCmd = &cobra.Command{
	Use:   "layerstats",
	Short: "Get layer statistics about the model",
	Long: `A longer description that spans multiple lines and likely contains examples
and usage of using your command. For example:

Cobra is a CLI library for Go that empowers applications.
This application is a tool to generate the needed files
to quickly create a Cobra application.`,
	RunE: func(cmd *cobra.Command, args []string) error {
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

		net, err := onnx.NewOnnx(modelPath)
		if err != nil {
			return err
		}

		infos := net.ModelInformation()

		writer := NewWriter(stat{}, humanFlops)
		defer writer.Close()

		for _, info := range infos {
			writer.Row(
				stat{
					Name:    info.Name(),
					Type:    info.OperatorType(),
					Outputs: info.Outputs(),
				},
			)
		}

		return nil
	},
}

func init() {
	rootCmd.AddCommand(layerstatsCmd)

	// Here you will define your flags and configuration settings.

	// Cobra supports Persistent Flags which will work for this command
	// and all subcommands, e.g.:
	// layerstatsCmd.PersistentFlags().String("foo", "", "A help for foo")

	// Cobra supports local flags which will only run when this command
	// is called directly, e.g.:
	// layerstatsCmd.Flags().BoolP("toggle", "t", false, "Help message for toggle")
}
