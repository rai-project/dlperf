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
	"github.com/rai-project/onnx"
	"github.com/spf13/cobra"
	"gonum.org/v1/gonum/graph/encoding/dot"
	"gonum.org/v1/gonum/graph/simple"
)

// todotCmd represents the todot command
var todotCmd = &cobra.Command{
	Use:     "todot",
	Aliases: []string{"dot"},
	Short:   "Converts graph to dot format",
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

		model, err := onnx.ReadModelShapeInfer(modelPath)
		if err != nil {
			return err
		}

		onnxGraph := model.GetGraph()
		graphIds := map[string]int64{}

		grph := simple.NewDirectedGraph()
		for _, graphNode := range onnxGraph.Node {
			nd := grph.NewNode()
			grph.AddNode(nd)
			graphIds[graphNode.GetName()] = nd.ID()
		}
		visit := func(graphNode *onnx.NodeProto) {
			inId := graphIds[graphNode.GetName()]
			inNd := grph.Node(inId)
			for _, inputNode := range graphNode.GetInput() {
				in2Id := graphIds[inputNode]
				if inId == in2Id {
					continue
				}
				itNd := grph.Node(in2Id)
				edge := grph.NewEdge(itNd, inNd)
				grph.SetEdge(edge)
			}

			for _, outputNode := range graphNode.GetOutput() {
				outId := graphIds[outputNode]
				if inId == outId {
					continue
				}
				outNd := grph.Node(outId)
				edge := grph.NewEdge(inNd, outNd)
				grph.SetEdge(edge)
			}
		}

		for _, nd := range onnxGraph.GetNode() {
			// pp.Println(nd.GetName())
			visit(nd)
		}

		dotEnc, err := dot.Marshal(grph, onnxGraph.GetName(), "", "  ", true)
		if err != nil {
			return err
		}
		println(string(dotEnc))
		return nil
	},
}

func init() {
	rootCmd.AddCommand(todotCmd)
}
