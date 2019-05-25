package cmd

import (
	"github.com/Unknwon/com"
	"github.com/pkg/errors"
	"github.com/rai-project/dlperf/pkg/onnx"
	"github.com/spf13/cobra"
	"gonum.org/v1/gonum/graph/encoding/dot"
)

var subgraphCmd = &cobra.Command{
	Use:     "subgraph",
	Aliases: []string{"subgraphs", "find_subgraphs"},
	Short:   "Finds subgraphs within a model",
	RunE: func(cmd *cobra.Command, args []string) error {

		modelPath = expandModelPath(modelPath)

		if !com.IsFile(modelPath) {
			return errors.Errorf("file %v does not exist", modelPath)
		}

		model, err := onnx.New(modelPath)
		if err != nil {
			return err
		}

		subgrphs, err := model.FindSubGraphs()
		if err != nil {
			return err
		}

		for _, subgrph := range subgrphs {
			dotEnc, err := dot.Marshal(subgrph, model.GetName(), "", "  ")
			if err != nil {
				println(err)
				continue
			}
			img, err := dotToImage(dotEnc)
			if err != nil {
				println(err)
				return err
			}

			println(img)
		}

		return nil
	},
}

func init() {
	subgraphCmd.PersistentFlags().BoolVar(&pruneGraph, "prune", false, "prune graph before finding subgraphs")
	rootCmd.AddCommand(subgraphCmd)
}
