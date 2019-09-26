package cmd

import (
	"github.com/Unknwon/com"
	"github.com/pkg/errors"
	"github.com/rai-project/dlperf/pkg/onnx"
	"github.com/spf13/cobra"
	"gonum.org/v1/gonum/graph/encoding/dot"
)

var (
	inferShape bool
)

// todotCmd represents the todot command
var todotCmd = &cobra.Command{
	Use:     "todot",
	Aliases: []string{"dot"},
	Short:   "Converts graph to dot format",
	RunE: func(cmd *cobra.Command, args []string) error {

		modelPath = expandModelPath(modelPath)

		if !com.IsFile(modelPath) {
			return errors.Errorf("file %v does not exist", modelPath)
		}

		model, err := onnx.New(modelPath, batchSize)
		if err != nil {
			return err
		}

		var grph *onnx.Graph
		if inferShape {
			model.Information()
			grph = model.Network()
		} else {
			grph = model.ToGraph(onnx.GraphPruneInputs(false), onnx.GraphInputsAsConstantNodes(true))
		}

		if pruneGraph {
			grph = grph.Prune(nil)
		}

		dotEnc, err := dot.Marshal(grph, model.GetName(), "", "  ")
		if err != nil {
			return err
		}

		img, err := dotToImage(dotEnc)
		if err != nil {
			return err
		}

		err = com.WriteFile(outputFileName, dotEnc)
		if err != nil {
			return err
		}

		println(img)

		return nil
	},
}

func init() {
	todotCmd.PersistentFlags().BoolVar(&pruneGraph, "prune", false, "prune graph")
	todotCmd.PersistentFlags().BoolVar(&inferShape, "infer_shape", true, "infer shape")
	rootCmd.AddCommand(todotCmd)
}
