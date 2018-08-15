package cmd

import (
	"path/filepath"

	sourcepath "github.com/GeertJohan/go-sourcepath"
	"github.com/rai-project/dlperf/pkg/onnx"
	"github.com/spf13/cobra"
)

var (
	patternLength int
)

var patternCmd = &cobra.Command{
	Use:     "pattern",
	Aliases: []string{"patterns", "find_pattern"},
	Short:   "Finds pattern chains within models",
	RunE: func(cmd *cobra.Command, args []string) error {
		if modelPath == "" {
			modelPath = filepath.Join(sourcepath.MustAbsoluteDir(), "..", "assets", "onnx_models", "mnist.onnx")
		} else {
			s, err := filepath.Abs(modelPath)
			if err == nil {
				modelPath = s
			}
		}

		models, err := readModels(modelPath)
		if err != nil {
			return err
		}

		subseqs, err := onnx.NodeSubsequences(
			models,
			onnx.PatternPruneGraph(pruneGraph),
			onnx.PatternLength(patternLength),
		)
		if err != nil {
			return err
		}

		patterns := subseqs.Counts()

		writer := NewWriter(pattern{}, humanFlops)
		defer writer.Close()

		for _, pat := range patterns {
			writer.Row(pattern{pat})
		}

		return nil
	},
}

func init() {
	patternCmd.PersistentFlags().IntVar(&patternLength, "length", 2, "length of the pattern sequence")
	patternCmd.PersistentFlags().BoolVar(&pruneGraph, "prune", false, "prune graph before finding patterns")
	rootCmd.AddCommand(patternCmd)
}
