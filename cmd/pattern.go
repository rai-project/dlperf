package cmd

import (
	"path/filepath"

	sourcepath "github.com/GeertJohan/go-sourcepath"
	"github.com/Unknwon/com"
	zglob "github.com/mattn/go-zglob"
	"github.com/pkg/errors"
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

		if !com.IsFile(modelPath) && !com.IsDir(modelPath) {
			return errors.Errorf("file %v does not exist", modelPath)
		}

		models := []*onnx.Onnx{}
		if com.IsFile(modelPath) {
			model, err := onnx.New(modelPath)
			if err != nil {
				return err
			}
			models = append(models, model)
		}
		if com.IsDir(modelPath) {
			modelPaths, err := zglob.Glob(filepath.Join(modelPath, "**", "*.onnx"))
			if err != nil {
				return err
			}
			for _, path := range modelPaths {
				model, err := onnx.New(path)
				if err != nil {
					return err
				}
				models = append(models, model)
			}
		}

		subseqs, err := onnx.NodeSubsequences(patternLength, models...)
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
	rootCmd.AddCommand(patternCmd)
}
