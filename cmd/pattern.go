package cmd

import (
	"context"
	"path/filepath"

	sourcepath "github.com/GeertJohan/go-sourcepath"
	"github.com/Unknwon/com"
	zglob "github.com/mattn/go-zglob"
	"github.com/pkg/errors"
	"github.com/rai-project/dlperf/pkg/onnx"
	"github.com/spf13/cobra"
	"golang.org/x/sync/errgroup"
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

		var models []*onnx.Onnx
		if com.IsFile(modelPath) {
			model, err := onnx.New(modelPath)
			if err != nil {
				return err
			}
			models = []*onnx.Onnx{model}
		}
		if com.IsDir(modelPath) {
			modelPaths, err := zglob.Glob(filepath.Join(modelPath, "**", "*.onnx"))
			if err != nil {
				return err
			}
			models = make([]*onnx.Onnx, len(modelPaths))
			g, _ := errgroup.WithContext(context.Background())
			for ii := range modelPaths {
				idx := ii
				g.Go(func() error {
					path := modelPaths[idx]
					model, err := onnx.New(path)
					if err != nil {
						return err
					}
					models[idx] = model
					return nil
				})
			}
			if err := g.Wait(); err != nil {
				return err
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
