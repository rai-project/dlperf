package cmd

import (
	"bytes"
	"io/ioutil"
	"path/filepath"

	"github.com/rai-project/dlperf/pkg"

	sourcepath "github.com/GeertJohan/go-sourcepath"
	perflayer "github.com/rai-project/dlperf/pkg/layer"
	"github.com/spf13/cobra"
)

var benchgenCmd = &cobra.Command{
	Use:     "benchgen",
	Aliases: []string{"benchmark_generate"},
	Short:   "Generates the benchmark files for layers",
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

		var layers dlperf.Layers
		for _, model := range models {
			model.Information() // no assignment on purpose
			nds, err := model.Network().TopologicallyOrderedNodes()
			if err != nil {
				return err
			}
			for _, nd := range nds {
				layers = append(layers, nd.Layer())
			}
		}

		layers = layers.FwdUnion("float", "")

		prog := bytes.NewBufferString("")

		for _, lyr := range layers {
			if lyr.OperatorType() == "ConstantInput" {
				continue
			}
			if lyr.OperatorType() == "Conv" {
				l := lyr.(*perflayer.Conv)
				prog.WriteString(l.FwdBenchmarkGenerator())
				continue
			}
		}

		if outputFileName == "automatic" || outputFileName == "" {
			println(prog.String())
			return nil
		}

		err = ioutil.WriteFile(outputFileName, prog.Bytes(), 0644)
		if err != nil {
			return err
		}

		return nil
	},
}

func init() {
	rootCmd.AddCommand(benchgenCmd)
}
