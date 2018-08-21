package cmd

import (
	"bytes"
	"context"
	"io/ioutil"
	"path/filepath"
	"strings"
	"sync"

	"github.com/k0kubun/pp"
	"github.com/rai-project/dlperf/pkg"
	"golang.org/x/sync/errgroup"

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
		var mut sync.Mutex

		g, _ := errgroup.WithContext(context.Background())

		generateProgress := newProgress("> Generating benchmarks", len(layers))

		for ii := range layers {
			lyr := layers[ii]
			g.Go(func() error {
				defer generateProgress.Increment()
				if lyr.OperatorType() == "ConstantInput" {
					return nil
				}
				var b string
				switch strings.ToLower(lyr.OperatorType()) {
				case "conv":
					l := lyr.(*perflayer.Conv)
					b = l.FwdBenchmarkGenerator()
				case "relu":
					l := lyr.(*perflayer.Relu)
					b = l.FwdBenchmarkGenerator()
				case "pooling":
					l := lyr.(*perflayer.Pooling)
					b = l.FwdBenchmarkGenerator()
				case "softmax":
					l := lyr.(*perflayer.Softmax)
					b = l.FwdBenchmarkGenerator()
				case "batchnorm":
					l := lyr.(*perflayer.BatchNorm)
					b = l.FwdBenchmarkGenerator()
				case "dropout":
					l := lyr.(*perflayer.Dropout)
					b = l.FwdBenchmarkGenerator()
				default:
					pp.Println(lyr.OperatorType())

				}
				if b == "" {
					return nil
				}
				mut.Lock()
				defer mut.Unlock()
				prog.WriteString(b)
				return nil
			})
		}

		if err := g.Wait(); err != nil {
			return err
		}
		generateProgress.Finish()

		if outputFileName == "automatic" || outputFileName == "" {
			// println(prog.String())
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
