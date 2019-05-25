package cmd

import (
	"bytes"
	"context"
	"fmt"
	"io/ioutil"
	"strings"
	"sync"

	"github.com/k0kubun/pp"
	dlperf "github.com/rai-project/dlperf/pkg"
	"golang.org/x/sync/errgroup"

	perflayer "github.com/rai-project/dlperf/pkg/layer"
	"github.com/spf13/cobra"
)

var (
	generateForward  bool
	generateBackward bool
)

var benchgenCmd = &cobra.Command{
	Use:     "benchgen",
	Aliases: []string{"benchmark_generate"},
	Short:   "Generates the benchmark files for layers",
	RunE: func(cmd *cobra.Command, args []string) error {
		modelPath = expandModelPath(modelPath)

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

		fmt.Printf("computing the union of %d layers\n", len(layers))

		layers = layers.FwdUnion("float", "")

		fmt.Printf("reduced the number of layers to %d layers\n", len(layers))

		prog := bytes.NewBufferString("")
		var mut sync.Mutex

		g, _ := errgroup.WithContext(context.Background())

		numToGenerate := 0
		if generateForward {
			numToGenerate += len(layers)
		}
		if generateBackward {
			numToGenerate += len(layers)
		}
		generateProgress := newProgress("> Generating benchmarks", numToGenerate)

		if generateForward {
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
						b = l.FwdBenchmarkGenerator(dlperf.FwdBenchmarkArgsOption.IsTraining(false))
						b += "\n"
						b += l.FwdBenchmarkGenerator(dlperf.FwdBenchmarkArgsOption.IsTraining(true))
					case "dropout":
						l := lyr.(*perflayer.Dropout)
						b = l.FwdBenchmarkGenerator()
					default:
						// pp.Println(lyr.OperatorType())
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
		}

		if generateBackward {
			for ii := range layers {
				lyr := layers[ii]
				g.Go(func() error {
					defer generateProgress.Increment()
					if lyr.OperatorType() == "ConstantInput" {
						return nil
					}
					var b string
					switch strings.ToLower(lyr.OperatorType()) {
					// case "conv":
					// 	l := lyr.(*perflayer.Conv)
					// 	b = l.BwdBenchmarkGenerator()
					// case "relu":
					// 	l := lyr.(*perflayer.Relu)
					// 	b = l.BwdBenchmarkGenerator()
					// case "pooling":
					// 	l := lyr.(*perflayer.Pooling)
					// 	b = l.BwdBenchmarkGenerator()
					// case "softmax":
					// 	l := lyr.(*perflayer.Softmax)
					// 	b = l.BwdBenchmarkGenerator()
					case "batchnorm":
						l := lyr.(*perflayer.BatchNorm)
						b = l.BwdBenchmarkGenerator()
					// case "dropout":
					// 	l := lyr.(*perflayer.Dropout)
					// 	b = l.BwdBenchmarkGenerator()
					default:
						// pp.Println(lyr.OperatorType())
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
		}

		if err := g.Wait(); err != nil {
			return err
		}
		generateProgress.Finish()

		if outputFileName == "automatic" || outputFileName == "" {
			// println(prog.String())
			pp.Println("Need to specify output_path to output to a file")
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
	benchgenCmd.PersistentFlags().BoolVar(&generateBackward, "backward", false, "generate backward layers")
	benchgenCmd.PersistentFlags().BoolVar(&generateForward, "forward", true, "generate forward layers")
	rootCmd.AddCommand(benchgenCmd)
}
