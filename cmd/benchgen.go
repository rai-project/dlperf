package cmd

import (
	"bytes"
	"context"
	"fmt"
	"io/ioutil"
	"path/filepath"
	"strings"
	"sync"

	"github.com/rai-project/dlperf/pkg"
	"golang.org/x/sync/errgroup"

	sourcepath "github.com/GeertJohan/go-sourcepath"
	perflayer "github.com/rai-project/dlperf/pkg/layer"
	"github.com/spf13/cobra"
)

var (
	standAloneGenerate = false
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

		fmt.Printf("computing the union of %d layers\n", len(layers))

		layers = layers.FwdUnion("float", "")

		fmt.Printf("reduced the number of layers to %d layers\n", len(layers))

		layerProgs := map[string]*bytes.Buffer{}

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
				layerType := strings.ToLower(lyr.OperatorType())
				switch layerType {
				case strings.ToLower(conv.OperatorType()):
					l := lyr.(*perflayer.Conv)
					b = l.FwdBenchmarkGenerator(standAloneGenerate)
				case strings.ToLower(relu.OperatorType()):
					l := lyr.(*perflayer.Relu)
					b = l.FwdBenchmarkGenerator(standAloneGenerate)
				case strings.ToLower(pooling.OperatorType()):
					l := lyr.(*perflayer.Pooling)
					b = l.FwdBenchmarkGenerator(standAloneGenerate)
				case strings.ToLower(softmax.OperatorType()):
					l := lyr.(*perflayer.Softmax)
					b = l.FwdBenchmarkGenerator(standAloneGenerate)
				case strings.ToLower(batchNorm.OperatorType()):
					l := lyr.(*perflayer.BatchNorm)
					b = l.FwdBenchmarkGenerator(standAloneGenerate)
				case strings.ToLower(dropout.OperatorType()):
					l := lyr.(*perflayer.Dropout)
					b = l.FwdBenchmarkGenerator(standAloneGenerate)
				default:
					// pp.Println(lyr.OperatorType())
				}
				if b == "" {
					return nil
				}
				mut.Lock()
				defer mut.Unlock()
				if _, ok := layerProgs[layerType]; !ok {
					layerProgs[layerType] = bytes.NewBufferString("")
				}
				layerProgs[layerType].WriteString(b)
				return nil
			})
		}

		if err := g.Wait(); err != nil {
			return err
		}

		prog := bytes.NewBufferString("")
		for layerType, val := range layerProgs {
			func() {
				switch layerType {
				case strings.ToLower(conv.OperatorType()):
					prog.WriteString(conv.FwdBenchmarkGeneratorPrefix(standAloneGenerate))
					defer prog.WriteString(conv.FwdBenchmarkGeneratorSuffix(standAloneGenerate))
				case strings.ToLower(relu.OperatorType()):
					prog.WriteString(relu.FwdBenchmarkGeneratorPrefix(standAloneGenerate))
					defer prog.WriteString(relu.FwdBenchmarkGeneratorSuffix(standAloneGenerate))
				case strings.ToLower(pooling.OperatorType()):
					prog.WriteString(pooling.FwdBenchmarkGeneratorPrefix(standAloneGenerate))
					defer prog.WriteString(pooling.FwdBenchmarkGeneratorSuffix(standAloneGenerate))
				case strings.ToLower(softmax.OperatorType()):
					prog.WriteString(softmax.FwdBenchmarkGeneratorPrefix(standAloneGenerate))
					defer prog.WriteString(softmax.FwdBenchmarkGeneratorSuffix(standAloneGenerate))
				case strings.ToLower(batchNorm.OperatorType()):
					prog.WriteString(batchNorm.FwdBenchmarkGeneratorPrefix(standAloneGenerate))
					defer prog.WriteString(batchNorm.FwdBenchmarkGeneratorSuffix(standAloneGenerate))
				case strings.ToLower(dropout.OperatorType()):
					prog.WriteString(dropout.FwdBenchmarkGeneratorPrefix(standAloneGenerate))
					defer prog.WriteString(dropout.FwdBenchmarkGeneratorSuffix(standAloneGenerate))
				default:
					// pp.Println(lyr.OperatorType())
				}

				prog.Write(val.Bytes())
			}()
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
	benchinfoCmd.PersistentFlags().BoolVar(&standAloneGenerate, "standalone", false, "generate benchmarks so that they are all standalone")
	rootCmd.AddCommand(benchgenCmd)
}
