package cmd

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/Unknwon/com"
	"github.com/k0kubun/pp"
	dlperf "github.com/rai-project/dlperf/pkg"
	perflayer "github.com/rai-project/dlperf/pkg/layer"
	"github.com/spf13/cobra"
	"golang.org/x/sync/errgroup"
)

var (
	generateFused             bool
	generateOnlyFused         bool
	generateAll               bool
	generateForward           bool
	generateBackward          bool
	generateRandomize         bool
	generateRandomizeLength   int = 5
	generatePadLayers         bool
	generatePadLayersMultiple int
)

var benchgenCmd = &cobra.Command{
	Use:     "benchgen",
	Aliases: []string{"benchmark_generate"},
	Short:   "Generates the benchmark files for layers",
	PreRunE: func(cmd *cobra.Command, args []string) error {
		if generateOnlyFused {
			generateFused = true
		}
		if err := setupDLPerfDataType(datatype); err != nil {
			return err
		}
		if generatePadLayers && generatePadLayersMultiple == 0 {
			generatePadLayersMultiple = 8
		}
		return nil
	},
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
					if generateOnlyFused && lyr.OperatorType() != "Conv" {
						return nil
					}
					var b string
					switch strings.ToLower(lyr.OperatorType()) {
					case "conv":
						l := lyr.(*perflayer.Conv)
						if !generateOnlyFused {
							if l.HasBias() {
								b += l.FwdBenchmarkGenerator(
									dlperf.FwdBenchmarkArgsOption.ConvFwdType(dlperf.ConvFwdTypeBias),
									dlperf.FwdBenchmarkArgsOption.RandomizeConv(generateRandomize),
									dlperf.FwdBenchmarkArgsOption.RandomizeConvLength(generateRandomizeLength),
									dlperf.FwdBenchmarkArgsOption.PadConv(generatePadLayers),
									dlperf.FwdBenchmarkArgsOption.PadConvMultiple(generatePadLayersMultiple),
								)
								b += "\n"
							}
							b += l.FwdBenchmarkGenerator(
								dlperf.FwdBenchmarkArgsOption.ConvFwdType(dlperf.ConvFwdTypeConv),
								dlperf.FwdBenchmarkArgsOption.RandomizeConv(generateRandomize),
								dlperf.FwdBenchmarkArgsOption.RandomizeConvLength(generateRandomizeLength),
								dlperf.FwdBenchmarkArgsOption.PadConv(generatePadLayers),
								dlperf.FwdBenchmarkArgsOption.PadConvMultiple(generatePadLayersMultiple),
							)
							b += "\n"
						}
						if generateFused {
							b += l.FwdBenchmarkGenerator(
								dlperf.FwdBenchmarkArgsOption.ConvFwdType(dlperf.ConvFwdTypeConvFusedActivation),
								dlperf.FwdBenchmarkArgsOption.RandomizeConv(generateRandomize),
								dlperf.FwdBenchmarkArgsOption.RandomizeConvLength(generateRandomizeLength),
								dlperf.FwdBenchmarkArgsOption.PadConv(generatePadLayers),
								dlperf.FwdBenchmarkArgsOption.PadConvMultiple(generatePadLayersMultiple),
							)
						}
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
					case "gemm":
						l := lyr.(*perflayer.Gemm)
						b = l.FwdBenchmarkGenerator()
					case "elementwise":
						l := lyr.(*perflayer.ElementWise)
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

		if generateBackward && !generateOnlyFused {
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
						b = l.BwdBenchmarkGenerator(dlperf.BwdBenchmarkArgsOption.ConvBwdType(dlperf.ConvBwdTypeData))
						b += "\n"
						b += l.BwdBenchmarkGenerator(dlperf.BwdBenchmarkArgsOption.ConvBwdType(dlperf.ConvBwdTypeFilter))
						b += "\n"
						b += l.BwdBenchmarkGenerator(dlperf.BwdBenchmarkArgsOption.ConvBwdType(dlperf.ConvBwdTypeBias))
					case "relu":
						l := lyr.(*perflayer.Relu)
						b = l.BwdBenchmarkGenerator()
					case "pooling":
						l := lyr.(*perflayer.Pooling)
						b = l.BwdBenchmarkGenerator()
					case "softmax":
						l := lyr.(*perflayer.Softmax)
						b = l.BwdBenchmarkGenerator()
					case "batchnorm":
						l := lyr.(*perflayer.BatchNorm)
						b = l.BwdBenchmarkGenerator()
					case "dropout":
						l := lyr.(*perflayer.Dropout)
						b = l.BwdBenchmarkGenerator()
					case "gemm":
						l := lyr.(*perflayer.Gemm)
						b = l.BwdBenchmarkGenerator()
					case "elementwise":
						l := lyr.(*perflayer.ElementWise)
						b = l.BwdBenchmarkGenerator()
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

		err = com.WriteFile(outputFileName, prog.Bytes())
		if err != nil {
			return err
		}

		return nil
	},
}

func init() {
	benchgenCmd.PersistentFlags().BoolVar(&generateOnlyFused, "only_fused", false, "generate only fused conv layers")
	benchgenCmd.PersistentFlags().BoolVar(&generateFused, "fused", false, "generate fused conv layers")
	benchgenCmd.PersistentFlags().BoolVar(&generateRandomize, "randomize", false, "generate randomized guard suffix to allow for different translation groups")
	benchgenCmd.PersistentFlags().IntVar(&generateRandomizeLength, "randomize_length", generateRandomizeLength, "number of randomized guard suffix to generate")
	benchgenCmd.PersistentFlags().BoolVar(&generatePadLayers, "pad_layers", false, "pad all layers to the nearest pad_layers_multiple")
	benchgenCmd.PersistentFlags().IntVar(&generatePadLayersMultiple, "pad_layers_multiple", 8, "padding multiple to use")
	benchgenCmd.PersistentFlags().BoolVar(&generateBackward, "backward", false, "generate the backward pass")
	benchgenCmd.PersistentFlags().BoolVar(&generateForward, "forward", true, "generate the forward pass")
	benchgenCmd.PersistentFlags().StringVar(&datatype, "datatype", "float32", "data type to use (default is float32)")
	rootCmd.AddCommand(benchgenCmd)
}
