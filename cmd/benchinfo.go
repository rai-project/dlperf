package cmd

import (
	"fmt"
	"path/filepath"
	"strings"
	"time"

	sourcepath "github.com/GeertJohan/go-sourcepath"
	"github.com/k0kubun/pp"
	"github.com/pkg/errors"
	"github.com/spf13/cobra"

	"github.com/rai-project/config"
	dlperf "github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/dlperf/pkg/benchmark"
	"github.com/rai-project/dlperf/pkg/onnx"
)

var (
	benchmarkResultsFolder string
	benchInfoTraining      bool
	benchInfoShort         bool
	benchInfoDataType      string
)

// benchinfoCmd represents the benchinfo command
var benchinfoCmd = &cobra.Command{
	Use:     "benchinfo",
	Aliases: []string{"info", "recall"},
	Short:   "Prints out the benchmark information",
	RunE: func(cmd *cobra.Command, args []string) error {
		modelPath = expandModelPath(modelPath)

		dlperf.SetBatchSize(batchSize)

		benchSuite, err := benchmark.New(benchmarkResultsFolder)
		if err != nil {
			return err
		}

		model, err := onnx.New(modelPath, batchSize)
		if err != nil {
			return err
		}

		net := model.Network()

		nds, err := net.TopologicallyOrderedNodes()
		if err != nil {
			return err
		}

		benchmarkInfo := []bench{}

		totalTime := time.Duration(0)
		totalFlops := dlperf.FlopsInformation{}

		debugPrint := func(val ...interface{}) {
			if config.App.IsDebug {
				fmt.Println(val...)
			}
		}

		for _, nd := range nds {
			lyr := nd.Layer()
			switch strings.ToLower(lyr.OperatorType()) {
			case "constantinput":
				continue
			case "lrn":
				debugPrint("LRN is not supported by CUDNN")
				continue
			case "reshape":
				debugPrint("Reshape is not supported by CUDNN")
				continue
			// case "dropout":
			// 	fmt.Println("Dropout is skipped for now")
			// 	continue
			case "concat":
				debugPrint("Concat is skipped for now")
				continue
			case "flatten":
				debugPrint("Faltten is skipped for now")
				continue
			case "globalpooling":
				debugPrint("GlobalPooling is skipped for now")
				continue
			case "identity":
				debugPrint("Identity is skipped for now")
				continue
			case "elementwise":
				debugPrint("Elementwise is not supported by CUDNN")
				continue
			}
			// if lyr.OperatorType() != "Conv" && lyr.OperatorType() != "Relu" {
			// 	pp.Println(lyr.OperatorType())
			// 	continue
			// }

			filterBenchmarks := func(benchInfoBackward bool, datatype string, algorithm string, iopts ...interface{}) *benchmark.Benchmark {
				if benchInfoBackward {
					opts := make([]dlperf.BwdBenchmarkArgsOptionFunc, len(iopts))
					for ii, opt := range iopts {
						opts[ii] = opt.(dlperf.BwdBenchmarkArgsOptionFunc)
					}
					filter := lyr.BwdBenchmarkFilter(datatype, algorithm, opts...)
					return &filter
				}

				opts := make([]dlperf.FwdBenchmarkArgsOptionFunc, len(iopts))
				for ii, opt := range iopts {
					opts[ii] = opt.(dlperf.FwdBenchmarkArgsOptionFunc)
				}
				filter := lyr.FwdBenchmarkFilter(datatype, algorithm, opts...)
				return &filter
			}

			getBenchmarkTime := func(filter *benchmark.Benchmark) (benchmark.Benchmarks, error) {
				if filter == nil {
					pp.Println(lyr.Name())
					pp.Println(filter)
					return nil, errors.New("empty filter")
				}
				bs, err := benchSuite.Filter(*filter)
				if err != nil {
					// pp.ColoringEnabled = false
					// log.WithError(err).WithField("filter", pp.Sprint(filter)).Error("failed to find benchmark within benchmark suite")
					// pp.ColoringEnabled = true
					// continue

					pp.Println(lyr.Name())
					pp.Println(filter)
					return nil, errors.New("invalid filter benchmarks")
				}
				if len(bs) == 0 {
					// pp.ColoringEnabled = false
					// log.WithField("filter", pp.Sprint(filter)).Error("unable to find benchmark within benchmark suite")
					// pp.ColoringEnabled = true
					// continue
					// pp.Println(lyr.OperatorType())
					// pp.Println(lyr.Name())
					pp.Println(filter)
					return nil, errors.New("no benchmarks")
				}
				return bs, nil
			}

			addLayerInfos := func(bs benchmark.Benchmarks) {
				bs.Sort()

				info := lyr.Information()
				if len(bs) > 0 {
					totalTime = totalTime + bs[0].RealTime
					totalFlops = totalFlops.Add(lyr.Information().Flops())
				}
				if !benchInfoShort {
					for _, b := range bs {
						benchmarkInfo = append(benchmarkInfo, bench{
							Benchmark: b,
							Layer:     lyr,
							Flops:     info.Flops(),
						})
					}
				}
			}

			switch strings.ToLower(lyr.OperatorType()) {
			case "relu", "pooling", "softmax", "dropout", "gemm", "matmul":
				filter := filterBenchmarks(false, benchInfoDataType, "")
				bs, err := getBenchmarkTime(filter)
				if err != nil {
					continue
				}
				addLayerInfos(bs)

				if benchInfoTraining {
					filter := filterBenchmarks(true, benchInfoDataType, "")
					bs, err := getBenchmarkTime(filter)
					if err != nil {
						continue
					}
					addLayerInfos(bs)
				}
			case "conv":
				filter := filterBenchmarks(false, benchInfoDataType, "")
				bs, err := getBenchmarkTime(filter)
				if err != nil {
					continue
				}
				addLayerInfos(bs)

				if benchInfoTraining {
					filter := filterBenchmarks(true, benchInfoDataType, "", dlperf.BwdBenchmarkArgsOption.ConvBwdType(dlperf.ConvBwdTypeData))
					bs, err := getBenchmarkTime(filter)
					if err != nil {
						pp.Println("unable to get conv data")
						continue
					}
					addLayerInfos(bs)

					filter = filterBenchmarks(true, benchInfoDataType, "", dlperf.BwdBenchmarkArgsOption.ConvBwdType(dlperf.ConvBwdTypeFilter))
					bs, err = getBenchmarkTime(filter)
					if err != nil {
						pp.Println("unable to get conv filter because of " + err.Error())
						continue
					}
					addLayerInfos(bs)

					filter = filterBenchmarks(true, benchInfoDataType, "", dlperf.BwdBenchmarkArgsOption.ConvBwdType(dlperf.ConvBwdTypeBias))
					bs, err = getBenchmarkTime(filter)
					if err != nil {
						pp.Println("unable to get conv bias")
						continue
					}
					addLayerInfos(bs)
				}

			case "batchnorm":
				var filter *benchmark.Benchmark
				if benchInfoTraining {
					filter = filterBenchmarks(false, benchInfoDataType, "", dlperf.FwdBenchmarkArgsOption.IsTraining(true))
				} else {
					filter = filterBenchmarks(false, benchInfoDataType, "", dlperf.FwdBenchmarkArgsOption.IsTraining(false))
				}
				bs, err := getBenchmarkTime(filter)
				if err != nil {
					continue
				}
				addLayerInfos(bs)

				if benchInfoTraining {
					filter := filterBenchmarks(true, benchInfoDataType, "")
					bs, err := getBenchmarkTime(filter)
					if err != nil {
						continue
					}
					addLayerInfos(bs)
				}

			default:
				pp.Println(lyr.OperatorType())

			}
		}

		benchmarkInfo = append(benchmarkInfo, bench{
			Benchmark: benchmark.Benchmark{
				Name:     "Total",
				RealTime: totalTime,
			},
			Layer: nil,
			Flops: totalFlops,
		})

		if benchSuite.GPUInformation != nil && len(benchSuite.GPUInformation.GPUS) != 0 {
			for _, gpu := range benchSuite.GPUInformation.GPUS {
				fmt.Println(gpu.ProductName)
			}
		}
		writer := NewWriter(bench{}, humanFlops)
		defer writer.Close()

		for _, bench := range benchmarkInfo {
			writer.Row(bench)
		}

		return nil
	},
}

func init() {
	benchmarkResultsFolder = filepath.Join(sourcepath.MustAbsoluteDir(), "..", "results")
	benchinfoCmd.PersistentFlags().BoolVar(&benchInfoTraining, "training", false, "compute the training information")
	benchinfoCmd.PersistentFlags().BoolVar(&benchInfoShort, "short", false, "only get info about the total, rather than reporting per-layer information")
	benchinfoCmd.PersistentFlags().StringVar(&benchInfoDataType, "datatype", "float32", "compute the information for the specified scalar datatype")
	benchinfoCmd.PersistentFlags().StringVar(&benchmarkResultsFolder, "benchmark_database", benchmarkResultsFolder, "path to the benchmark results folder")
	rootCmd.AddCommand(benchinfoCmd)
}
