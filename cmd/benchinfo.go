package cmd

import (
	"fmt"
	"path/filepath"
	"strings"
	"time"

	"github.com/rai-project/dlperf/pkg"

	sourcepath "github.com/GeertJohan/go-sourcepath"
	"github.com/k0kubun/pp"
	"github.com/rai-project/dlperf/pkg/benchmark"
	"github.com/rai-project/dlperf/pkg/onnx"
	"github.com/spf13/cobra"
)

var (
	benchmarkResultsFolder string
)

// benchinfoCmd represents the benchinfo command
var benchinfoCmd = &cobra.Command{
	Use:     "benchinfo",
	Aliases: []string{"bench"},
	Short:   "Prints out the benchmark information",
	RunE: func(cmd *cobra.Command, args []string) error {
		if modelPath == "" {
			modelPath = filepath.Join(sourcepath.MustAbsoluteDir(), "..", "assets", "onnx_models", "mnist.onnx")
		} else {
			s, err := filepath.Abs(modelPath)
			if err == nil {
				modelPath = s
			}
		}

		benchSuite, err := benchmark.New(benchmarkResultsFolder)
		if err != nil {
			return err
		}

		model, err := onnx.New(modelPath)
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

		for _, nd := range nds {
			lyr := nd.Layer()
			switch strings.ToLower(lyr.OperatorType()) {
			case "constantinput":
				continue
			case "lrn":
				fmt.Println("LRN is not supported by CUDNN")
				continue
			case "reshape":
				fmt.Println("Reshape is not supported by CUDNN")
				continue
			case "gemm":
				fmt.Println("Gemm is not supported by CUDNN")
				continue
			case "dropout":
				fmt.Println("Dropout is skipped for now")
				continue
			case "concat":
				fmt.Println("Concat is skipped for now")
				continue
			case "identity":
				fmt.Println("Identity is skipped for now")
				continue
			case "elementwise":
				fmt.Println("Elementwise is not supported by CUDNN")
				continue
			}
			// if lyr.OperatorType() != "Conv" && lyr.OperatorType() != "Relu" {
			// 	pp.Println(lyr.OperatorType())
			// 	continue
			// }
			filter := lyr.FwdBenchmarkFilter("float32", "")
			bs, err := benchSuite.Filter(filter)
			if err != nil {
				// pp.ColoringEnabled = false
				// log.WithError(err).WithField("filter", pp.Sprint(filter)).Error("failed to find benchmark within benchmark suite")
				// pp.ColoringEnabled = true
				// continue

				pp.Println(lyr.Name())
				pp.Println(filter)
				continue
			}
			if len(bs) == 0 {
				// pp.ColoringEnabled = false
				// log.WithField("filter", pp.Sprint(filter)).Error("unable to find benchmark within benchmark suite")
				// pp.ColoringEnabled = true
				// continue
				pp.Println(lyr.Name())
				pp.Println(filter)
				continue
			}
			info := lyr.Information()
			if len(bs) > 0 {
				totalTime = totalTime + bs[0].RealTime
			}
			for _, b := range bs {
				benchmarkInfo = append(benchmarkInfo, bench{
					benchmark: b,
					layer:     lyr,
					flops:     info.Flops(),
				})
			}
		}
		benchmarkInfo = append(benchmarkInfo, bench{
			benchmark: benchmark.Benchmark{
				Name:     "Total",
				RealTime: totalTime,
			},
			layer: nil,
			flops: dlperf.FlopsInformation{},
		})

		writer := NewWriter(bench{}, humanFlops)
		defer writer.Close()

		for _, bench := range benchmarkInfo {
			writer.Row(bench)
		}

		return nil
	},
}

func init() {
	benchmarkResultsFolder = filepath.Join(raiSrcPath, "microbench", "results", "cudnn", "tesla.ncsa.illinois.edu")
	benchinfoCmd.PersistentFlags().StringVar(&benchmarkResultsFolder, "benchmark_database", benchmarkResultsFolder, "path to the benchmark results folder")
	rootCmd.AddCommand(benchinfoCmd)
}
