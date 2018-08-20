package cmd

import (
	"path/filepath"

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

		nds, err := net.ToplologicallyOrderedNodes()
		if err != nil {
			return err
		}

		benchmarkInfo := []bench{}

		for _, nd := range nds {
			layer := nd.Layer()
			if layer.OperatorType() == "ConstantInput" {
				continue
			}
			if layer.OperatorType() != "Conv" && layer.OperatorType() != "Relu" {
				pp.Println(layer.OperatorType())
				continue
			}
			filter := layer.FwdBenchmarkFilter("float", "")
			bs, err := benchSuite.Filter(filter)
			if err != nil {
				pp.ColoringEnabled = false
				log.WithError(err).WithField("filter", pp.Sprint(filter)).Error("failed to find benchmark within benchmark suite")
				pp.ColoringEnabled = true
				continue
			}
			if len(bs) == 0 {
				pp.ColoringEnabled = false
				log.WithField("filter", pp.Sprint(filter)).Error("unable to find benchmark within benchmark suite")
				pp.ColoringEnabled = true
				continue
			}
			info := layer.Information()
			for _, b := range bs {
				benchmarkInfo = append(benchmarkInfo, bench{
					benchmark: b,
					layer:     layer,
					flops:     info.Flops(),
				})
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
	benchmarkResultsFolder = filepath.Join(raiSrcPath, "microbench", "results", "cudnn", "tesla.ncsa.illinois.edu")
	benchinfoCmd.PersistentFlags().StringVar(&benchmarkResultsFolder, "benchmark_database", benchmarkResultsFolder, "path to the benchmark results folder")
	rootCmd.AddCommand(benchinfoCmd)
}
