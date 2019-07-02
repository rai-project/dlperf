package cmd

import (
	"fmt"
	"regexp"
	"strings"
	"time"

	dlperf "github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/dlperf/pkg/benchmark"
	"github.com/rai-project/dlperf/pkg/onnx"
	"github.com/rai-project/dlperf/pkg/writer"
	"github.com/rai-project/utils"
	terminal "github.com/wayneashleyberry/terminal-dimensions"
)

type pattern struct {
	onnx.Pattern
}

func (pattern) Header(opts ...writer.Option) []string {
	return []string{"Pattern", "Occurrences"}
}

func (l pattern) Row(opts ...writer.Option) []string {
	opTypes := []string{}
	for _, nd := range l.Nodes {
		opTypes = append(opTypes, nd.GetOpType())
	}
	pattern := strings.Join(opTypes, ">")
	return []string{pattern, fmt.Sprint(l.Occurrences)}
}

type bench struct {
	Type      string                  `json:"type"`
	Benchmark benchmark.Benchmark     `json:"benchmark"`
	Flops     dlperf.FlopsInformation `json:"flops_information"`
	Layer     dlperf.Layer            `json:"-"`
}

func (b bench) Header(iopts ...writer.Option) []string {
	opts := writer.NewOptions(iopts...)
	base := []string{"LayerName", "LayerType", "BenchmarkName", "RealTime(ms)", "Flops"}
	if opts.ShowMetrics {
		base = append([]string{"kernels", "metrics"})
	}
	return base
	// flopsInfo := dlperf.FlopsInformation{}.Header()
	// for ii, f := range flopsInfo {
	// 	flopsInfo[ii] = "Flops" + f
	// }
	// flopsInfo = append(flopsInfo, "FlopsTotal")
	// return append(base, flopsInfo...)
}

func getTerminalWidth() int {
	termWidth, err := terminal.Width()
	if err != nil {
		termWidth = 80
	}
	termWidth = uint(float64(termWidth) * 0.75)

	return int(termWidth)
}

func (l bench) Row(iopts ...writer.Option) []string {
	opts := writer.NewOptions(iopts...)
	termWidth := getTerminalWidth()

	ms := float64(l.Benchmark.RealTime.Nanoseconds()) / float64(time.Millisecond)
	realTime := fmt.Sprintf("%f", ms)
	benchmarkName := l.Benchmark.Name

	if opts.TrimBenchmarkName {
		benchmarkName = strings.TrimPrefix(benchmarkName, "LAYER_CUDNN_")
		benchmarkName = strings.TrimPrefix(benchmarkName, "LAYER_CUBLAS_")
		benchmarkName = strings.Replace(benchmarkName, "_FLOAT32_", "", -1)
		benchmarkName = regexp.MustCompile(`_BatchSize_\d+__`).ReplaceAllString(benchmarkName, "")
		benchmarkName = strings.Split(benchmarkName, "/")[0]
		// benchmarkName = strings.ReplaceAll(benchmarkName, "__Batch_Size_", "")

		if len(benchmarkName) > termWidth/2 {
			benchmarkName = benchmarkName[0:termWidth/2] + "..."
		}
	}

	layerName := ""
	operatorType := ""
	flops := int64(0)
	if l.Layer != nil {
		layerName = l.Layer.Name()
		if len(layerName) > 15 {
			layerName = layerName[0:14] + "..."
		}
		operatorType = l.Layer.OperatorType()
	}
	flops = l.Flops.Total()

	flopsString := fmt.Sprintf("%v", flops)
	if humanFlops {
		flopsString = utils.Flops(uint64(flops))
	}

	base := []string{layerName, operatorType, benchmarkName, realTime, flopsString}
	if opts.ShowMetrics {
		kernels := []string{}
		metrics := []string{}
		for ii, kernelInfo := range l.Benchmark.KernelInfos {
			kernels = append(kernels, fmt.Sprintf("%d:%s", ii, kernelInfo.Name))
			kernelMetrics := []string{}
			for meticName, metricValues := range kernelInfo.Metrics {
				mean := trimmedMeanUint64Slice(metricValues, DefaultTrimmedMeanFraction)
				kernelMetrics = append(kernelMetrics, fmt.Sprintf("%d:%s:%v", ii, meticName, mean))
			}
			metrics = append(metrics, strings.Join(kernelMetrics, ";"))
		}
		base = append(base, strings.Join(kernels, ";"), strings.Join(metrics, ";"))
	}
	return base
	// flops := l.flops.Row(humanFlops)
	// flops = append(flops, flopsToString(l.flops.Total(), humanFlops))

	// return append(base, flops...)
}

type stat struct {
	Name                    string   `json:"name,omitempty"`
	Type                    string   `json:"type,omitempty"`
	InputNames              []string `json:"inputs,omitempty"`
	OutputNames             []string `json:"outputs,omitempty"`
	dlperf.ShapeInformation `json:"input_dimensions"`
}

func (stat) Header(opts ...writer.Option) []string {
	base := dlperf.ShapeInformation{}.Header()
	return append([]string{"LayerName", "LayerType", "InputNames", "OutputNames"}, base...)
}

func (l stat) Row(opts ...writer.Option) []string {
	base := l.ShapeInformation.Row(opts...)
	return append([]string{l.Name, l.Type, strings.Join(l.InputNames, ";"), strings.Join(l.OutputNames, ";")}, base...)
}

type layerFlops struct {
	Name                    string `json:"name"`
	Type                    string `json:"type"`
	dlperf.FlopsInformation `json:",inline,flatten""`
	Total                   int64 `json:"total"`
}

func (layerFlops) Header(opts ...writer.Option) []string {
	base := dlperf.FlopsInformation{}.Header(opts...)
	base = append(base, "Total")
	return append([]string{"LayerName", "LayerType"}, base...)
}

func (l layerFlops) Row(iopts ...writer.Option) []string {
	base := l.FlopsInformation.Row(iopts...)
	opts := writer.NewOptions(iopts...)
	base = append(base, flopsToString(l.FlopsInformation.Total(), opts.ShowHumanFlops))
	return append([]string{l.Name, l.Type}, base...)
}

type layerWeights struct {
	Name              string  `json:"name"`
	Type              string  `json:"type"`
	Length            int     `json:"length"`
	Max               float64 `json:"max"`
	Min               float64 `json:"min"`
	StandardDeviation float64 `json:"standard_deviation"`
}

func (layerWeights) Header(opts ...writer.Option) []string {
	return []string{"LayerName", "LayerType", "Length", "LayerWeightsMax", "LayerWeightsMin", "LayerWeightsSdev"}
}

func (l layerWeights) Row(opts ...writer.Option) []string {
	return []string{l.Name, l.Type, fmt.Sprint(l.Length), fmt.Sprint(l.Max), fmt.Sprint(l.Min), fmt.Sprint(l.StandardDeviation)}
}

type netFlopsSummary struct {
	Name  string `json:"name"`
	Value int64  `json:"value"`
}

func (netFlopsSummary) Header(opts ...writer.Option) []string {
	return []string{"Flop Type", "#"}
}

func (l netFlopsSummary) Row(iopts ...writer.Option) []string {
	opts := writer.NewOptions(iopts...)
	return []string{l.Name, flopsToString(l.Value, opts.ShowHumanFlops)}
}
