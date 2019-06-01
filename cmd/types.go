package cmd

import (
	"fmt"
	"strings"
	"time"

	dlperf "github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/dlperf/pkg/benchmark"
	"github.com/rai-project/dlperf/pkg/onnx"
	terminal "github.com/wayneashleyberry/terminal-dimensions"
)

type pattern struct {
	onnx.Pattern
}

func (pattern) Header() []string {
	return []string{"Pattern", "Occurrences"}
}

func (l pattern) Row(humanFlops bool) []string {
	opTypes := []string{}
	for _, nd := range l.Nodes {
		opTypes = append(opTypes, nd.GetOpType())
	}
	pattern := strings.Join(opTypes, ">")
	return []string{pattern, fmt.Sprint(l.Occurrences)}
}

type bench struct {
	Layer     dlperf.Layer            `json:"layer"`
	Benchmark benchmark.Benchmark     `json:"benchmark"`
	Flops     dlperf.FlopsInformation `json:"flops_information"`
}

func (bench) Header() []string {
	base := []string{"LayerName", "LayerType", "BenchmarkName", "RealTime(ms)", "Flops"}
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

func (l bench) Row(humanFlops bool) []string {
	termWidth := getTerminalWidth()

	ms := float64(l.Benchmark.RealTime.Nanoseconds()) / float64(time.Millisecond)
	realTime := fmt.Sprintf("%f", ms)
	benchmarkName := l.Benchmark.Name

	benchmarkName = strings.TrimPrefix(benchmarkName, "LAYER_CUDNN_")
	benchmarkName = strings.TrimPrefix(benchmarkName, "LAYER_CUBLAS_")
	benchmarkName = strings.ReplaceAll(benchmarkName, "_FLOAT32_", "")
	benchmarkName = strings.Split(benchmarkName, "/")[0]
	// benchmarkName = strings.ReplaceAll(benchmarkName, "__Batch_Size_", "")

	if len(benchmarkName) > termWidth/2 {
		benchmarkName = benchmarkName[0:termWidth/2] + "..."
	}
	layerName := ""
	operatorType := ""
	flops := int64(0)
	if l.Layer != nil {
		layerName = l.Layer.Name()
		operatorType = l.Layer.OperatorType()
	}
	flops = l.Flops.Total()

	base := []string{layerName, operatorType, benchmarkName, realTime, fmt.Sprintf("%v", flops)}
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

func (stat) Header() []string {
	base := dlperf.ShapeInformation{}.Header()
	return append([]string{"LayerName", "LayerType", "InputNames", "OutputNames"}, base...)
}

func (l stat) Row(humanFlops bool) []string {
	base := l.ShapeInformation.Row()
	return append([]string{l.Name, l.Type, strings.Join(l.InputNames, ";"), strings.Join(l.OutputNames, ";")}, base...)
}

type layerFlops struct {
	Name                    string `json:"name"`
	Type                    string `json:"type"`
	dlperf.FlopsInformation `json:",inline,flatten""`
	Total                   int64 `json:"total"`
}

func (layerFlops) Header() []string {
	base := dlperf.FlopsInformation{}.Header()
	base = append(base, "Total")
	return append([]string{"LayerName", "LayerType"}, base...)
}

func (l layerFlops) Row(humanFlops bool) []string {
	base := l.FlopsInformation.Row(humanFlops)
	base = append(base, flopsToString(l.FlopsInformation.Total(), humanFlops))
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

func (layerWeights) Header() []string {
	return []string{"LayerName", "LayerType", "Length", "LayerWeightsMax", "LayerWeightsMin", "LayerWeightsSdev"}
}

func (l layerWeights) Row(humanFlops bool) []string {
	return []string{l.Name, l.Type, fmt.Sprint(l.Length), fmt.Sprint(l.Max), fmt.Sprint(l.Min), fmt.Sprint(l.StandardDeviation)}
}

type netFlopsSummary struct {
	Name  string `json:"name"`
	Value int64  `json:"value"`
}

func (netFlopsSummary) Header() []string {
	return []string{"Flop Type", "#"}
}

func (l netFlopsSummary) Row(humanFlops bool) []string {
	return []string{l.Name, flopsToString(l.Value, humanFlops)}
}
