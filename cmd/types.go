package cmd

import (
	"fmt"
	"strings"

	"github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/dlperf/pkg/onnx"
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

type stat struct {
	Name             string   `json:"name"`
	Type             string   `json:"type"`
	Inputs           []string `json:"inputs"`
	Outputs          []string `json:"outputs"`
	InputDimensions  []int64  `json:"input_dimensions"`
	OutputDimensions []int64  `json:"output_dimensions"`
}

func (stat) Header() []string {
	return []string{"LayerName", "LayerType", "Inputs", "Outputs", "InputDimension", "OutputDimension"}
}

func (l stat) Row(humanFlops bool) []string {
	inputs := strings.Join(l.Inputs, ";")
	outputs := strings.Join(l.Outputs, ";")
	inputDimensions := strings.Trim(strings.Replace(fmt.Sprint(l.InputDimensions), " ", ";", -1), "[]")
	outputDimensions := strings.Trim(strings.Replace(fmt.Sprint(l.OutputDimensions), " ", ";", -1), "[]")

	return []string{l.Name, l.Type, inputs, outputs, inputDimensions, outputDimensions}
}

type layer struct {
	Name                    string `json:"name"`
	Type                    string `json:"type"`
	dlperf.FlopsInformation `json:",inline,flatten""`
	Total                   int64 `json:"total"`
}

func (layer) Header() []string {
	base := dlperf.FlopsInformation{}.Header()
	base = append(base, "Total")
	return append([]string{"LayerName", "LayerType"}, base...)
}

func (l layer) Row(humanFlops bool) []string {
	base := l.FlopsInformation.Row(humanFlops)
	base = append(base, flopsToString(l.FlopsInformation.Total(), humanFlops))
	return append([]string{l.Name, l.Type}, base...)
}

type netSummary struct {
	Name  string `json:"name"`
	Value int64  `json:"value"`
}

func (netSummary) Header() []string {
	return []string{"Flop Type", "#"}
}

func (l netSummary) Row(humanFlops bool) []string {
	return []string{l.Name, flopsToString(l.Value, humanFlops)}
}
