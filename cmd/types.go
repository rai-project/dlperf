package cmd

import (
	"github.com/rai-project/dlperf"
)

type stat struct {
	Name                    string `json:"name"`
	Type                    string `json:"type"`
	dlperf.ShapeInformation `json:",inline,flatten""`
}

func (stat) Header() []string {
	base := dlperf.ShapeInformation{}.Header()
	return append([]string{"LayerName", "LayerType"}, base...)
}

func (l stat) Row(humanFlops bool) []string {
	base := l.ShapeInformation.Row(humanFlops)
	return append([]string{l.Name, l.Type}, base...)
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
