package dlperf

import (
	"encoding/json"
	"fmt"

	"github.com/rai-project/utils"
)

type Layer interface {
	Name() string
	OperatorType() string
	SetName(string)
	LayerInformation() LayerInfo
}

type LayerInfo interface {
	Name() string
	OperatorType() string
	Flops() FlopsInformation
	Memory() MemoryInformation
}

type FlopsInformation struct {
	InputDimensions  []int64 `json:"input_dimensions,omitempty"`
	OutputDimensions []int64 `json:"output_dimensions,omitempty"`
	MultiplyAdds     int64   `json:"multiply_adds"`
	Additions        int64   `json:"additions"`
	Divisions        int64   `json:"divisions"`
	Exponentiations  int64   `json:"exponentiations"`
	Comparisons      int64   `json:"comparisons"`
	General          int64   `json:"general"`
}

func (FlopsInformation) Header() []string {
	return []string{"InputDimensions", "OutputDimensions", "MultiplyAdds", "Additions", "Divisions", "Exponentiations", "Comparisons", "General"}
}

func (flops FlopsInformation) Row(humanFlops bool) []string {
	flopsToString := func(e int64) string {
		return fmt.Sprintf("%v", e)
	}
	dimsToString := func(e []int64) string {
		if len(e) == 0 {
			return ""
		}
		bts, err := json.Marshal(e)
		if err != nil {
			return fmt.Sprintf("%v", e)
		}
		return string(bts)
	}
	if humanFlops {
		flopsToString = func(e int64) string {
			return utils.Flops(uint64(e))
		}
	}
	return []string{
		dimsToString(flops.InputDimensions),
		dimsToString(flops.OutputDimensions),
		flopsToString(flops.MultiplyAdds),
		flopsToString(flops.Additions),
		flopsToString(flops.Divisions),
		flopsToString(flops.Exponentiations),
		flopsToString(flops.Comparisons),
		flopsToString(flops.General),
	}
}

func (this FlopsInformation) Total() int64 {
	return this.MultiplyAdds + this.Additions + this.Divisions +
		this.Exponentiations + this.Comparisons + this.General
}

func (this FlopsInformation) Add(other FlopsInformation) FlopsInformation {
	return FlopsInformation{
		MultiplyAdds:    this.MultiplyAdds + other.MultiplyAdds,
		Additions:       this.Additions + other.Additions,
		Divisions:       this.Divisions + other.Divisions,
		Exponentiations: this.Exponentiations + other.Exponentiations,
		Comparisons:     this.Comparisons + other.Comparisons,
		General:         this.General + other.General,
	}
}

type MemoryInformation struct {
	Weights    int64 `json:"weights,omitempty"`
	Activation int64 `json:"activation,omitempty"`
}

func (this MemoryInformation) Add(other MemoryInformation) MemoryInformation {
	return MemoryInformation{
		Weights:    this.Weights + other.Weights,
		Activation: this.Activation + other.Activation,
	}
}
