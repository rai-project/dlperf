package onnx

type GraphOptions struct {
	PruneInputs           bool
	InputsAsConstantNodes bool
}

type GraphOption func(*GraphOptions)

func GraphPruneInputs(b bool) GraphOption {
	return func(o *GraphOptions) {
		o.PruneInputs = b
	}
}

func GraphInputsAsConstantNodes(b bool) GraphOption {
	return func(o *GraphOptions) {
		o.InputsAsConstantNodes = b
	}
}

func NewGraphOptions(oo ...GraphOption) GraphOptions {
	opts := &GraphOptions{
		PruneInputs:           true,
		InputsAsConstantNodes: false,
	}
	for _, o := range oo {
		o(opts)
	}
	return *opts
}

type PatternOptions struct {
	PruneGraph       bool
	PruneGraphLayers []string
	Length           int
}

type PatternOption func(*PatternOptions)

func PatternPruneGraph(b bool) PatternOption {
	return func(o *PatternOptions) {
		o.PruneGraph = b
	}
}

func PatternPruneGraphLayers(s []string) PatternOption {
	return func(o *PatternOptions) {
		o.PruneGraphLayers = s
	}
}

func PatternLength(i int) PatternOption {
	return func(o *PatternOptions) {
		o.Length = i
	}
}

func NewPatternOptions(oo ...PatternOption) PatternOptions {
	opts := &PatternOptions{
		PruneGraph:       false,
		PruneGraphLayers: nil,
		Length:           2,
	}
	for _, o := range oo {
		o(opts)
	}
	return *opts
}
