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
