package dlperf

type ConvFwdType int

const (
	ConvFwdTypeUndefined           ConvFwdType = 0
	ConvFwdTypeConv                ConvFwdType = 1
	ConvFwdTypeBias                ConvFwdType = 2
	ConvFwdTypeConvFusedActivation ConvFwdType = 2
)

type fwdBenchmarkArgsOptions struct {
	IsTraining  bool
	ConvFwdType ConvFwdType
}

type FwdBenchmarkArgsOptionFunc func(*fwdBenchmarkArgsOptions)

type fwdBenchmarkArgsOptionHandler struct {
	IsTraining  func(bool) FwdBenchmarkArgsOptionFunc
	ConvFwdType func(ConvFwdType) FwdBenchmarkArgsOptionFunc
}

var FwdBenchmarkArgsOption = fwdBenchmarkArgsOptionHandler{
	IsTraining: func(isTraining bool) FwdBenchmarkArgsOptionFunc {
		return func(o *fwdBenchmarkArgsOptions) {
			o.IsTraining = isTraining
		}
	},
	ConvFwdType: func(convFWDType ConvFwdType) FwdBenchmarkArgsOptionFunc {
		return func(o *fwdBenchmarkArgsOptions) {
			o.ConvFwdType = convFWDType
		}
	},
}

func CreateFwdBenchmarkArgsOption(os ...FwdBenchmarkArgsOptionFunc) *fwdBenchmarkArgsOptions {
	opts := &fwdBenchmarkArgsOptions{
		IsTraining:  false,
		ConvFwdType: ConvFwdTypeConv,
	}
	for _, o := range os {
		o(opts)
	}
	return opts
}
