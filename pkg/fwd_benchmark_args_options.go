package dlperf

type ConvFwdType int

const (
	ConvFwdTypeUndefined           ConvFwdType = 0
	ConvFwdTypeConv                ConvFwdType = 1
	ConvFwdTypeBias                ConvFwdType = 2
	ConvFwdTypeConvFusedActivation ConvFwdType = 3
)

// type ElementWiseType int

// const (
// 	ElementWiseTypeUndefined ElementWiseType = 0
// 	ElementWiseTypeAdd       ElementWiseType = 1
// 	ElementWiseTypeMul       ElementWiseType = 2
// 	ElementWiseTypeMin       ElementWiseType = 3
// 	ElementWiseTypeMax       ElementWiseType = 4
// 	ElementWiseTypeSqrt      ElementWiseType = 5
// 	ElementWiseTypeNot       ElementWiseType = 6
// )

type fwdBenchmarkArgsOptions struct {
	IsTraining          bool
	ConvFwdType         ConvFwdType
	RandomizeConv       bool
	RandomizeConvLength int
	// ElementWiseType ElementWiseType
}

type FwdBenchmarkArgsOptionFunc func(*fwdBenchmarkArgsOptions)

type fwdBenchmarkArgsOptionHandler struct {
	IsTraining          func(bool) FwdBenchmarkArgsOptionFunc
	ConvFwdType         func(ConvFwdType) FwdBenchmarkArgsOptionFunc
	RandomizeConv       func(bool) FwdBenchmarkArgsOptionFunc
	RandomizeConvLength func(int) FwdBenchmarkArgsOptionFunc
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
	RandomizeConv: func(bl bool) FwdBenchmarkArgsOptionFunc {
		return func(o *fwdBenchmarkArgsOptions) {
			o.RandomizeConv = bl
		}
	},
	RandomizeConvLength: func(ln int) FwdBenchmarkArgsOptionFunc {
		return func(o *fwdBenchmarkArgsOptions) {
			o.RandomizeConvLength = ln
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
