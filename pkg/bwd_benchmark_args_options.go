package dlperf

type ConvBwdType int

const (
	ConvBwdTypeData   ConvBwdType = 0
	ConvBwdTypeFilter ConvBwdType = 1
	ConvBwdTypeBias   ConvBwdType = 2
)

type bwdBenchmarkArgsOptions struct {
	ConvBwdType ConvBwdType
}

type BwdBenchmarkArgsOptionFunc func(*bwdBenchmarkArgsOptions)

type bwdBenchmarkArgsOptionHandler struct {
	ConvBwdType func(ConvBwdType) BwdBenchmarkArgsOptionFunc
}

var BwdBenchmarkArgsOption = bwdBenchmarkArgsOptionHandler{
	ConvBwdType: func(convBWDType ConvBwdType) BwdBenchmarkArgsOptionFunc {
		return func(o *bwdBenchmarkArgsOptions) {
			o.ConvBwdType = convBWDType
		}
	},
}

func CreateBwdBenchmarkArgsOption(os ...BwdBenchmarkArgsOptionFunc) *bwdBenchmarkArgsOptions {
	opts := &bwdBenchmarkArgsOptions{
		ConvBwdType: ConvBwdTypeData,
	}
	for _, o := range os {
		o(opts)
	}
	return opts
}
