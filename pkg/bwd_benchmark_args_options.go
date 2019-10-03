package dlperf

type ConvBwdType int

const (
	ConvBwdTypeUndefined ConvBwdType = 0
	ConvBwdTypeData      ConvBwdType = 1
	ConvBwdTypeFilter    ConvBwdType = 2
	ConvBwdTypeBias      ConvBwdType = 3
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
