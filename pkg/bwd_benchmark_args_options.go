package dlperf

type bwdBenchmarkArgsOptions struct {
}

type BwdBenchmarkArgsOptionFunc func(*bwdBenchmarkArgsOptions)

type bwdBenchmarkArgsOptionHandler struct {
}

var BwdBenchmarkArgsOption = bwdBenchmarkArgsOptionHandler{}

func CreateBwdBenchmarkArgsOption(os ...BwdBenchmarkArgsOptionFunc) *bwdBenchmarkArgsOptions {
	opts := &bwdBenchmarkArgsOptions{}
	for _, o := range os {
		o(opts)
	}
	return opts
}
