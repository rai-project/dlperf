package dlperf

type fwdBenchmarkArgsOptions struct {
	IsTraining bool
}

type FwdBenchmarkArgsOptionFunc func(*fwdBenchmarkArgsOptions)

type fwdBenchmarkArgsOptionHandler struct {
	IsTraining func(bool) FwdBenchmarkArgsOptionFunc
}

var FwdBenchmarkArgsOption = fwdBenchmarkArgsOptionHandler{
	IsTraining: func(isTraining bool) FwdBenchmarkArgsOptionFunc {
		return func(o *fwdBenchmarkArgsOptions) {
			o.IsTraining = true
		}
	},
}

func CreateFwdBenchmarkArgsOption(os ...FwdBenchmarkArgsOptionFunc) *fwdBenchmarkArgsOptions {
	opts := &fwdBenchmarkArgsOptions{
		IsTraining: false,
	}
	for _, o := range os {
		o(opts)
	}
	return opts
}
