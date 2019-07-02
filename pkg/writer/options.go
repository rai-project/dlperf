package writer

type Options struct {
	Format                string
	ShowBenchmarkName     bool
	ShowMetrics           bool
	ShowSummary           bool
	ShowKernelName        bool
	ShowLayerName         bool
	ShowHumanFlops        bool
	TrimBenchmarkName     bool
	TrimLayerkName        bool
	ShowMangledKernelName bool
}

type Option func(*Options)

func ShowBenchmarkName(b bool) Option {
	return func(w *Options) {
		w.ShowBenchmarkName = b
	}
}

func ShowKernelName(b bool) Option {
	return func(w *Options) {
		w.ShowKernelName = b
	}
}

func ShowLayerName(b bool) Option {
	return func(w *Options) {
		w.ShowLayerName = b
	}
}

func ShowMetrics(b bool) Option {
	return func(w *Options) {
		w.ShowMetrics = b
	}
}

func ShowSummary(b bool) Option {
	return func(w *Options) {
		w.ShowSummary = b
	}
}

func ShowHumanFlops(b bool) Option {
	return func(w *Options) {
		w.ShowHumanFlops = b
	}
}

func Format(b string) Option {
	return func(w *Options) {
		w.Format = b
	}
}

func TrimBenchmarkName(b bool) Option {
	return func(w *Options) {
		w.TrimBenchmarkName = b
	}
}

func TrimLayerkName(b bool) Option {
	return func(w *Options) {
		w.TrimLayerkName = b
	}
}

func ShowMangledKernelName(b bool) Option {
	return func(w *Options) {
		w.ShowMangledKernelName = b
	}
}

func NewOptions(opts ...Option) Options {
	res := &Options{
		Format:                "table",
		ShowBenchmarkName:     true,
		ShowKernelName:        true,
		ShowLayerName:         true,
		ShowMetrics:           true,
		ShowSummary:           true,
		ShowHumanFlops:        true,
		TrimBenchmarkName:     true,
		TrimLayerkName:        true,
		ShowMangledKernelName: true,
	}

	for _, o := range opts {
		o(res)
	}

	return *res
}
