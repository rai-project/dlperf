package writer

import "strings"

type Options struct {
	Format                  string
	ShowBenchmarkName       bool
	ShowMetrics             bool
	ShowSummary             bool
	ShowKernelName          bool
	ShowLayerName           bool
	ShowHumanFlops          bool
	TrimBenchmarkName       bool
	TrimLayerName           bool
	ShowMangledKernelName   bool
	ShowKernelNamesOnly     bool
	ShowFlopsMetricsOnly    bool
	AggregateFlopsMetrics   bool
	HideEmptyMetrics        bool
	TheoreticalFlopsFMAOnly bool
	MetricsFilter           []string
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

func TrimLayerName(b bool) Option {
	return func(w *Options) {
		w.TrimLayerName = b
	}
}

func ShowMangledKernelName(b bool) Option {
	return func(w *Options) {
		w.ShowMangledKernelName = b
	}
}

func ShowFlopsMetricsOnly(b bool) Option {
	return func(w *Options) {
		w.ShowFlopsMetricsOnly = b
	}
}

func ShowKernelNamesOnly(b bool) Option {
	return func(w *Options) {
		w.ShowKernelNamesOnly = b
	}
}

func AggregateFlopsMetrics(b bool) Option {
	return func(w *Options) {
		w.AggregateFlopsMetrics = b
	}
}

func HideEmptyMetrics(b bool) Option {
	return func(w *Options) {
		w.HideEmptyMetrics = b
	}
}

func MetricsFilter(s []string) Option {
	if s == nil {
		s = []string{}
	}
	for ii, e := range s {
		s[ii] = strings.TrimSpace(strings.ToLower(e))
	}
	return func(w *Options) {
		w.MetricsFilter = s
	}
}

func TheoreticalFlopsFMAOnly(b bool) Option {
	return func(w *Options) {
		w.TheoreticalFlopsFMAOnly = b
	}
}

func NewOptions(opts ...Option) Options {
	res := &Options{
		Format:                  "table",
		ShowBenchmarkName:       true,
		ShowKernelName:          true,
		ShowLayerName:           true,
		ShowMetrics:             true,
		ShowSummary:             true,
		ShowHumanFlops:          true,
		TrimBenchmarkName:       true,
		TrimLayerName:           true,
		ShowMangledKernelName:   true,
		ShowFlopsMetricsOnly:    false,
		AggregateFlopsMetrics:   false,
		ShowKernelNamesOnly:     false,
		HideEmptyMetrics:        true,
		TheoreticalFlopsFMAOnly: false,
		MetricsFilter:           []string{},
	}

	for _, o := range opts {
		o(res)
	}

	return *res
}
