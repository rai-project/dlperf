package benchmark

import (
	"regexp"

	"github.com/google/go-cmp/cmp"
	"github.com/spf13/cast"
)

func (bs Benchmarks) Merge(other Benchmarks) Benchmarks {
	return append(bs, other...)
}

func (bs Benchmarks) FilterByName(rx string) (Benchmarks, error) {
	benches := []Benchmark{}

	regex, err := regexp.Compile(rx)
	if err != nil {
		return benches, err
	}

	for _, b := range bs {
		if regex.MatchString(b.Name) {
			benches = append(benches, b)
		}
	}

	return benches, nil
}

func (bs Benchmarks) Filter(filter Benchmark) (Benchmarks, error) {
	benches := []Benchmark{}

	isSame := func(a, b interface{}) bool {
		if cmp.Equal(a, b) {
			return true
		}

		a0, err := cast.ToFloat64E(a)
		if err != nil {
			return false
		}

		b0, err := cast.ToFloat64E(b)
		if err != nil {
			return false
		}

		floatEquals := func(a, b float64) bool {
			const EPSILON float64 = 0.0001
			return (a-b) < EPSILON && (b-a) < EPSILON
		}
		return floatEquals(a0, b0)
	}

	if filter.Name != "" {
		var err error
		bs, err = bs.FilterByName(filter.Name)
		if err != nil {
			return benches, err
		}
	}

	for _, b := range bs {
		if filter.Iterations != 0 && b.Iterations != filter.Iterations {
			continue
		}
		if filter.RealTime != 0 && b.RealTime != filter.RealTime {
			continue
		}
		if filter.CPUTime != 0 && b.CPUTime != filter.CPUTime {
			continue
		}
		if filter.TimeUnit != "" && b.TimeUnit != filter.TimeUnit {
			continue
		}
		for k, filterVal := range filter.Attributes {
			val, ok := b.Attributes[k]
			if !ok {
				continue
			}
			if !isSame(filterVal, val) {
				continue
			}
		}
		benches = append(benches, b)
	}

	return benches, nil
}
