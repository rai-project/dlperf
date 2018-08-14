package benchmark

import (
	"regexp"

	"github.com/google/go-cmp/cmp"
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
			if !cmp.Equal(filterVal, val) {
				continue
			}
		}
		benches = append(benches, b)
	}

	return benches, nil
}
