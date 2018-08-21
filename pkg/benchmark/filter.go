package benchmark

import (
	"regexp"

	"github.com/google/go-cmp/cmp"
	"github.com/spf13/cast"
)

func (bs Benchmarks) Merge(other Benchmarks) Benchmarks {
	return append(bs, other...)
}

func (s Suite) FilterByName(rx string) (Benchmarks, error) {
	return s.Benchmarks.FilterByName(rx)
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

func (s Suite) Filter(filter Benchmark) (Benchmarks, error) {
	return s.Benchmarks.Filter(filter)
}

func isSameScalar(a, b interface{}) bool {
	if cmp.Equal(a, b) {
		return true
	}

	a0, err := cast.ToFloat64E(a)
	if err != nil {
		return false
	}
	if a0 < 0 {
		// panic("a0 < 0")
		a0 = float64(0)
	}

	b0, err := cast.ToFloat64E(b)
	if err != nil {
		return false
	}
	if b0 < 0 {
		// panic("b0 < 0")
		b0 = float64(0)
	}

	floatEquals := func(a, b float64) bool {
		const EPSILON float64 = 0.0001
		return (a-b) < EPSILON && (b-a) < EPSILON
	}
	return floatEquals(a0, b0)
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
		toAdd := true
		for k, filterVal := range filter.Attributes {
			val, ok := b.Attributes[k]
			if !ok {
				toAdd = false
				break
			}
			if !isSameScalar(filterVal, val) {
				toAdd = false
				break
			}
		}
		if toAdd {
			benches = append(benches, b)
		}
	}

	return benches, nil
}

func (b Benchmark) IsEqual(other Benchmark) bool {
	if b.Name != other.Name {
		return false
	}

	if b.Iterations != other.Iterations {
		return false
	}
	if b.RealTime != other.RealTime {
		return false
	}
	if b.CPUTime != other.CPUTime {
		return false
	}
	if b.TimeUnit != other.TimeUnit {
		return false
	}
	for k, filterVal := range other.Attributes {
		val, ok := b.Attributes[k]
		if !ok {
			return false
		}
		if !isSameScalar(filterVal, val) {
			return false
		}
	}

	return true
}
