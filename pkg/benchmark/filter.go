package benchmark

import (
	"fmt"
	"reflect"
	"regexp"

	"github.com/k0kubun/pp"
	"github.com/spf13/cast"
)

func (bs Benchmarks) Len() int             { return len(bs) }
func (bs Benchmarks) Less(ii, jj int) bool { return bs[ii].RealTime < bs[jj].RealTime }
func (p Benchmarks) Swap(i, j int)         { p[i], p[j] = p[j], p[i] }

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
		fmt.Println(err)
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

func (bs Benchmarks) Filter(filter Benchmark) (Benchmarks, error) {
	benches := []Benchmark{}

	if filter.Name != "" {
		var err error
		bs, err = bs.FilterByName(filter.Name)
		if err != nil {
			return benches, err
		}
	}

	delete(filter.Attributes, "batch_size")    // we do not care about batchsize in filter
	delete(filter.Attributes, "conv_bwd_type") // we do not care about conv_bwd_type since it's encoded in name

next:
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
				continue next
			}

			if k == "conv_bwd_type" || k == "conv_fwd_type" {
				filterVal = int(reflect.ValueOf(filterVal).Int())
				val = cast.ToInt(val)
			}

			if !isSameScalar(filterVal, val) {
				continue next
			}
			if false {
				if k == "conv_bwd_type" {
					pp.Println(filterVal)
					pp.Println(val)
				}
			}
		}
		benches = append(benches, b)
	}

	res := Benchmarks(benches)

	res.Sort()

	return res, nil
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
	if len(b.Attributes) != len(other.Attributes) {
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
