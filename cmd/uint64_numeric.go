package cmd

import (
	"sort"
)

func trimmedMeanUint64Slice(data []uint64, frac float64) uint64 {

	// Sum returns the sum of the elements of the slice.
	total := func(s []uint64) uint64 {
		var sum uint64
		for _, val := range s {
			sum += val
		}
		return sum
	}

	// Mean computes the weighted mean of the data set.
	//  sum_i {w_i * x_i} / sum_i {w_i}
	// If weights is nil then all of the weights are 1. If weights is not nil, then
	// len(x) must equal len(weights).
	mean := func(x, weights []uint64) uint64 {
		if weights == nil {
			return total(x) / uint64(len(x))
		}
		if len(x) != len(weights) {
			panic("stat: slice length mismatch")
		}
		var (
			sumValues  uint64
			sumWeights uint64
		)
		for i, w := range weights {
			sumValues += w * x[i]
			sumWeights += w
		}
		return sumValues / sumWeights
	}

	if frac == 0 {
		frac = DefaultTrimmedMeanFraction
	}
	if len(data) == 0 {
		return 0
	}
	if len(data) < 3 {
		return mean(data, nil)
	}
	if len(data) == 3 {
		sortUint64s(data)
		return data[1]
	}

	cnt := len(data)

	sortUint64s(data)

	start := maxInt(0, floor(float64(cnt)*frac))
	end := minInt(cnt-1, cnt-floor(float64(cnt)*frac))

	// pp.Println("start = ", start, "   end = ", end)
	trimmed := data[start:end]

	ret := mean(trimmed, nil)

	return ret
}

type uint64Slice []uint64

func (p uint64Slice) Len() int           { return len(p) }
func (p uint64Slice) Less(i, j int) bool { return p[i] < p[j] }
func (p uint64Slice) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

func sortUint64s(a []uint64) { sort.Sort(uint64Slice(a)) }
