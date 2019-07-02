package cmd

import (
	"math"
	"sort"
)

var (
	DefaultTrimmedMeanFraction = 0.2
)

func trimmedMean(data []float64, frac float64) float64 {

	// Sum returns the sum of the elements of the slice.
	total := func(s []float64) float64 {
		var sum float64
		for _, val := range s {
			sum += val
		}
		return sum
	}

	// Mean computes the weighted mean of the data set.
	//  sum_i {w_i * x_i} / sum_i {w_i}
	// If weights is nil then all of the weights are 1. If weights is not nil, then
	// len(x) must equal len(weights).
	mean := func(x, weights []float64) float64 {
		if weights == nil {
			return total(x) / float64(len(x))
		}
		if len(x) != len(weights) {
			panic("stat: slice length mismatch")
		}
		var (
			sumValues  float64
			sumWeights float64
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
		sort.Float64s(data)
		return data[1]
	}

	cnt := len(data)

	sort.Float64s(data)

	start := maxInt(0, floor(float64(cnt)*frac))
	end := minInt(cnt-1, cnt-floor(float64(cnt)*frac))

	// pp.Println("start = ", start, "   end = ", end)
	trimmed := data[start:end]

	ret := mean(trimmed, nil)

	return ret
}

func floor(x float64) int {
	return int(math.Floor(x))
}

func ceil(x float64) int {
	return int(math.Ceil(x))
}

func maxInt(x, y int) int {
	if x > y {
		return x
	}
	return y
}

func minInt(x, y int) int {
	if x > y {
		return y
	}
	return x
}
