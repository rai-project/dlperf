package onnx

import (
	"testing"

	"github.com/k0kubun/pp"
)

func subsequences(seq []string, length int) [][]string {
	subsetsLength := len(seq) - length + 1
	result := make([][]string, subsetsLength)
	for ii := 0; ii < subsetsLength; ii++ {
		result[ii] = seq[ii : ii+length]
	}
	return result
}

func TestSubsequencesAlgorithm(t *testing.T) {

	seq := []string{"a", "b", "c", "d", "e", "f"}
	subseqs := subsequences(seq, 3)
	pp.Println(subseqs)
}
