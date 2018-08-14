package benchmark

import (
	"encoding/json"
	"testing"

	"github.com/k0kubun/pp"
	"github.com/stretchr/testify/assert"
)

func TestUnmarshal(t *testing.T) {
	benchmarkData := `
  {
    "context": {
      "date": "2017-11-30 18:51:39",
      "num_cpus": 20,
      "mhz_per_cpu": 4023,
      "cpu_scaling_enabled": false,
      "library_build_type": "release"
    },
    "benchmarks": [
      {
        "name": "SGEMM/1000/1/1",
        "iterations": 88881,
        "real_time": 7.8830189572615145e+03,
        "cpu_time": 7.8850406273556928e+03,
        "time_unit": "ns",
        "K": 1.0000000000000000e+00,
        "M": 1.0000000000000000e+03,
        "N": 1.0000000000000000e+00
      },
      {
        "name": "SGEMM/128/169/1728",
        "iterations": 878,
        "real_time": 7.9884078360094829e+05,
        "cpu_time": 7.9861304783598939e+05,
        "time_unit": "ns",
        "K": 1.7280000000000000e+03,
        "M": 1.2800000000000000e+02,
        "N": 1.6900000000000000e+02
      }
    ]
  }`
	bench := &Suite{}
	err := json.Unmarshal([]byte(benchmarkData), bench)
	assert.NoError(t, err)
	assert.NotEmpty(t, bench)
	assert.NotEmpty(t, bench.Benchmarks)
	pp.Println(bench)
}
