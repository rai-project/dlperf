package benchmark

import (
	"encoding/json"
	"time"

	"github.com/fatih/structs"
	"github.com/linkosmos/mapop"
)

type Machine struct {
}

type Context struct {
	Date              string  `json:"date"`
	NumCpus           int     `json:"num_cpus"`
	MhzPerCPU         int     `json:"mhz_per_cpu"`
	CPUScalingEnabled bool    `json:"cpu_scaling_enabled"`
	LibraryBuildType  string  `json:"library_build_type"`
	Machine           Machine `json:"machine,omitempty"`
}

type Suite struct {
	Context    Context    `json:"context"`
	Benchmarks Benchmarks `json:"benchmarks"`
}

type Benchmarks []Benchmark

type Benchmark struct {
	Name       string                 `json:"name"`
	Iterations float64                `json:"iterations"`
	RealTime   time.Duration          `json:"real_time"`
	CPUTime    time.Duration          `json:"cpu_time"`
	TimeUnit   string                 `json:"time_unit"`
	Attributes map[string]interface{} `json:"-"`
}

func (s *Suite) Merge(other Suite) {
	s.Benchmarks = s.Benchmarks.Merge(other.Benchmarks)
}

func (w *Benchmark) UnmarshalJSON(data []byte) error {
	elems := map[string]interface{}{}
	err := json.Unmarshal(data, &elems)
	if err != nil {
		return err
	}

	jsonTagMap := map[string]string{}
	for _, field := range structs.New(Benchmark{}).Fields() {
		fieldName := field.Name()
		jsonTagMap[field.Tag("json")] = fieldName
	}

	w.Attributes = map[string]interface{}{}

	st := structs.New(w)
	for k, v := range elems {
		if f, ok := st.FieldOk(k); ok {
			f.Set(v)
			continue
		}
		if jk, ok := jsonTagMap[k]; ok {
			if f, ok := st.FieldOk(jk); ok {
				f.Set(v)
				continue
			}
		}
		w.Attributes[k] = v
	}

	return nil
}

func (w Benchmark) MarshalJSON() ([]byte, error) {
	m := structs.Map(w)
	attrs, ok := m["Attributes"].(map[string]interface{})
	if ok {
		m = mapop.Reject(m, "Attributes")
		m = mapop.Merge(m, attrs)
	}
	return json.Marshal(m)
}
