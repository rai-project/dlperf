package benchmark

import (
	"encoding/json"
	"sort"
	"time"

	"github.com/fatih/structs"
	"github.com/k0kubun/pp"
	"github.com/linkosmos/mapop"
	nvidiasmi "github.com/rai-project/nvidia-smi"
	"github.com/spf13/cast"
)

type Machine struct {
	Architecture    string            `json:"architecture"`
	Hostname        string            `json:"hostname"`
	GPUArchitecture string            `json:"gpu_architecture"`
	Attributes      map[string]string `json:"attributes"`
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
	Context        Context              `json:"context"`
	Benchmarks     Benchmarks           `json:"benchmarks"`
	GPUInformation *nvidiasmi.NvidiaSmi `json:"gpu_information,omitempty"`
}

type Benchmarks []Benchmark

type Benchmark struct {
	Name               string                 `json:"name,omitempty"`
	Iterations         float64                `json:"iterations,omitempty"`
	RealTime           time.Duration          `json:"real_time,omitempty"`
	CPUTime            time.Duration          `json:"cpu_time,omitempty"`
	TimeUnit           string                 `json:"time_unit,omitempty"`
	Flops              *float64               `json:"predicted_flops_count,omitempty"`
	CUDADriverVersion  string                 `json:"cuda_driver_version,omitempty"`
	CUDARuntimeVersion string                 `json:"cuda_runtime_version,omitempty"`
	GPUName            string                 `json:"gpu_name,omitempty"`
	HostName           string                 `json:"host_name,omitempty"`
	CUPTIVersion       string                 `json:"cupti_version,omitempty"`
	ComputeCapability  string                 `json:"compute_capability,omitempty"`
	CUBLASVersion      string                 `json:"cublas_version,omitempty"`
	CUDNNVersion       string                 `json:"cudnn_version,omitempty"`
	KernelInfos        KernelInfos            `json:"kernel_infos,omitempty"`
	Attributes         map[string]interface{} `json:"-,omitempty"`
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

	delete(elems, "batch_size") // we do not care about batchsize in filter

	if e, ok := elems["error_occurred"]; ok {
		if eok, ok := e.(bool); ok && eok {
			w = &Benchmark{}
			return nil
			// return errors.New(elems["error_message"].(string))
		}
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
		if false {
			pp.Println(k, "  ", jsonTagMap[k], "  ", elems[k])
		}
		if jk, ok := jsonTagMap[k]; ok {
			if f, ok := st.FieldOk(jk); ok {
				f.Set(v)
				continue
			}
		}

		if f, ok := elems["predicted_flops_count"]; ok {
			val := cast.ToFloat64(f)
			w.Flops = &val
		}

		w.Attributes[k] = v
	}

	delete(w.Attributes, "batch_size") // we do not care about batchsize in filter

	// if strings.HasPrefix(w.Name, "LAYER_CUDNN_CONV_FWD_FLOAT<CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM>/W:14/H:14/C:5") {
	// 	pp.Println(elems)
	// }

	if e, err := cast.ToDurationE(elems["real_time"]); err == nil {
		w.RealTime = e
	}

	if e, err := cast.ToDurationE(elems["cpu_time"]); err == nil {
		w.CPUTime = e
	}

	if ver, err := getBenchmarkCUDADriverVersion(elems); err != nil {
		w.CUDADriverVersion = ver
	}
	if ver, err := getBenchmarkCUDARuntimeVersion(elems); err != nil {
		w.CUDARuntimeVersion = ver
	}
	if ver, err := getBenchmarkCUPTIVersion(elems); err != nil {
		w.CUPTIVersion = ver
	}
	if ver, err := getBenchmarkCUDNNVersion(elems); err != nil {
		w.CUDNNVersion = ver
	}
	if ver, err := getBenchmarkComputeCapability(elems); err != nil {
		w.ComputeCapability = ver
	}
	if ver, err := getBenchmarkCUBLASVersion(elems); err != nil {
		w.CUBLASVersion = ver
	}
	if ver, err := getBenchmarkKernelInfos(elems); err != nil {
		w.KernelInfos = ver
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
	// w.RealTime = w.RealTime / time.Microsecond
	// w.CPUTime = w.CPUTime / time.Microsecond
	return json.Marshal(m)
}

func (b Benchmarks) Sort() {
	sort.Sort(b)
}
