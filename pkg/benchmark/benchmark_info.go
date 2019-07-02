package benchmark

import (
	"errors"
	"strings"
)

type selectorFunction func(s string) bool

func getAttributeString(attrs map[string]interface{}, selector selectorFunction) (string, error) {
	for k, _ := range attrs {
		if selector(k) {
			return k, nil
		}
	}
	return "", errors.New("cannot find benchmark attribute")
}

func getAttributeStringByPrefix(attrs map[string]interface{}, prefix string) (string, error) {
	for k, _ := range attrs {
		if strings.HasPrefix(k, prefix) {
			s := strings.TrimLeft(k, prefix)
			return s, nil
		}

		if strings.HasPrefix(k, prefix+":") {
			s := strings.TrimLeft(k, prefix+":")
			return s, nil
		}
	}
	return "", errors.New("cannot find " + strings.Replace(strings.TrimSuffix(prefix, ":"), "_", " ", -1))
}

func getBenchmarkFunctionName(attrs map[string]interface{}) (string, error) {
	return getAttributeStringByPrefix(attrs, "benchmark_func:")
}

func getBenchmarkCUDADriverVersion(attrs map[string]interface{}) (string, error) {
	return getAttributeStringByPrefix(attrs, "cuda_driver_version:")
}

func getBenchmarkCUDARuntimeVersion(attrs map[string]interface{}) (string, error) {
	return getAttributeStringByPrefix(attrs, "cuda_runtime_version:")
}

func getBenchmarkCUBLASVersion(attrs map[string]interface{}) (string, error) {
	return getAttributeStringByPrefix(attrs, "cublas_version:")
}

func getBenchmarkCUPTIVersion(attrs map[string]interface{}) (string, error) {
	return getAttributeStringByPrefix(attrs, "cupti_version:")
}

func getBenchmarkCUDNNVersion(attrs map[string]interface{}) (string, error) {
	return getAttributeStringByPrefix(attrs, "cudnn_version:")
}

func getBenchmarkComputeCapability(attrs map[string]interface{}) (string, error) {
	return getAttributeStringByPrefix(attrs, "compute_capability:")
}

func getBenchmarkGPUName(attrs map[string]interface{}) (string, error) {
	return getAttributeStringByPrefix(attrs, "gpu_name:")
}

func getBenchmarkHostName(attrs map[string]interface{}) (string, error) {
	return getAttributeStringByPrefix(attrs, "host_name:")
}

func getBenchmarkNumIterations(attrs map[string]interface{}) (string, error) {
	return getAttributeStringByPrefix(attrs, "num_iterations")
}
