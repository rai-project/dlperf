package benchmark

import (
	"hash/fnv"
	"regexp"
	"strings"

	"github.com/pkg/errors"
	"github.com/spf13/cast"
)

type MetricInfo struct {
	Name             string
	KernelName       string
	CurrentIteration int
	Value            uint64
}

type KernelInfo struct {
	Name        string              `json:"name,omitempty"`
	MangledName string              `json:"mangled_name,omitempty"`
	NameHash    uint64              `json:"-"`
	Metrics     map[string][]uint64 `json:"metrics,omitempty"`
}

type KernelInfos []KernelInfo

func fnvOfString(b string) uint64 {
	hash := fnv.New64a()
	hash.Write([]byte(b))
	return hash.Sum64()
}

func getMetricInfo(str string, v interface{}) (*MetricInfo, error) {
	var re = regexp.MustCompile(`(?m)kernel_metric:(.*)\/current_iter:(.*)\/metric:(.*)`)
	allMatches := re.FindAllStringSubmatch(str, -1)
	if len(allMatches) != 1 {
		return nil, errors.Errorf("cannot find any metric matches in %s", str)
	}
	matches := allMatches[0]
	if len(matches) != 4 {
		return nil, errors.Errorf("unexpected number of metrics matches in %s", str)
	}
	kernelName, currentIter, metricName := matches[1], matches[2], matches[3]
	return &MetricInfo{
		Name:             metricName,
		KernelName:       kernelName,
		CurrentIteration: cast.ToInt(currentIter),
		Value:            cast.ToUint64(v),
	}, nil
}

func getBenchmarkKernelInfo(attrs map[string]interface{}, kernelName string) KernelInfo {
	nameHash := fnvOfString(kernelName)

	res := KernelInfo{
		Name:        demangleName(kernelName),
		MangledName: kernelName,
		NameHash:    nameHash,
		Metrics:     map[string][]uint64{},
	}

	for k, v := range attrs {
		// get demangled name
		if strings.HasPrefix(k, "demangled_kernel:") {
			hash := cast.ToUint64(v)
			if hash == nameHash {
				res.MangledName = strings.TrimPrefix(k, "demangled_kernel:")
			}
			continue
		}

		if strings.HasPrefix(k, "kernel_metric:") {
			metricInfo, err := getMetricInfo(k, v)
			if err != nil {
				log.WithError(err).WithField("metric", k).Warn("unable to parse metric")
				continue
			}
			if _, ok := res.Metrics[metricInfo.Name]; ok {
				// has metric we just need to append
				res.Metrics[metricInfo.Name] = append(res.Metrics[metricInfo.Name], metricInfo.Value)
				continue
			}
			// no metric, we need to create it
			res.Metrics[metricInfo.Name] = []uint64{metricInfo.Value}
		}
	}
	return res
}

func getBenchmarkKernelInfos(attrs map[string]interface{}) (KernelInfos, error) {
	if _, err := getAttributeStringByPrefix(attrs, "cupti_enabled"); err != nil {
		return nil, errors.New("benchmark was not run with cupti profile information")
	}

	visitedKernels := map[string]bool{}

	keyExists := func(s string) bool {
		_, ok := visitedKernels[s]
		return ok
	}

	infos := []KernelInfo{}

	for k, _ := range attrs {
		if strings.HasPrefix(k, "kernel:") && !keyExists(k) {
			infos = append(infos, getBenchmarkKernelInfo(attrs, strings.TrimPrefix(k, "kernel:")))
			visitedKernels[k] = true
		}
	}
	return infos, nil
}
