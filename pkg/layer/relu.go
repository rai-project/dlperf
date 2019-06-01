package layer

import (
	"strings"

	"github.com/mitchellh/hashstructure"
	dlperf "github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/dlperf/pkg/benchmark"
)

//easyjson:json
type Relu struct {
	*Base `json:",inline,flatten,omitempty"`
}

func (Relu) OperatorType() string {
	return "Relu"
}

func (Relu) Description() string {
	return ``
}

func (c *Relu) InferShape(inputLayers dlperf.Layers) {
	inputShapes := getOutputShapes(inputLayers)
	c.SetInputShapes(inputShapes)
	c.SetOutputShapes(inputShapes)
}

func (c Relu) FwdBenchmarkName(opts ...dlperf.FwdBenchmarkArgsOptionFunc) string {
	return "LAYER_CUDNN_ACTIVATION_FWD"
}

func (c Relu) BwdBenchmarkName(opts ...dlperf.BwdBenchmarkArgsOptionFunc) string {
	return "LAYER_CUDNN_ACTIVATION_BWD"
}

func (c Relu) FwdCUDNNName() string {
	return ""
}

func (c Relu) BwdCUDNNName() string {
	return ""
}

func (c Relu) FwdTiming(system string /* hardware/software struct */) string {
	return ""
}

func (c Relu) BwdTiming(system string /* hardware/software struct */) string {
	return ""
}

func (c Relu) FwdBenchmarkAlgorithms(...dlperf.FwdBenchmarkArgsOptionFunc) []string {
	return c.BenchmarkAlgorithms()
}

func (c Relu) BwdBenchmarkAlgorithms(...dlperf.BwdBenchmarkArgsOptionFunc) []string {
	return c.BenchmarkAlgorithms()
}

func (c Relu) BenchmarkAlgorithms() []string {
	switch strings.ToLower(c.OnnxOperatorType()) {
	case "sigmoid":
		return []string{
			"CUDNN_ACTIVATION_SIGMOID",
		}
	case "tanh":
		return []string{
			"CUDNN_ACTIVATION_TANH",
		}
	case "relu":
		return []string{
			"CUDNN_ACTIVATION_RELU",
			"CUDNN_ACTIVATION_CLIPPED_RELU",
		}
	case "elu":
		return []string{
			"CUDNN_ACTIVATION_ELU",
		}
	case "prelu": // NOT IN CUDNN
		return []string{
			"ACTIVATION_PRELU",
		}
	case "leakyrelu": // NOT IN CUDNN
		return []string{
			"ACTIVATION_LEAKY_RELU",
		}
	case "identity":
		return []string{
			"CUDNN_ACTIVATION_IDENTITY",
		}
	}

	panic("invalid relu operator = " + c.OnnxOperatorType())

	return nil
}

//easyjson:json
type reluBenchmarkArgs struct {
	BaseBenchmarkArgs      `json:",inline,flatten,omitempty"`
	BaseBenchmarkInputArgs `json:",inline,flatten,omitempty"`
	BatchSize              int64 `json:"batch_size,omitempty"`
}

func (c Relu) substituteAlgorithm(alg string) string {
	// substitution because cudnn does not support certain algorithms
	switch strings.ToUpper(alg) {
	case "ACTIVATION_PRELU", "ACTIVATION_LEAKY_RELU":
		return "CUDNN_ACTIVATION_RELU"
	}
	return alg
}

func (c Relu) FwdBenchmarkArgs(opts ...dlperf.FwdBenchmarkArgsOptionFunc) interface{} {
	res := reluBenchmarkArgs{
		BaseBenchmarkInputArgs: mkBaseBenchmarkInputArgs(&c),
		BaseBenchmarkArgs:      mkBaseBenchmarkFWDArgs(&c, opts...),
		BatchSize:              dlperf.GetBatchSize(),
	}

	// substitution because cudnn does not support certain algorithms
	for ii, alg := range res.Algorithms {
		res.Algorithms[ii] = c.substituteAlgorithm(alg)
	}

	hash, err := hashstructure.Hash(
		res,
		&hashstructure.HashOptions{
			TagName: "hash",
		},
	)
	if err != nil {
		log.Fatal(err)
	}
	res.UniqueBenchmarkID = hash

	return res
}

func (c Relu) BwdBenchmarkArgs(opts ...dlperf.BwdBenchmarkArgsOptionFunc) interface{} {
	res := reluBenchmarkArgs{
		BaseBenchmarkInputArgs: mkBaseBenchmarkInputArgs(&c),
		BaseBenchmarkArgs:      mkBaseBenchmarkBWDArgs(&c, opts...),
		BatchSize:              dlperf.GetBatchSize(),
	}

	// substitution because cudnn does not support certain algorithms
	for ii, alg := range res.Algorithms {
		res.Algorithms[ii] = c.substituteAlgorithm(alg)
	}

	hash, err := hashstructure.Hash(
		res,
		&hashstructure.HashOptions{
			TagName: "hash",
		},
	)
	if err != nil {
		log.Fatal(err)
	}
	res.UniqueBenchmarkID = hash

	return res
}

func (c Relu) FwdBenchmarkFilter(datatype, algorithm string, opts ...dlperf.FwdBenchmarkArgsOptionFunc) benchmark.Benchmark {
	return benchmark.Benchmark{
		Name:       mkFwdBenchmarkFilterName(&c, datatype, algorithm, opts...),
		Attributes: benchmarkAttributes(c.FwdBenchmarkArgs()),
	}
}

func (c Relu) BwdBenchmarkFilter(datatype, algorithm string, opts ...dlperf.BwdBenchmarkArgsOptionFunc) benchmark.Benchmark {
	return benchmark.Benchmark{
		Name:       mkBwdBenchmarkFilterName(&c, datatype, algorithm, opts...),
		Attributes: benchmarkAttributes(c.FwdBenchmarkArgs()),
	}
}

func (c Relu) FwdBenchmarkGenerator() string {
	templString := _escFSMustString(false, "/scope/relu.tmpl")

	return templateExecFWD(&c, templateBasePrefix+templString+templateBaseSuffix)
}

func (c Relu) BwdBenchmarkGenerator() string {
	templString := _escFSMustString(false, "/scope/relu.tmpl")

	return templateExecBWD(&c, templateBasePrefix+templString+templateBaseSuffix)
}

func (c Relu) FwdBenchmarkGeneratorArgNames() []string {
	return benchmarkArgNames(reluBenchmarkArgs{})
}

func (c Relu) BwdBenchmarkGeneratorArgNames() []string {
	return benchmarkArgNames(reluBenchmarkArgs{})
}

func (c Relu) Shape() dlperf.ShapeInformation {
	return c.Information().Shape()
}

func (c Relu) Information() dlperf.LayerInformation {
	info := &Information{
		Base: c.Base,
		shape: dlperf.ShapeInformation{
			InputShapes:  c.InputShapes(),
			OutputShapes: c.OutputShapes(),
		},
	}

	if isAnyEmpty(c.OutputShapes()) {
		log.WithField("layer", c.OperatorType()).Info("len(OutputShapes) is 0")
		return info
	}

	checkNumber(c.InputShapes, []int{1, 2}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputShapes, []int{1}, c.OperatorType(), "number of outputs")

	inputShapes := c.InputShapes()[0]

	numOps := int64(1)
	for _, s := range inputShapes {
		numOps *= s
	}

	info.flops = dlperf.FlopsInformation{
		Comparisons: numOps,
	}

	return info
}

func init() {
	dlperf.Register(&Relu{})
}
