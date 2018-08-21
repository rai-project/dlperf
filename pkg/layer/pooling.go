package layer

import (
	"math"
	"strings"

	"github.com/mitchellh/hashstructure"
	"github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/dlperf/pkg/benchmark"
)

type Pooling struct {
	*Base       `json:",inline,flatten,omitempty"`
	KernelShape dlperf.Shape `json:"kernel_shape,omitempty"`
	Pads        dlperf.Shape `json:"pads,omitempty"`
	Strides     dlperf.Shape `json:"strides,omitempty"`
}

func (Pooling) OperatorType() string {
	return "Pooling"
}

func (Pooling) Description() string {
	return ``
}

func (c *Pooling) InferShape(inputLayers dlperf.Layers) {
	inputShapes := getOutputShapes(inputLayers)
	xShape := c.InputShapes()[0]

	yShape := dlperf.Shape{xShape[0], xShape[1]}
	for ii, xs := range xShape[2:] {
		ys := int64(math.Floor(float64(xs+c.Pads[ii]+c.Pads[ii+len(xShape)-2]-c.KernelShape[ii])/float64(c.Strides[ii])) + 1)
		yShape = append(yShape, ys)
	}

	c.SetInputShapes(inputShapes)
	c.SetOutputShapes([]dlperf.Shape{yShape})
}

func (c Pooling) FwdBenchmarkName() string {
	return "LAYER_CUDNN_POOLING_FWD"
}

func (c Pooling) FwdCUDNNName() string {
	return ""
}

func (c Pooling) FwdTiming(system string /* hardware/software struct */) string {
	return ""
}

func (c Pooling) FwdBenchmarkAlgorithms() []string {
	switch strings.ToLower(c.onnxOperatorType) {
	case "maxpool":
		return []string{
			"CUDNN_POOLING_MAX",
			"CUDNN_POOLING_MAX_DETERMINISTIC",
		}
	case "averagepool":
		return []string{
			"CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING",
			"CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING",
		}
	}
	panic("invalid pooling operator " + c.onnxOperatorType)

	return nil
}

type poolingBenchmarkArgs struct {
	baseBenchmarkArgs
	BaseBenchmarkInputArgs
}

func (c Pooling) FwdBenchmarkGeneratorArgNames() []string {
	return benchmarkArgNames(poolingBenchmarkArgs{})
}

func (c Pooling) FwdBenchmarkArgs() interface{} {
	res := poolingBenchmarkArgs{
		BaseBenchmarkInputArgs: mkBaseBenchmarkInputArgs(&c),
		baseBenchmarkArgs:      mkBaseBenchmarkArgs(&c),
	}

	hash, err := hashstructure.Hash(
		res,
		&hashstructure.HashOptions{
			TagName: "args",
		},
	)
	if err != nil {
		panic(err)
	}
	res.UniqueBenchmarkID = hash

	return res
}

func (c Pooling) FwdBenchmarkFilter(datatype, algorithm string) benchmark.Benchmark {
	if algorithm == "" {
		algorithm = c.FwdBenchmarkAlgorithms()[0]
	}
	return benchmark.Benchmark{
		Name:       mkBenchmarkFilterName(&c, datatype, algorithm),
		Attributes: benchmarkAttributes(c.FwdBenchmarkArgs()),
	}
}

func (c Pooling) FwdBenchmarkGenerator() string {
	const templString = `
  [[ range $datatype := .DataTypes ]]
  template <cudnnActivationMode_t activation_mode>
  static void [[ $.BenchmarkName ]]_[[ $datatype.Name | upper ]]__[[$.UniqueBenchmarkID]](benchmark::State& state) {
    [[ $.BenchmarkName ]]_Impl<[[ $datatype.CType ]], activation_mode>(state);
    BENCHMARK_[[ $.BenchmarkName ]]_ADD_COUNTERS__[[$.UniqueBenchmarkID]](state);
  }
  [[ end ]]
`

	return templateExec(&c, templateBasePrefix+templString+templateBaseSuffix)
}

func (c Pooling) Shape() dlperf.ShapeInformation {
	return c.Information().Shape()
}

func (c Pooling) Information() dlperf.LayerInformation {
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

	checkNumber(c.InputShapes, []int{1}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputShapes, []int{1}, c.OperatorType(), "number of outputs")

	outputShape := c.OutputShapes()[0] // (N x C x ...)

	nOut := outputShape[0]
	cOut := outputShape[1]
	hOut := outputShape[2]
	wOut := outputShape[3]

	flops := dlperf.FlopsInformation{}
	switch c.onnxOperatorType {
	case "maxpool":
		flops.Comparisons = hOut * wOut * nOut * cOut * c.KernelShape[0] * c.KernelShape[1]
	case "averagepool":
		flops.Additions = hOut * wOut * nOut * cOut * c.KernelShape[0] * c.KernelShape[1]
	}

	info.flops = flops

	return info
}

func init() {
	dlperf.Register(&Pooling{})
}
