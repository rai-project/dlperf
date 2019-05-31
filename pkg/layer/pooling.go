package layer

import (
	"math"
	"strings"

	"github.com/mitchellh/hashstructure"
	dlperf "github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/dlperf/pkg/benchmark"
)

//easyjson:json
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

func (c Pooling) FwdBenchmarkName(opts ...dlperf.FwdBenchmarkArgsOptionFunc) string {
	return "LAYER_CUDNN_POOLING_FWD"
}

func (c Pooling) BwdBenchmarkName(opts ...dlperf.BwdBenchmarkArgsOptionFunc) string {
	return "LAYER_CUDNN_POOLING_BWD"
}

func (c Pooling) FwdCUDNNName() string {
	return ""
}

func (c Pooling) BwdCUDNNName() string {
	return ""
}

func (c Pooling) FwdTiming(system string /* hardware/software struct */) string {
	return ""
}

func (c Pooling) BwdTiming(system string /* hardware/software struct */) string {
	return ""
}

func (c Pooling) FwdBenchmarkAlgorithms(...dlperf.FwdBenchmarkArgsOptionFunc) []string {
	return c.BenchmarkAlgorithms()
}

func (c Pooling) BwdBenchmarkAlgorithms(...dlperf.BwdBenchmarkArgsOptionFunc) []string {
	return c.BenchmarkAlgorithms()
}

func (c Pooling) BenchmarkAlgorithms() []string {
	switch strings.ToLower(c.OnnxOperatorType()) {
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
	panic("invalid pooling operator " + c.OnnxOperatorType())

	return nil
}

//easyjson:json
type poolingBenchmarkArgs struct {
	BaseBenchmarkArgs `json:",inline,flatten,omitempty"`
	Input0            int64 `args:"input[0]" hash:"input[0]" json:"input_0,omitempty"`
	Input1            int64 `args:"input[1]" hash:"input[1]" json:"input_1,omitempty"`
	Input2            int64 `args:"input[2]" hash:"input[2]" json:"input_2,omitempty"`
	Input3            int64 `args:"input[3]" hash:"input[3]" json:"input_3,omitempty"`
	FilterHeight      int64 `args:"filter_height" hash:"filter_height" json:"filter_height,omitempty"`
	FilterWidth       int64 `args:"filter_width" hash:"filter_width" json:"filter_width,omitempty"`
	PadHeight         int64 `args:"pad_height" hash:"pad_height" json:"pad_height,omitempty"`
	PadWidth          int64 `args:"pad_width" hash:"pad_width" json:"pad_width,omitempty"`
	StrideHeight      int64 `args:"stride_height" hash:"stride_height" json:"stride_height,omitempty"`
	StrideWidth       int64 `args:"stride_width" hash:"stride_width" json:"stride_width,omitempty"`
	BatchSize         int64 `args:"batch_size" hash:"batch_size" json:"batch_size,omitempty"`
}

func (c Pooling) FwdBenchmarkArgs(opts ...dlperf.FwdBenchmarkArgsOptionFunc) interface{} {
	inShapes := c.InputShapes()

	res := poolingBenchmarkArgs{
		Input0:            inShapes[0][0],
		Input1:            inShapes[0][1],
		Input2:            inShapes[0][2],
		Input3:            inShapes[0][3],
		FilterHeight:      c.KernelShape[0],
		FilterWidth:       c.KernelShape[1],
		PadHeight:         c.Pads[0],
		PadWidth:          c.Pads[2],
		StrideHeight:      c.Strides[0],
		StrideWidth:       c.Strides[1],
		BatchSize:         dlperf.GetBatchSize(),
		BaseBenchmarkArgs: mkBaseBenchmarkFWDArgs(&c, opts...),
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

func (c Pooling) BwdBenchmarkArgs(opts ...dlperf.BwdBenchmarkArgsOptionFunc) interface{} {
	inShapes := c.InputShapes()

	res := poolingBenchmarkArgs{
		Input0:            inShapes[0][0],
		Input1:            inShapes[0][1],
		Input2:            inShapes[0][2],
		Input3:            inShapes[0][3],
		FilterHeight:      c.KernelShape[0],
		FilterWidth:       c.KernelShape[1],
		PadHeight:         c.Pads[0],
		PadWidth:          c.Pads[2],
		StrideHeight:      c.Strides[0],
		StrideWidth:       c.Strides[1],
		BaseBenchmarkArgs: mkBaseBenchmarkBWDArgs(&c, opts...),
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

func (c Pooling) FwdBenchmarkFilter(datatype, algorithm string, opts ...dlperf.FwdBenchmarkArgsOptionFunc) benchmark.Benchmark {
	return benchmark.Benchmark{
		Name:       mkFwdBenchmarkFilterName(&c, datatype, algorithm),
		Attributes: benchmarkAttributes(c.FwdBenchmarkArgs(opts...)),
	}
}

func (c Pooling) BwdBenchmarkFilter(datatype, algorithm string, opts ...dlperf.BwdBenchmarkArgsOptionFunc) benchmark.Benchmark {
	return benchmark.Benchmark{
		Name:       mkBwdBenchmarkFilterName(&c, datatype, algorithm),
		Attributes: benchmarkAttributes(c.BwdBenchmarkArgs(opts...)),
	}
}

func (c Pooling) FwdBenchmarkGenerator() string {
	templString := _escFSMustString(false, "/scope/pooling.tmpl")
	return templateExecFWD(&c, templateBasePrefix+templString+templateBaseSuffix)
}

func (c Pooling) BwdBenchmarkGenerator() string {
	templString := _escFSMustString(false, "/scope/pooling.tmpl")
	return templateExecBWD(&c, templateBasePrefix+templString+templateBaseSuffix)
}

func (c Pooling) FwdBenchmarkGeneratorArgNames() []string {
	return benchmarkArgNames(poolingBenchmarkArgs{})
}

func (c Pooling) BwdBenchmarkGeneratorArgNames() []string {
	return benchmarkArgNames(poolingBenchmarkArgs{})
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
	switch c.OnnxOperatorType() {
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
