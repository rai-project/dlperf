package layer

import (
	"github.com/mitchellh/hashstructure"
	dlperf "github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/dlperf/pkg/benchmark"
	"github.com/spf13/cast"
)

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gemm

//easyjson:json
type Gemm struct {
	*Base  `json:",inline,flatten,omitempty"`
	Alpha  float64 `json:"alpha,omitempty"`
	Beta   float64 `json:"beta,omitempty"`
	TransA int64   `json:"transa,omitempty"`
	TransB int64   `json:"transb,omitempty"`
}

func (Gemm) OperatorType() string {
	return "Gemm"
}

func (Gemm) Description() string {
	return ``
}

func (c *Gemm) InferShape(inputLayers dlperf.Layers) {
	c.SetInputShapes(getOutputShapes(inputLayers))

	aShape := c.InputShapes()[0]
	var am int64
	if c.TransA == 0 {
		am = aShape[0]
	} else {
		am = aShape[1]
	}

	bShape := c.InputShapes()[1]
	var bn int64
	if c.TransB == 0 {
		bn = bShape[1]
	} else {
		bn = bShape[0]
	}

	yShape := dlperf.Shape{am, bn}
	c.SetOutputShapes([]dlperf.Shape{yShape})
}

func (c Gemm) FwdBenchmarkName(opts ...dlperf.FwdBenchmarkArgsOptionFunc) string {
	aShape := c.InputShapes()[0]
	var am int64
	if c.TransA == 0 {
		am = aShape[0]
	} else {
		am = aShape[1]
	}

	bShape := c.InputShapes()[1]
	var bn int64
	if c.TransB == 0 {
		bn = bShape[1]
	} else {
		bn = bShape[0]
	}

	if am == 1 || bn == 1 {
		return "LAYER_CUBLAS_GEMV_FWD"
	}

	return "LAYER_CUBLAS_GEMM_FWD"
}

func (c Gemm) BwdBenchmarkName(opts ...dlperf.BwdBenchmarkArgsOptionFunc) string {
	aShape := c.InputShapes()[0]
	var am int64
	if c.TransA == 0 {
		am = aShape[0]
	} else {
		am = aShape[1]
	}

	bShape := c.InputShapes()[1]
	var bn int64
	if c.TransB == 0 {
		bn = bShape[1]
	} else {
		bn = bShape[0]
	}

	if am == 1 || bn == 1 {
		return "LAYER_CUBLAS_GEMV_BWD"
	}

	return "LAYER_CUBLAS_GEMM_BWD"
}

func (c Gemm) FwdTiming(system string /* hardware/software struct */) string {
	return ""
}

func (c Gemm) BwdTiming(system string /* hardware/software struct */) string {
	return ""
}

func (c Gemm) FwdBenchmarkAlgorithms(...dlperf.FwdBenchmarkArgsOptionFunc) []string {
	return []string{
		"",
	}
}

func (c Gemm) BwdBenchmarkAlgorithms(...dlperf.BwdBenchmarkArgsOptionFunc) []string {
	return []string{
		"",
	}
}

//easyjson:json
type gemmBenchmarkArgs struct {
	BaseBenchmarkArgs      `json:",inline,flatten,omitempty"`
	BaseBenchmarkInputArgs `json:",inline,flatten,omitempty"`
	BatchSize              int64 `json:"batch_size,omitempty"`
}

func (c Gemm) mkGemmBenchmarkInputArgs() BaseBenchmarkInputArgs {

	aShape := c.InputShapes()[0]
	var am, ak int64
	if c.TransA == 0 {
		am = aShape[0]
		ak = aShape[1]
	} else {
		am = aShape[1]
		ak = aShape[0]
	}

	bShape := c.InputShapes()[1]
	var bn int64
	if c.TransB == 0 {
		bn = bShape[1]
	} else {
		bn = bShape[0]
	}

	if am == 1 {
		return BaseBenchmarkInputArgs{
			Input0:    bn,
			Input1:    ak,
			Input2:    c.TransB,
			Input3:    cast.ToInt64(c.Alpha),
			Input4:    cast.ToInt64(c.Beta),
			Input5:    -1,
			BatchSize: dlperf.GetBatchSize(),
		}
	} else if bn == 1 {
		return BaseBenchmarkInputArgs{
			Input0:    am,
			Input1:    ak,
			Input2:    c.TransB,
			Input3:    cast.ToInt64(c.Alpha),
			Input4:    cast.ToInt64(c.Beta),
			Input5:    -1,
			BatchSize: dlperf.GetBatchSize(),
		}
	}

	return BaseBenchmarkInputArgs{
		Input0:    am,
		Input1:    bn,
		Input2:    ak,
		Input3:    c.TransA,
		Input4:    c.TransB,
		Input5:    cast.ToInt64(c.Alpha),
		Input6:    cast.ToInt64(c.Beta),
		Input7:    -1,
		BatchSize: dlperf.GetBatchSize(),
	}
}

func (c Gemm) FwdBenchmarkArgs(opts ...dlperf.FwdBenchmarkArgsOptionFunc) interface{} {
	res := gemmBenchmarkArgs{
		BaseBenchmarkInputArgs: c.mkGemmBenchmarkInputArgs(),
		BaseBenchmarkArgs:      mkBaseBenchmarkFWDArgs(&c, opts...),
		BatchSize:              dlperf.GetBatchSize(),
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

func (c Gemm) BwdBenchmarkArgs(opts ...dlperf.BwdBenchmarkArgsOptionFunc) interface{} {
	res := gemmBenchmarkArgs{
		BaseBenchmarkInputArgs: c.mkGemmBenchmarkInputArgs(),
		BaseBenchmarkArgs:      mkBaseBenchmarkBWDArgs(&c, opts...),
		BatchSize:              dlperf.GetBatchSize(),
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

func (c Gemm) FwdBenchmarkFilter(datatype, algorithm string, opts ...dlperf.FwdBenchmarkArgsOptionFunc) benchmark.Benchmark {
	return benchmark.Benchmark{
		Name:       mkFwdBenchmarkFilterName(&c, datatype, algorithm, opts...),
		Attributes: benchmarkAttributes(c.FwdBenchmarkArgs()),
	}
}

func (c Gemm) BwdBenchmarkFilter(datatype, algorithm string, opts ...dlperf.BwdBenchmarkArgsOptionFunc) benchmark.Benchmark {
	return benchmark.Benchmark{
		Name:       mkBwdBenchmarkFilterName(&c, datatype, algorithm, opts...),
		Attributes: benchmarkAttributes(c.BwdBenchmarkArgs()),
	}
}

func (c Gemm) FwdBenchmarkGenerator(opts ...dlperf.FwdBenchmarkArgsOptionFunc) string {
	templString := _escFSMustString(false, "/scope/gemm.tmpl")
	return templateExecFWD(&c, templateBasePrefix+templString)
}

func (c Gemm) BwdBenchmarkGenerator(opts ...dlperf.BwdBenchmarkArgsOptionFunc) string {
	templString := _escFSMustString(false, "/scope/gemm.tmpl")
	return templateExecBWD(&c, templateBasePrefix+templString)
}

func (c Gemm) FwdBenchmarkGeneratorArgNames(opts ...dlperf.FwdBenchmarkArgsOptionFunc) []string {
	return benchmarkArgNames(gemmBenchmarkArgs{})
}

func (c Gemm) BwdBenchmarkGeneratorArgNames(opts ...dlperf.BwdBenchmarkArgsOptionFunc) []string {
	return benchmarkArgNames(gemmBenchmarkArgs{})
}

func (c Gemm) Shape() dlperf.ShapeInformation {
	return c.Information().Shape()
}

func (c Gemm) Information() dlperf.LayerInformation {
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

	checkNumber(c.InputShapes, []int{3}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputShapes, []int{1}, c.OperatorType(), "number of outputs")

	aShape := c.InputShapes()[0]
	var am, ak int64
	if c.TransA == 0 {
		am = aShape[0]
		ak = aShape[1]
	} else {
		am = aShape[1]
		ak = aShape[0]
	}

	bShape := c.InputShapes()[1]
	var bn int64
	if c.TransB == 0 {
		bn = bShape[1]
	} else {
		bn = bShape[0]
	}

	numOuts := am * bn

	info.flops = dlperf.FlopsInformation{
		MultiplyAdds: numOuts * ak,
		Additions:    numOuts,
	}

	return info
}

func init() {
	dlperf.Register(&Gemm{})
}
