package layer

import (
	"bytes"
	"math"
	"strings"
	"text/template"

	"github.com/fatih/structs"
	"github.com/mitchellh/hashstructure"
	"github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/dlperf/pkg/benchmark"
)

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv
// NCHW tensor layout for passing inputs and outputs

type Conv struct {
	*Base       `json:",inline,flatten,omitempty"`
	AutoPad     string       `json:"auto_pad,omitempty"`
	Dilations   dlperf.Shape `json:"dilation,omitempty"`
	Group       int64        `json:"group,omitempty"`
	KernelShape dlperf.Shape `json:"kernel_shape,omitempty"`
	Pads        dlperf.Shape `json:"pads,omitempty"`
	Strides     dlperf.Shape `json:"strides,omitempty"`
}

func (Conv) OperatorType() string {
	return "Conv"
}

func (Conv) Description() string {
	return ``
}

func (c *Conv) InferShape(inputLayers dlperf.Layers) {
	xShape := c.inputShapes[0]
	xn := xShape[0]
	xh := xShape[2]
	xw := xShape[3]

	wShape := c.inputShapes[1]
	wn := wShape[0]

	wh := wShape[2]
	ww := wShape[3]

	yn := xn
	yc := wn

	var yh, yw int64
	switch strings.ToLower(c.AutoPad) {
	case "notset", "valid":
		yh = int64(math.Ceil(float64(xh+c.Pads[0]+c.Pads[1]-(c.Dilations[0]*(wh-1)+1))/float64(c.Strides[0]))) + 1
		yw = int64(math.Ceil(float64(xw+c.Pads[2]+c.Pads[3]-(c.Dilations[1]*(ww-1)+1))/float64(c.Strides[1]))) + 1
	case "same_upper", "same_lower":
		yh = int64(math.Ceil(float64(xh) / float64(c.Strides[0])))
		yw = int64(math.Ceil(float64(xw) / float64(c.Strides[1])))
	default:
		panic("invalid pad form " + c.AutoPad)
	}
	yShape := dlperf.Shape{yn, yc, yh, yw}
	c.SetOutputShapes([]dlperf.Shape{yShape})
}

func (c Conv) FwdBenchmarkName() string {
	return "LAYER_CUDNN_CONV_FWD"
}

func (c Conv) FwdCUDNNName() string {
	return ""
}

func (c Conv) FwdTiming(system string /* hardware/software struct */) string {
	return ""
}

func (c Conv) FwdBenchmarkAlgorithms() []string {
	return []string{
		"CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM",
		"CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_​PRECOMP_GEMM",
		"CUDNN_CONVOLUTION_FWD_ALGO_GEMM",
		"CUDNN_CONVOLUTION_FWD_ALGO_DIRECT",
		"CUDNN_CONVOLUTION_FWD_ALGO_FFT",
		"CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",
		"CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD",
		"CUDNN_CONVOLUTION_FWD_ALGO_​WINOGRAD_NONFUSED",
	}
}

type convBenchmarkArgs struct {
	Input0            int64  `args:"input[0]"`
	Input1            int64  `args:"input[1]"`
	Input2            int64  `args:"input[2]"`
	Input3            int64  `args:"input[3]"`
	FilterCount       int64  `args:"filter_count"`
	FilterHeight      int64  `args:"filter_height"`
	FilterWidth       int64  `args:"filter_width"`
	PadHeight         int64  `args:"pad_height"`
	PadWidth          int64  `args:"pad_width"`
	StrideHeight      int64  `args:"stride_height"`
	StrideWidth       int64  `args:"stride_width"`
	DilationWidth     int64  `args:"dilation_height"`
	DilationHeight    int64  `args:"dilation_width"`
	ArgNames          string `args:"-"`
	UniqueBenchmarkID uint64 `args:"-"`
}

func (c Conv) FwdBenchmarkArgs() convBenchmarkArgs {
	inShapes := c.InputShapes()

	res := convBenchmarkArgs{
		Input0:         inShapes[0][0],
		Input1:         inShapes[0][1],
		Input2:         inShapes[0][2],
		Input3:         inShapes[0][3],
		FilterCount:    inShapes[1][0],
		FilterHeight:   c.KernelShape[0],
		FilterWidth:    c.KernelShape[1],
		PadHeight:      c.Pads[0],
		PadWidth:       c.Pads[2],
		StrideHeight:   c.Strides[0],
		StrideWidth:    c.Strides[1],
		DilationHeight: c.Dilations[0],
		DilationWidth:  c.Dilations[1],
		ArgNames:       c.FwdBenchmarkGeneratorArgNames(),
	}

	hash, err := hashstructure.Hash(res, nil)
	if err != nil {
		panic(err)
	}
	res.UniqueBenchmarkID = hash

	return res
}

func (c Conv) FwdBenchmarkFilter(datatype, algorithm string) benchmark.Benchmark {
	if algorithm == "" {
		algorithm = c.FwdBenchmarkAlgorithms()[0]
	}

	attrs := map[string]interface{}{}
	for _, field := range structs.New(c.FwdBenchmarkArgs()).Fields() {
		tag := field.Tag("args")
		if tag == "" || tag == "-" {
			continue
		}
		attrs[tag] = field.Value()
	}

	return benchmark.Benchmark{
		Name:       mkBenchmarkFilterName(&c, datatype, algorithm),
		Attributes: attrs,
	}
}

func (c Conv) FwdBenchmarkGeneratorArgNames() string {
	tags := []string{}
	for _, field := range structs.New(convBenchmarkArgs{}).Fields() {
		tag := field.Tag("args")
		if tag == "" || tag == "-" {
			continue
		}
		tags = append(tags, "\""+tag+"\"")
	}
	return strings.Join(tags, ",")
}

func (c Conv) FwdBenchmarkGenerator() string {
	templString := `
namespace conv__{{.UniqueBenchmarkID}} {
  template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
  static void LAYER_CUDNN_CONV_FWD_INT8__{{.UniqueBenchmarkID}}(benchmark::State& state) {
    CUDNN_CONV_FWD_Impl<int8_t, convolution_algorithm>(state);
  }

  template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
  static void LAYER_CUDNN_CONV_FWD_INT32__{{.UniqueBenchmarkID}}(benchmark::State& state) {
    CUDNN_CONV_FWD_Impl<int32_t, convolution_algorithm>(state);
  }

  template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
  static void LAYER_CUDNN_CONV_FWD_HALF__{{.UniqueBenchmarkID}}(benchmark::State& state) {
    CUDNN_CONV_FWD_Impl<__half, convolution_algorithm>(state);
  }

  #ifdef CUDNN_SUPPORTS_TENSOR_OPS
  template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
  static void LAYER_CUDNN_CONV_FWD_HALF_TENSOROP__{{.UniqueBenchmarkID}}(benchmark::State& state) {
    CUDNN_CONV_FWD_Impl<__half, convolution_algorithm, CUDNN_TENSOR_OP_MATH>(state);
  }
  #endif

  template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
  static void LAYER_CUDNN_CONV_FWD_FLOAT__{{.UniqueBenchmarkID}}(benchmark::State& state) {
    CUDNN_CONV_FWD_Impl<float, convolution_algorithm>(state);
  }

  template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
  static void LAYER_CUDNN_CONV_FWD_DOUBLE__{{.UniqueBenchmarkID}}(benchmark::State& state) {
    CUDNN_CONV_FWD_Impl<double, convolution_algorithm>(state);
  }


#define InputArgs \
  Arg(
    {{.Input0}}, /* Input0 */ \
    {{.Input1}}, /* Input1 */ \
    {{.Input2}}, /* Input2 */ \
    {{.Input3}}, /* Input3 */ \
    {{.FilterCount}}, /* FilterCount */ \
    {{.FilterHeight}}, /* FilterHeight */ \
    {{.FilterWidth}}, /* FilterWidth */ \
    {{.PadHeight}}, /* PadHeight */ \
    {{.PadWidth}}, /* PadWidth */ \
    {{.StrideHeight}}, /* StrideHeight */ \
    {{.StrideWidth}}, /* StrideWidth */ \
    {{.DilationHeight}}, /* DilationHeight */ \
    {{.DilationWidth}}, /* DilationWidth */ \
  )

#define BENCHMARK_LAYER_CUDNN_CONV_FWD(b)                                                                                             \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)->ArgNames({{.ArgNames}})->InputArgs()->UseManualTime();                   \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)->ArgNames({{.ArgNames}})->InputArgs()->UseManualTime();           \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_GEMM)->ArgNames({{.ArgNames}})->InputArgs()->UseManualTime();                            \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_DIRECT)->ArgNames({{.ArgNames}})->InputArgs()->UseManualTime();                          \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_FFT)->ArgNames({{.ArgNames}})->InputArgs()->UseManualTime();                             \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING)->ArgNames({{.ArgNames}})->InputArgs()->UseManualTime();                      \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD)->ArgNames({{.ArgNames}})->InputArgs()->UseManualTime();                        \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED)->ArgNames({{.ArgNames}})->InputArgs()->UseManualTime()

  /* BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_INT8__{{.UniqueBenchmarkID}}); */
  /* BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_INT32__{{.UniqueBenchmarkID}}); */
  BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_HALF__{{.UniqueBenchmarkID}});
  #ifdef CUDNN_SUPPORTS_TENSOR_OPS
  BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_HALF_TENSOROP__{{.UniqueBenchmarkID}});
  #endif // CUDNN_SUPPORTS_TENSOR_OPS
  BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_FLOAT__{{.UniqueBenchmarkID}});
  BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_DOUBLE__{{.UniqueBenchmarkID}});

#undef InputArgs
#undef BENCHMARK_LAYER_CUDNN_CONV_FWD
}
`

	templArgs := c.FwdBenchmarkArgs()
	tmpl, err := template.New(c.OperatorType()).Parse(templString)
	if err != nil {
		panic(err)
	}
	buf := bytes.NewBufferString("")
	err = tmpl.Execute(buf, templArgs)
	if err != nil {
		panic(err)
	}
	return buf.String()
}

func (c Conv) Shape() dlperf.ShapeInformation {
	return c.Information().Shape()
}

func (c Conv) Information() dlperf.LayerInformation {
	info := &Information{
		Base: c.Base,
		shape: dlperf.ShapeInformation{
			InputShapes:  c.inputShapes,
			OutputShapes: c.outputShapes,
		},
	}

	if isAnyEmpty(c.outputShapes) {
		log.WithField("layer", c.OperatorType()).Info("len(OutputShapes) is 0")
		return info
	}

	checkNumber(c.InputShapes, []int{2, 3}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputShapes, []int{1}, c.OperatorType(), "number of outputs")

	inputShapes := c.InputShapes()[0]
	outputShapes := c.OutputShapes()[0]

	nIn := inputShapes[0]
	cIn := inputShapes[1]

	cOut := outputShapes[1]
	hOut := outputShapes[2]
	wOut := outputShapes[3]

	kernelH := c.Dilations[0]*(c.KernelShape[0]-1) + 1
	kernelW := c.Dilations[1]*(c.KernelShape[1]-1) + 1

	info.flops = dlperf.FlopsInformation{
		MultiplyAdds: int64(kernelH*kernelW*hOut*wOut*cIn*cOut*nIn) / c.Group,
	}

	return info
}

func init() {
	dlperf.Register(&Conv{})
}
