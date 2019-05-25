package layer

import (
	"math"
	"strings"

	"github.com/mitchellh/hashstructure"
	dlperf "github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/dlperf/pkg/benchmark"
)

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv
// NCHW tensor layout for passing inputs and outputs

//easyjson:json
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
	inputShapes := getOutputShapes(inputLayers)

	xShape := inputShapes[0]
	xn := xShape[0]
	xh := xShape[2]
	xw := xShape[3]

	wShape := inputShapes[1]
	wn := wShape[0]

	wh := wShape[2]
	ww := wShape[3]

	yn := xn
	yc := wn

	var yh, yw int64
	switch strings.ToLower(c.AutoPad) {
	case "notset", "valid":
		yh = int64(math.Floor(float64(xh+c.Pads[0]+c.Pads[1]-(c.Dilations[0]*(wh-1)+1))/float64(c.Strides[0])) + 1)
		yw = int64(math.Floor(float64(xw+c.Pads[2]+c.Pads[3]-(c.Dilations[1]*(ww-1)+1))/float64(c.Strides[1])) + 1)
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
		"CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",
		"CUDNN_CONVOLUTION_FWD_ALGO_GEMM",
		"CUDNN_CONVOLUTION_FWD_ALGO_DIRECT",
		"CUDNN_CONVOLUTION_FWD_ALGO_COUNT",
		"CUDNN_CONVOLUTION_FWD_ALGO_FFT",
		"CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",
		"CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD",
		"CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED",
	}
}

type convBenchmarkArgs struct {
	BaseBenchmarkArgs
	Input0         int64 `args:"input[0]" hash:"input[0]" json:"input_0,omitempty"`
	Input1         int64 `args:"input[1]" hash:"input[1]" json:"input_1,omitempty"`
	Input2         int64 `args:"input[2]" hash:"input[2]" json:"input_2,omitempty"`
	Input3         int64 `args:"input[3]" hash:"input[3]" json:"input_3,omitempty"`
	FilterCount    int64 `args:"filter_count" hash:"filter_count" json:"filter_count,omitempty"`
	FilterHeight   int64 `args:"filter_height" hash:"filter_height" json:"filter_height,omitempty"`
	FilterWidth    int64 `args:"filter_width" hash:"filter_width" json:"filter_width,omitempty"`
	PadHeight      int64 `args:"pad_height" hash:"pad_height" json:"pad_height,omitempty"`
	PadWidth       int64 `args:"pad_width" hash:"pad_width" json:"pad_width,omitempty"`
	StrideHeight   int64 `args:"stride_height" hash:"stride_height" json:"stride_height,omitempty"`
	StrideWidth    int64 `args:"stride_width" hash:"stride_width" json:"stride_width,omitempty"`
	DilationWidth  int64 `args:"dilation_height" hash:"dilation_height" json:"dilation_width,omitempty"`
	DilationHeight int64 `args:"dilation_width" hash:"dilation_width" json:"dilation_height,omitempty"`
}

func (c Conv) FwdBenchmarkArgs() interface{} {
	inShapes := c.InputShapes()

	res := convBenchmarkArgs{
		Input0:            inShapes[0][0],
		Input1:            inShapes[0][1],
		Input2:            inShapes[0][2],
		Input3:            inShapes[0][3],
		FilterCount:       inShapes[1][0],
		FilterHeight:      c.KernelShape[0],
		FilterWidth:       c.KernelShape[1],
		PadHeight:         c.Pads[0],
		PadWidth:          c.Pads[2],
		StrideHeight:      c.Strides[0],
		StrideWidth:       c.Strides[1],
		DilationHeight:    c.Dilations[0],
		DilationWidth:     c.Dilations[1],
		BaseBenchmarkArgs: mkBaseBenchmarkFWDArgs(&c),
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

func (c Conv) FwdBenchmarkFilter(datatype, algorithm string) benchmark.Benchmark {
	if algorithm == "" {
		algorithm = c.FwdBenchmarkAlgorithms()[0]
	}

	return benchmark.Benchmark{
		Name:       mkBenchmarkFilterName(&c, datatype, algorithm),
		Attributes: benchmarkAttributes(c.FwdBenchmarkArgs()),
	}
}

func (c Conv) DataTypes() []dlperf.DataType {
	dts := c.Base.DataTypes()
	// dts = append(
	// 	dts,
	// 	dlperf.DataType{
	// 		Name:  "TensorCoreHalf",
	// 		CType: "__half",
	// 	},
	// )

	return dts
}

func (c Conv) FwdBenchmarkGeneratorArgNames() []string {
	return benchmarkArgNames(convBenchmarkArgs{})
}

func (c Conv) FwdBenchmarkGenerator() string {
	templString := _escFSMustString(false, "/scope/conv.tmpl")

	return templateExecFWD(&c, templateBasePrefix+templString+templateBaseSuffix)
}

func (c Conv) Shape() dlperf.ShapeInformation {
	return c.Information().Shape()
}

func (c Conv) Information() dlperf.LayerInformation {
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
