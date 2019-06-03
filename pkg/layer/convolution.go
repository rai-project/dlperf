package layer

import (
	"math"
	"strings"

	"github.com/k0kubun/pp"
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
	HasBias     bool         `json:"has_bias,omitempty"`
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

func (c Conv) FwdBenchmarkName(iopts ...dlperf.FwdBenchmarkArgsOptionFunc) string {
	opts := dlperf.CreateFwdBenchmarkArgsOption(iopts...)
	switch opts.ConvFwdType {

	case dlperf.ConvFwdTypeBias:
		return "LAYER_CUDNN_ADD_TENSOR"
	case dlperf.ConvFwdTypeConv:
		return "LAYER_CUDNN_CONV_FWD"
	case dlperf.ConvFwdTypeConvFusedActivation:
		return "LAYER_CUDNN_CONV_BIAS_ACTIVATION_FWD"
	default:
		panic("unknown conv fwd type")
	}
}

func (c Conv) BwdBenchmarkName(iopts ...dlperf.BwdBenchmarkArgsOptionFunc) string {
	opts := dlperf.CreateBwdBenchmarkArgsOption(iopts...)
	switch opts.ConvBwdType {
	case dlperf.ConvBwdTypeData:
		return "LAYER_CUDNN_CONV_BWD_DATA"
	case dlperf.ConvBwdTypeFilter:
		return "LAYER_CUDNN_CONV_BWD_FILTER"
	case dlperf.ConvBwdTypeBias:
		return "LAYER_CUDNN_CONV_BWD_BIAS"
	default:
		panic("unknown conv bwd type")
	}
}

func (c Conv) FwdCUDNNName() string {
	return ""
}

func (c Conv) BwdCUDNNName() string {
	return ""
}

func (c Conv) FwdTiming(system string /* hardware/software struct */) string {
	return ""
}

func (c Conv) BwdTiming(system string /* hardware/software struct */) string {
	return ""
}

func (c Conv) FwdBenchmarkAlgorithms(iopts ...dlperf.FwdBenchmarkArgsOptionFunc) []string {
	opts := dlperf.CreateFwdBenchmarkArgsOption(iopts...)
	if opts.ConvFwdType == dlperf.ConvFwdTypeBias {
		return []string{}
	}
	convAlgs := []string{
		"CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM",
		"CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",
		"CUDNN_CONVOLUTION_FWD_ALGO_GEMM",
		"CUDNN_CONVOLUTION_FWD_ALGO_DIRECT",
		"CUDNN_CONVOLUTION_FWD_ALGO_FFT",
		"CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",
		"CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD",
		"CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED",
	}

	if opts.ConvFwdType == dlperf.ConvFwdTypeConv {
		return convAlgs
	}

	if opts.ConvFwdType == dlperf.ConvFwdTypeConvFusedActivation {
		outers := outerProductString(convAlgs, allReluAlgorithms())
		algs := make([]string, len(outers))
		for ii, outer := range outers {
			algs[ii] = strings.Join(outer, ", ")
		}
		return algs
	}

	panic("invalid conv type")

	return nil
}

func (c Conv) BwdBenchmarkAlgorithms(iopts ...dlperf.BwdBenchmarkArgsOptionFunc) []string {
	opts := dlperf.CreateBwdBenchmarkArgsOption(iopts...)
	if false {
		pp.Println(opts.ConvBwdType.String())
	}
	switch opts.ConvBwdType {
	case dlperf.ConvBwdTypeData:
		return []string{
			"CUDNN_CONVOLUTION_BWD_DATA_ALGO_0",
			"CUDNN_CONVOLUTION_BWD_DATA_ALGO_1",
			"CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT",
			"CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING",
			"CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD",
			"CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED",
		}
	case dlperf.ConvBwdTypeFilter:
		return []string{
			"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0",
			"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1",
			"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT",
			"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3",
			"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED",
			"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING",
		}
	case dlperf.ConvBwdTypeBias:
		return []string{}
	default:
		return []string{}
	}
}

//easyjson:json
type convBenchmarkArgs struct {
	BaseBenchmarkArgs `json:",inline,flatten,omitempty"`
	Input0            int64              `args:"input[0]" hash:"input[0]" json:"input_0,omitempty"`
	Input1            int64              `args:"input[1]" hash:"input[1]" json:"input_1,omitempty"`
	Input2            int64              `args:"input[2]" hash:"input[2]" json:"input_2,omitempty"`
	Input3            int64              `args:"input[3]" hash:"input[3]" json:"input_3,omitempty"`
	FilterCount       int64              `args:"filter_count" hash:"filter_count" json:"filter_count,omitempty"`
	FilterHeight      int64              `args:"filter_height" hash:"filter_height" json:"filter_height,omitempty"`
	FilterWidth       int64              `args:"filter_width" hash:"filter_width" json:"filter_width,omitempty"`
	PadHeight         int64              `args:"pad_height" hash:"pad_height" json:"pad_height,omitempty"`
	PadWidth          int64              `args:"pad_width" hash:"pad_width" json:"pad_width,omitempty"`
	StrideHeight      int64              `args:"stride_height" hash:"stride_height" json:"stride_height,omitempty"`
	StrideWidth       int64              `args:"stride_width" hash:"stride_width" json:"stride_width,omitempty"`
	DilationWidth     int64              `args:"dilation_height" hash:"dilation_height" json:"dilation_width,omitempty"`
	DilationHeight    int64              `args:"dilation_width" hash:"dilation_width" json:"dilation_height,omitempty"`
	ConvFwdType       dlperf.ConvFwdType `args:"conv_fwd_type" hash:"conv_fwd_type" json:"conv_fwd_type,omitempty"`
	ConvBwdType       dlperf.ConvBwdType `args:"conv_bwd_type" hash:"conv_bwd_type" json:"conv_bwd_type,omitempty"`
	BatchSize         int64              `args:"batch_size" hash:"batch_size" json:"batch_size,omitempty"`
	Group             int64              `args:"-" hash:"-" json:"group,omitempty"`
	BiasDim           int64              `args:"bias_dim" hash:"bias_dim" json:"bias_dim,omitempty"`
	Alpha             float64            `args:"alpha" hash:"alpha" json:"alpha,omitempty"`
	Beta              float64            `args:"beta" hash:"beta" json:"beta,omitempty"`
}

func (c Conv) FwdBenchmarkArgs(iopts ...dlperf.FwdBenchmarkArgsOptionFunc) interface{} {
	opts := dlperf.CreateFwdBenchmarkArgsOption(iopts...)
	inShapes := c.InputShapes()

	var res convBenchmarkArgs
	if opts.ConvFwdType == dlperf.ConvFwdTypeBias {
		res = convBenchmarkArgs{
			Input0:            inShapes[0][0],
			Input1:            inShapes[0][1],
			Input2:            inShapes[0][2],
			Input3:            inShapes[0][3],
			BiasDim:           inShapes[2][0],
			Alpha:             1.0,
			Beta:              0.0,
			BatchSize:         dlperf.GetBatchSize(),
			ConvFwdType:       opts.ConvFwdType,
			BaseBenchmarkArgs: mkBaseBenchmarkFWDArgs(&c, iopts...),
		}
	} else {
		res = convBenchmarkArgs{
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
			BatchSize:         dlperf.GetBatchSize(),
			ConvFwdType:       opts.ConvFwdType,
			BaseBenchmarkArgs: mkBaseBenchmarkFWDArgs(&c, iopts...),
			Group:             c.Group,
		}
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

func (c Conv) BwdBenchmarkArgs(iopts ...dlperf.BwdBenchmarkArgsOptionFunc) interface{} {
	opts := dlperf.CreateBwdBenchmarkArgsOption(iopts...)
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
		BaseBenchmarkArgs: mkBaseBenchmarkBWDArgs(&c, iopts...),
		ConvBwdType:       opts.ConvBwdType,
		BatchSize:         dlperf.GetBatchSize(),
		Group:             c.Group,
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

func (c Conv) FwdBenchmarkFilter(datatype, algorithm string, opts ...dlperf.FwdBenchmarkArgsOptionFunc) benchmark.Benchmark {
	return benchmark.Benchmark{
		Name:       mkFwdBenchmarkFilterName(&c, datatype, algorithm, opts...),
		Attributes: benchmarkAttributes(c.FwdBenchmarkArgs(opts...)),
	}
}

func (c Conv) BwdBenchmarkFilter(datatype, algorithm string, opts ...dlperf.BwdBenchmarkArgsOptionFunc) benchmark.Benchmark {
	return benchmark.Benchmark{
		Name:       mkBwdBenchmarkFilterName(&c, datatype, algorithm, opts...),
		Attributes: benchmarkAttributes(c.BwdBenchmarkArgs(opts...)),
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

func (c Conv) FwdBenchmarkGenerator(iopts ...dlperf.FwdBenchmarkArgsOptionFunc) string {
	var templString string
	opts := dlperf.CreateFwdBenchmarkArgsOption(iopts...)

	switch opts.ConvFwdType {
	case dlperf.ConvFwdTypeBias:
		templString = _escFSMustString(false, "/scope/add_tensor.tmpl")
	case dlperf.ConvFwdTypeConv:
		templString = _escFSMustString(false, "/scope/conv_fwd.tmpl")
	case dlperf.ConvFwdTypeConvFusedActivation:
		templString = _escFSMustString(false, "/scope/cudnn_conv_bias_activation_fwd.tmpl")
	default:
		panic("invalid fwd convolution algorithm")
	}
	return templateExecFWD(&c, templateBasePrefix+templString+templateBaseSuffix, iopts...)
}

func (c Conv) BwdBenchmarkGenerator(iopts ...dlperf.BwdBenchmarkArgsOptionFunc) string {
	var templString string
	opts := dlperf.CreateBwdBenchmarkArgsOption(iopts...)

	switch opts.ConvBwdType {
	case dlperf.ConvBwdTypeData, dlperf.ConvBwdTypeFilter:
		templString = _escFSMustString(false, "/scope/conv_bwd.tmpl")
	case dlperf.ConvBwdTypeBias:
		templString = _escFSMustString(false, "/scope/conv_bias.tmpl")
	default:
		panic("invalid bwd convolution algorithm")
	}

	return templateExecBWD(&c, templateBasePrefix+templString+templateBaseSuffix, iopts...)
}

func (c Conv) FwdBenchmarkGeneratorArgNames() []string {
	return benchmarkArgNames(convBenchmarkArgs{})
}

func (c Conv) BwdBenchmarkGeneratorArgNames() []string {
	return benchmarkArgNames(convBenchmarkArgs{})
}

func (c Conv) Shape() dlperf.ShapeInformation {
	return c.Information().Shape()
}

func (c Conv) Flops(algorithm string) dlperf.FlopsInformation {

	if algorithm == "" {

	}
	inputShapes := c.InputShapes()[0]
	outputShapes := c.OutputShapes()[0]

	nIn := inputShapes[0]
	cIn := inputShapes[1]

	cOut := outputShapes[1]
	hOut := outputShapes[2]
	wOut := outputShapes[3]

	kernelH := c.Dilations[0]*(c.KernelShape[0]-1) + 1
	kernelW := c.Dilations[1]*(c.KernelShape[1]-1) + 1

	// expand
	// see https://arxiv.org/pdf/1802.09941.pdf Page 17 Table 6
	flops := dlperf.FlopsInformation{
		MultiplyAdds: int64(kernelH*kernelW*hOut*wOut*cIn*cOut*nIn) / c.Group,
		// Additions: int64(cOut*hOut*wOut*nIn), // bias
	}

	return flops

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

	info.flops = c.Flops("")

	return info
}

func init() {
	dlperf.Register(&Conv{})
}
