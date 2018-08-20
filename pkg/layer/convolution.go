package layer

import (
	"math"
	"strings"

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

func (Conv) Description() string {
	return ``
}

func (Conv) OperatorType() string {
	return "Conv"
}

func (c *Conv) InferShape(inputLayers []dlperf.Layer) {
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

func (c Conv) FwdBenchmarkArgs() []string {
	return []string{""}
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

func (c Conv) FwdBenchmarkFilter(datatype, algorithm string) benchmark.Benchmark {
	// pp.Println(c.InputShapes()[0])
	// pp.Println(c.KernelShape)
	if algorithm == "" {
		algorithm = c.FwdBenchmarkAlgorithms()[0]
	}
	return benchmark.Benchmark{
		Name: mkBenchmarkFilterName(&c, datatype, algorithm),
		Attributes: map[string]interface{}{
			"input_batch_size": c.inputShapes[0][0],
			"input_channels":   c.inputShapes[0][1],
			"input_width":      c.inputShapes[0][2],
			"input_height":     c.inputShapes[0][3],
			"filter_height":    c.KernelShape[0],
			"filter_width":     c.KernelShape[1],
			"pad_height":       c.Pads[0],
			"pad_width":        c.Pads[1],
			"stride_height":    c.Strides[0],
			"stride_width":     c.Strides[1],
			// "dilation_height":  c.Dilations[0],
			// "dilation_width":   c.Dilations[1],
		},
	}
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
