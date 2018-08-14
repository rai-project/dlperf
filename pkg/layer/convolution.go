package layer

import (
	"github.com/rai-project/dlperf/pkg"
	"github.com/rai-project/dlperf/pkg/benchmark"
)

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv
// NCHW tensor layout for passing inputs and outputs

type Conv struct {
	Base        `json:",inline,flatten,omitempty"`
	AutoPad     string  `json:"auto_pad,omitempty"`
	Dilations   []int64 `json:"dilation,omitempty"`
	Group       int64   `json:"group,omitempty"`
	KernelShape []int64 `json:"kernel_shape,omitempty"`
	Pads        []int64 `json:"pad_h,omitempty"`
	Strides     []int64 `json:"stride_h,omitempty"`
}

func (Conv) Description() string {
	return ``
}

func (c Conv) FwdBenchmarkName() string {
	return ""
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

func (c *Conv) FwdBenchmarkFilter() benchmark.Benchmark {
	return benchmark.Benchmark{
		Name: "^" + c.FwdBenchmarkName() + ".*",
		Attributes: map[string]interface{}{
			"N": c.W(),
			"C": c.C(),
			"H": c.H(),
			"W": c.W(),
		},
	}
}

func (c *Conv) InferShape() {
	//c.inputdimensions =  dlperf.ShapeInformation{}
}

func (c *Conv) N() int {
	return 0
}

func (c *Conv) C() int {
	return 0
}

func (c *Conv) H() int {
	return 0
}

func (c *Conv) W() int {
	return 0
}

func (c Conv) Shape() dlperf.ShapeInformation {
	return dlperf.ShapeInformation{}
}

func (c Conv) Information() dlperf.LayerInformation {
	info := &Information{
		Base: c.Base,
	}

	if isAnyEmpty(c.inputsDimensions) {
		log.WithField("layer", c.OperatorType()).Info("len(InputsDimensions) is 0")
		return info
	}

	if isAnyEmpty(c.outputsDimensions) {
		log.WithField("layer", c.OperatorType()).Info("len(OutputsDimensions) is 0")
		return info
	}

	checkNumber(c.InputsDimensions, []int{2, 3}, c.OperatorType(), "number of inputs")
	checkNumber(c.OutputsDimensions, []int{1}, c.OperatorType(), "number of outputs")

	inputDimensions := c.InputsDimensions()[0]   // (N x C x H x W)
	outputDimensions := c.OutputsDimensions()[0] // (N x C x H x W)
	// weightDimensions := c.InputsDimensions()[1]  // (C x M x kH x kW)

	nIn := inputDimensions[0]
	cIn := inputDimensions[1]

	cOut := outputDimensions[1]
	hOut := outputDimensions[2]
	wOut := outputDimensions[3]

	// if weightDimensions[2] != c.KernelShape[0] || weightDimensions[3] != c.KernelShape[1] {
	// 	log.WithField("layer", c.OperatorType()).WithField("weight dimensions", weightDimensions).Error("weight dimensions do not match kernel_shape")

	// 	return info
	// }

	kernelH := c.Dilations[0]*(c.KernelShape[0]-1) + 1
	kernelW := c.Dilations[1]*(c.KernelShape[1]-1) + 1

	info.flops = dlperf.FlopsInformation{
		MultiplyAdds: int64(kernelH*kernelW*hOut*wOut*cIn*cOut*nIn) / c.Group,
	}

	info.shape = dlperf.ShapeInformation{
		InputDimensions:  inputDimensions,
		OutputDimensions: outputDimensions,
	}

	return info
}

func init() {
	dlperf.Register(&Conv{})
}
