package layer

import (
	"fmt"
	"math"

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

func (c *Pooling) InferShape(inputLayers []dlperf.Layer) {
	inputShapes := getOutputShapes(inputLayers)
	xShape := c.inputShapes[0]

	yShape := dlperf.Shape{xShape[0], xShape[1]}
	for ii, xs := range xShape[2:] {
		ys := int64(math.Floor(float64(xs+c.Pads[ii]+c.Pads[ii+1]-c.KernelShape[ii])/float64(c.Strides[ii]))) + 1
		yShape = append(yShape, ys)
	}

	c.SetInputShapes(inputShapes)
	c.SetOutputShapes([]dlperf.Shape{yShape})
}

func (c Pooling) FwdBenchmarkName() string {
	return "CUDNN_POOLING_FWD"
}

func (c Pooling) FwdBenchmarkArgs() []string {
	return []string{""}
}

func (c Pooling) FwdCUDNNName() string {
	return ""
}

func (c Pooling) FwdTiming(system string /* hardware/software struct */) string {
	return ""
}

func (c Pooling) FwdBenchmarkAlgorithms() []string {
	if c.onnxOperatorType == "maxpool" {
		return []string{
			"CUDNN_POOLING_MAX",
			"CUDNN_POOLING_MAX_DETERMINISTIC",
		}
	} else if c.onnxOperatorType == "averagepool" {
		return []string{
			"CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING",
			"CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING",
		}
	}

	return nil
}

func (c Pooling) FwdBenchmarkFilter(datatype, algorithm string) benchmark.Benchmark {
	if algorithm == "" {
		algorithm = c.FwdBenchmarkAlgorithms()[0]
	}
	attrs := map[string]interface{}{}
	for ii, dim := range c.InputShapes()[0] {
		attrs[fmt.Sprintf("input[%d]", ii)] = dim
	}
	return benchmark.Benchmark{
		Name:       mkBenchmarkFilterName(&c, datatype, algorithm),
		Attributes: attrs,
	}
}

func (c Pooling) Shape() dlperf.ShapeInformation {
	return c.Information().Shape()
}

func (c Pooling) Information() dlperf.LayerInformation {
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
