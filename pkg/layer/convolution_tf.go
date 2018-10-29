// +build tf

package layer

import (
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

// use with https://github.com/golang/go/blob/master/src/testing/benchmark.go#L688

func toTensorflowShape(dims []int64) tf.Shape {
	return tf.MakeShape(dims...)
}

func (c Conv) FwdTensorflow(root *op.Scope) tf.Output {
	inShapes := c.InputShapes()
	in := op.Placeholder(
		root.SubScope(c.Name()),
		tf.Float,
		op.PlaceholderShape(toTensorflowShape(inShapes[0])),
	)
	weights := op.Placeholder(
		root.SubScope(c.Name()),
		tf.Float,
		op.PlaceholderShape(toTensorflowShape(Int64Slice(c.KernelShape).Reverse())),
	)

	conv := op.Conv2D(
		root.SubScope(c.Name()),
		in,
		weights,
		[]int64{1, c.Strides[0], c.Strides[1], 1},
		c.AutoPad,
		op.Conv2DUseCudnnOnGpu(true),
		op.Conv2DDilations(c.Dilations),
		op.Conv2DDataFormat("NCHW"),
	)

	if len(inShapes) > 2 {
		bias := op.Placeholder(
			root.SubScope(c.Name()),
			tf.Float,
			op.PlaceholderShape(tf.MakeShape(inShapes[2][0])),
		)
		conv = op.BiasAdd(
			root.SubScope(c.Name()+"_bias"),
			conv,
			bias,
			op.BiasAddDataFormat("NCHW"),
		)
	}

	return conv
}
