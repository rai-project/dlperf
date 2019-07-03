// +build ignore

package main

import (
	"testing"

	"github.com/k0kubun/pp"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

// https://github.com/rahulbhalley/AlexNet-TensorFlow/blob/master/alexnet-design.py
// https://github.com/tensorflow/models/blob/master/research/slim/nets/alexnet.py
// https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

func mkAlexNet() {
	root := op.NewScope().WithDevice("/device:CPU:0")

	data0 := op.Placeholder(root.SubScope("conv1"), tf.Float, op.PlaceholderShape(tf.MakeShape(1, 3, 224, 224)))
	conv1W0 := op.Placeholder(root.SubScope("conv1"), tf.Float, op.PlaceholderShape(tf.MakeShape(11, 11, 3, 96)))
	conv1 := op.Conv2D(
		root.SubScope("conv1"),
		data0,
		conv1W0,
		[]int64{1, 4, 4, 1},
		"VALID",
		op.Conv2DDataFormat("NCHW"),
	)

	conv1B0 := op.Placeholder(root.SubScope("conv1"), tf.Float, op.PlaceholderShape(tf.MakeShape(96)))
	op.BiasAdd(
		root.SubScope("conv1_bias"),
		conv1,
		conv1B0,
		op.BiasAddDataFormat("NCHW"),
	)

	graph, err := root.Finalize()
	if err != nil {
		panic(err.Error())
	}

	for _, g := range graph.Operations() {
		pp.Println(g.Name())
	}

	var sess *tf.Session
	sess, err = tf.NewSession(graph, &tf.SessionOptions{})
	if err != nil {
		panic(err.Error())
	}

	_ = sess
	_ = conv1
}

func TestMakeAlexNet(t *testing.T) {
	mkAlexNet()
}
