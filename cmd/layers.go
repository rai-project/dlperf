package cmd

import (
	perflayer "github.com/rai-project/dlperf/pkg/layer"
)

var (
	conv          = perflayer.Conv{}
	relu          = perflayer.Relu{}
	pooling       = perflayer.Pooling{}
	softmax       = perflayer.Softmax{}
	batchNorm     = perflayer.BatchNorm{}
	dropout       = perflayer.Dropout{}
	clip          = perflayer.Clip{}
	concat        = perflayer.Concat{}
	constant      = perflayer.Constant{}
	constantInput = perflayer.ConstantInput{}
	elementwise   = perflayer.ElementWise{}
	flatten       = perflayer.Flatten{}
	gemm          = perflayer.Gemm{}
	globalPooling = perflayer.GlobalPooling{}
	identity      = perflayer.Identity{}
	lrn           = perflayer.LRN{}
	matmul        = perflayer.MatMul{}
	reshape       = perflayer.Reshape{}
	scale         = perflayer.Scale{}
	transpose     = perflayer.Transpose{}
	unsqueeze     = perflayer.Unsqueeze{}
)
