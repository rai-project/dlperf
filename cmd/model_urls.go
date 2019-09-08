package cmd

import (
	"github.com/rai-project/dlperf/pkg/writer"
	"sort"
)

type modelURLInfo struct {
	Name string `json:"name,omitempty"`
	URL  string `json:"url,omitempty"`
	Year int `json:"year,omitempty"`
}

func (modelURLInfo) Header(opts ...writer.Option) []string {
	return []string{"Name", "URL", "Year"}
}
func (m modelURLInfo) Row(opts ...writer.Option) []string {
	return []string{m.Name, m.URL, m.Year}
}

var ourModelURLs = []modelURLInfo{
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/alexnet_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/cifar_resnet110_v1_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/cifar_resnet110_v2_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/cifar_resnet20_v1_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/cifar_resnet20_v2_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/cifar_resnet56_v1_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/cifar_resnet56_v2_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/cifar_resnext29_16x64d_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/cifar_resnext29_32x4d_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/cifar_wideresnet16_10_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/cifar_wideresnet28_10_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/cifar_wideresnet40_8_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/darknet53_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/densenet121_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/densenet161_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/densenet169_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/densenet201_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/inceptionv3_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/mobilenet0.25_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/mobilenet0.5_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/mobilenet0.75_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/mobilenet1.0_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/mobilenetv2_0.25_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/mobilenetv2_0.5_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/mobilenetv2_0.75_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/mobilenetv2_1.0_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/resnet101_v1_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/resnet101_v1b_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/resnet101_v1c_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/resnet101_v1d_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/resnet101_v1e_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/resnet101_v2_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/resnet152_v1_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/resnet152_v1b_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/resnet152_v1c_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/resnet152_v1d_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/resnet152_v1e_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/resnet152_v2_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/resnet18_v1_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/resnet18_v1b_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/resnet18_v2_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/resnet34_v1_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/resnet34_v1b_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/resnet34_v2_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/resnet50_v1_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/resnet50_v1b_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/resnet50_v1c_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/resnet50_v1d_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/resnet50_v1e_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/resnet50_v2_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/resnext101_32x4d_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/resnext101_64x4d_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/resnext50_32x4d_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/se_resnet101_v1_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/se_resnet101_v2_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/se_resnet152_v1_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/se_resnet152_v2_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/se_resnet18_v1_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/se_resnet18_v2_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/se_resnet34_v1_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/se_resnet34_v2_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/se_resnet50_v1_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/se_resnet50_v2_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/squeezenet1.0_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/squeezenet1.1_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/vgg11_bn_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/vgg11_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/vgg13_bn_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/vgg13_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/vgg16_bn_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/vgg16_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/vgg19_bn_float32.onnx",
	// "http://s3.amazonaws.com/store.carml.org/models/onnx/gluoncv_convert/vgg19_float32.onnx",
}

var onnxModelURLs = []modelURLInfo{
	// modelURLInfo{
	// 	URL:  "https://onnxzoo.blob.core.windows.net/models/opset_9/bidaf/bidaf.tar.gz",
	// 	Name: "BiDAF",
	// },
	// "https://s3.amazonaws.com/download.onnx/models/opset_3/bvlc_alexnet.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_6/bvlc_alexnet.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_7/bvlc_alexnet.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_8/bvlc_alexnet.tar.gz",
	modelURLInfo{
		URL:  "https://s3.amazonaws.com/download.onnx/models/opset_9/bvlc_alexnet.tar.gz",
		Name: "BVLC_AlexNet",
		Year: 2012,
	},
	// "https://s3.amazonaws.com/download.onnx/models/opset_3/bvlc_googlenet.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_6/bvlc_googlenet.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_7/bvlc_googlenet.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_8/bvlc_googlenet.tar.gz",
	modelURLInfo{
		URL:  "https://s3.amazonaws.com/download.onnx/models/opset_9/bvlc_googlenet.tar.gz",
		Name: "BVLC_GoogleNet",
		Year: 2014,
	},
	// "https://s3.amazonaws.com/download.onnx/models/opset_3/bvlc_reference_caffenet.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_6/bvlc_reference_caffenet.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_7/bvlc_reference_caffenet.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_8/bvlc_reference_caffenet.tar.gz",
	modelURLInfo{
		URL:  "https://s3.amazonaws.com/download.onnx/models/opset_9/bvlc_reference_caffenet.tar.gz",
		Name: "BVLC_CaffeNet",
		Year: 2012,
	},
	// "https://s3.amazonaws.com/download.onnx/models/opset_3/bvlc_reference_rcnn_ilsvrc13.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_6/bvlc_reference_rcnn_ilsvrc13.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_7/bvlc_reference_rcnn_ilsvrc13.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_8/bvlc_reference_rcnn_ilsvrc13.tar.gz",
	modelURLInfo{
		URL:  "https://s3.amazonaws.com/download.onnx/models/opset_9/bvlc_reference_rcnn_ilsvrc13.tar.gz",
		Name: "BVLC_RCNN_ILSVRC13",
		Year: 2013,
	},
	// "https://s3.amazonaws.com/download.onnx/models/opset_3/densenet121.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_6/densenet121.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_7/densenet121.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_8/densenet121.tar.gz",
	modelURLInfo{
		URL:  "https://s3.amazonaws.com/download.onnx/models/opset_9/densenet121.tar.gz",
		Name: "DenseNet-121",
		Year: 2016,
	},
	// "https://onnxzoo.blob.core.windows.net/models/opset_2/emotion_ferplus/emotion_ferplus.tar.gz",
	// "https://onnxzoo.blob.core.windows.net/models/opset_7/emotion_ferplus/emotion_ferplus.tar.gz",
	modelURLInfo{
		URL:  "https://onnxzoo.blob.core.windows.net/models/opset_8/emotion_ferplus/emotion_ferplus.tar.gz",
		Name: "Emotion-FerPlus",
		Year: 2016,
	},
	// "https://s3.amazonaws.com/download.onnx/models/opset_3/inception_v1.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_6/inception_v1.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_7/inception_v1.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_8/inception_v1.tar.gz",
	modelURLInfo{
		URL:  "https://s3.amazonaws.com/download.onnx/models/opset_9/inception_v1.tar.gz",
		Name: "Inception-v1",
		Year: 2015,
	},
	// "https://s3.amazonaws.com/download.onnx/models/opset_3/inception_v2.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_6/inception_v2.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_7/inception_v2.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_8/inception_v2.tar.gz",
	modelURLInfo{
		URL:  "https://s3.amazonaws.com/download.onnx/models/opset_9/inception_v2.tar.gz",
		Name: "Inception-v2",
		Year: 2012,
	},
	// "https://onnxzoo.blob.core.windows.net/models/opset_1/mnist/mnist.tar.gz",
	// "https://onnxzoo.blob.core.windows.net/models/opset_7/mnist/mnist.tar.gz",
	modelURLInfo{
		URL:  "https://onnxzoo.blob.core.windows.net/models/opset_8/mnist/mnist.tar.gz",
		Name: "MNIST",
		Year: 2010,
	},
	modelURLInfo{
		URL:  "https://s3.amazonaws.com/onnx-model-zoo/arcface/resnet100/resnet100.tar.gz",
		Name: "ArcFace",
		Year: 2018,
	},
	modelURLInfo{
		URL:  "https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.tar.gz",
		Name: "MobileNet-v2",
		Year: 2017,
	},
	modelURLInfo{
		URL:  "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v1/resnet18v1.tar.gz",
		Name: "ResNet018-v1",
		Year: 2015,
	},
	modelURLInfo{
		URL:  "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet34v1/resnet34v1.tar.gz",
		Name: "ResNet034-v1",
		Year: 2015,
	},
	modelURLInfo{
		URL:  "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v1/resnet50v1.tar.gz",
		Name: "ResNet050-v1",
		Year: 2015,
	},
	modelURLInfo{
		URL:  "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet101v1/resnet101v1.tar.gz",
		Name: "ResNet101-v1",
		Year: 2015,
	},
	modelURLInfo{
		URL:  "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet152v1/resnet152v1.tar.gz",
		Name: "ResNet152-v1",
		Year: 2015,
	},
	modelURLInfo{
		URL:  "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v2/resnet18v2.tar.gz",
		Name: "ResNet018-v2",
		Year: 2016,
	},
	modelURLInfo{
		URL:  "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet34v2/resnet34v2.tar.gz",
		Name: "ResNet034-v2",
		Year: 2016,
	},
	modelURLInfo{
		URL:  "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v2/resnet50v2.tar.gz",
		Name: "ResNet050-v2",
		Year: 2016,
	},
	modelURLInfo{
		URL:  "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet101v2/resnet101v2.tar.gz",
		Name: "ResNet101-v2",
		Year: 2016,
	},
	modelURLInfo{
		URL:  "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet152v2/resnet152v2.tar.gz",
		Name: "ResNet152-v2",
		Year: 2016,
	},
	modelURLInfo{
		URL:  "https://s3.amazonaws.com/onnx-model-zoo/squeezenet/squeezenet1.1/squeezenet1.1.tar.gz",
		Name: "Squeezenet-v1.1",
		Year: 2016,
	},
	modelURLInfo{
		URL:  "https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16/vgg16.tar.gz",
		Name: "VGG16",
		Year: 2014,
	},
	modelURLInfo{
		URL:  "https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16-bn/vgg16-bn.tar.gz",
		Name: "VGG16-BN",
		Year: 2014,
	},
	modelURLInfo{
		URL:  "https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg19/vgg19.tar.gz",
		Name: "VGG19",
		Year: 2014,
	},
	modelURLInfo{
		URL:  "https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg19-bn/vgg19-bn.tar.gz",
		Name: "VGG19-BN",
		Year: 2014,
	},
	modelURLInfo{
		URL:  "https://s3.amazonaws.com/onnx-model-zoo/duc/ResNet101_DUC_HDC.tar.gz",
		Name: "DUC",
		Year: 2017,
	},
	// "https://s3.amazonaws.com/download.onnx/models/opset_3/resnet50.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_6/resnet50.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_7/resnet50.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_8/resnet50.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_9/resnet50.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_3/shufflenet.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_6/shufflenet.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_7/shufflenet.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_8/shufflenet.tar.gz",
	modelURLInfo{
		URL:  "https://s3.amazonaws.com/download.onnx/models/opset_9/shufflenet.tar.gz",
		Name: "Shufflenet",
		Year: 2015,
	},
	// "https://s3.amazonaws.com/download.onnx/models/opset_3/squeezenet.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_6/squeezenet.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_7/squeezenet.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_8/squeezenet.tar.gz",
	// modelURLInfo{
	//   URL: "https://s3.amazonaws.com/download.onnx/models/opset_9/squeezenet.tar.gz",
	//   Name: "",
	// },
	// modelURLInfo{
	// 	URL:  "https://onnxzoo.blob.core.windows.net/models/opset_10/ssd/ssd.onnx",
	// 	Name: "ssd",
	// },
	// "https://onnxzoo.blob.core.windows.net/models/opset_1/tiny_yolov2/tiny_yolov2.tar.gz",
	// "https://onnxzoo.blob.core.windows.net/models/opset_7/tiny_yolov2/tiny_yolov2.tar.gz",
	modelURLInfo{
		URL:  "https://onnxzoo.blob.core.windows.net/models/opset_8/tiny_yolov2/tiny_yolov2.tar.gz",
		Name: "Tiny_YOLO-v2",
		Year: 2016,
	},
	// "https://s3.amazonaws.com/download.onnx/models/opset_3/vgg19.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_6/vgg19.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_7/vgg19.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_8/vgg19.tar.gz",
	// modelURLInfo{
	//   URL: "https://s3.amazonaws.com/download.onnx/models/opset_9/vgg19.tar.gz",
	//   Name: "",
	// },
	// modelURLInfo{
	// 	URL:  "https://onnxzoo.blob.core.windows.net/models/opset_10/yolov3/yolov3.onnx",
	// 	Name: "Yolov3",
	// },
	// "https://s3.amazonaws.com/download.onnx/models/opset_3/zfnet512.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_6/zfnet512.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_7/zfnet512.tar.gz",
	// "https://s3.amazonaws.com/download.onnx/models/opset_8/zfnet512.tar.gz",
	modelURLInfo{
		URL:  "https://s3.amazonaws.com/download.onnx/models/opset_9/zfnet512.tar.gz",
		Name: "Zfnet512",
		Year: 2013,
	},
}

var modelURLs = append(ourModelURLs, onnxModelURLs...)

func init() {
	sort.Slice(modelURLs, func(ii, jj int) bool {
		return modelURLs[ii].Name < modelURLs[jj].Name
	})
}
