package cmd

import (
	"context"
	"net/http"
	"os"

	"github.com/Unknwon/com"
	"github.com/rai-project/archive"
	"github.com/spf13/cobra"
	"golang.org/x/sync/errgroup"
)

var modelURLs = []string{
	"https://s3.amazonaws.com/download.onnx/models/opset_3/bvlc_alexnet.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_6/bvlc_alexnet.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_7/bvlc_alexnet.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_8/bvlc_alexnet.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_3/bvlc_googlenet.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_6/bvlc_googlenet.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_7/bvlc_googlenet.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_8/bvlc_googlenet.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_3/bvlc_reference_caffenet.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_6/bvlc_reference_caffenet.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_7/bvlc_reference_caffenet.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_8/bvlc_reference_caffenet.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_3/bvlc_reference_rcnn_ilsvrc13.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_6/bvlc_reference_rcnn_ilsvrc13.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_7/bvlc_reference_rcnn_ilsvrc13.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_8/bvlc_reference_rcnn_ilsvrc13.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_3/densenet121.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_6/densenet121.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_7/densenet121.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_8/densenet121.tar.gz",
	"https://www.cntk.ai/OnnxModels/emotion_ferplus/opset_2/emotion_ferplus.tar.gz",
	"https://www.cntk.ai/OnnxModels/emotion_ferplus/opset_7/emotion_ferplus.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_3/inception_v1.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_6/inception_v1.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_7/inception_v1.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_8/inception_v1.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_3/inception_v2.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_6/inception_v2.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_7/inception_v2.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_8/inception_v2.tar.gz",
	"https://www.cntk.ai/OnnxModels/mnist/opset_1/mnist.tar.gz",
	"https://www.cntk.ai/OnnxModels/mnist/opset_7/mnist.tar.gz",
	"https://s3.amazonaws.com/onnx-model-zoo/arcface/resnet100/resnet100.tar.gz",
	"https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.tar.gz",
	"https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v1/resnet18v1.tar.gz",
	"https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet34v1/resnet34v1.tar.gz",
	"https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v1/resnet50v1.tar.gz",
	"https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet101v1/resnet101v1.tar.gz",
	"https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet152v1/resnet152v1.tar.gz",
	"https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v2/resnet18v2.tar.gz",
	"https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet34v2/resnet34v2.tar.gz",
	"https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v2/resnet50v2.tar.gz",
	"https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet101v2/resnet101v2.tar.gz",
	"https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet152v2/resnet152v2.tar.gz",
	"https://s3.amazonaws.com/onnx-model-zoo/squeezenet/squeezenet1.1/squeezenet1.1.tar.gz",
	"https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16/vgg16.tar.gz",
	"https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16-bn/vgg16-bn.tar.gz",
	"https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg19/vgg19.tar.gz",
	"https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg19-bn/vgg19-bn.tar.gz",
	"https://s3.amazonaws.com/onnx-model-zoo/duc/ResNet101_DUC_HDC.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_3/resnet50.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_6/resnet50.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_7/resnet50.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_8/resnet50.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_3/shufflenet.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_6/shufflenet.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_7/shufflenet.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_8/shufflenet.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_3/squeezenet.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_6/squeezenet.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_7/squeezenet.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_8/squeezenet.tar.gz",
	"https://www.cntk.ai/OnnxModels/tiny_yolov2/opset_1/tiny_yolov2.tar.gz",
	"https://www.cntk.ai/OnnxModels/tiny_yolov2/opset_7/tiny_yolov2.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_3/vgg19.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_6/vgg19.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_7/vgg19.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_8/vgg19.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_3/zfnet512.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_6/zfnet512.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_7/zfnet512.tar.gz",
	"https://s3.amazonaws.com/download.onnx/models/opset_8/zfnet512.tar.gz",
}

// downloadModelsCmd represents the downloadmodels command
var downloadModelsCmd = &cobra.Command{
	Use:     "downloadmodels",
	Aliases: []string{"download"},
	RunE: func(cmd *cobra.Command, args []string) error {
		g, _ := errgroup.WithContext(context.Background())
		if !com.IsDir(outputFileName) {
			os.MkdirAll(outputFileName, os.ModePerm)
		}

		for ii := range modelURLs {
			url := modelURLs[ii]
			g.Go(func() error {
				resp, err := http.Get(url)
				if err != nil {
					log.WithError(err).WithField("url", url).Error("failed to download model")
					return nil
				}
				defer resp.Body.Close()

				err = archive.Unzip(resp.Body, outputFileName)
				if err != nil {
					log.WithError(err).WithField("url", url).Error("failed to decompress model")
					return nil
				}

				return nil
			})
		}

		if err := g.Wait(); err != nil {
			return err
		}
		return nil
	},
}

func init() {
	rootCmd.AddCommand(downloadModelsCmd)
}
