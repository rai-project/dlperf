package cmd

import (
	"os"
	"path/filepath"

	sourcepath "github.com/GeertJohan/go-sourcepath"
	"github.com/Unknwon/com"
	"github.com/k0kubun/pp"
	zglob "github.com/mattn/go-zglob"
	"github.com/pkg/errors"
	"github.com/rai-project/dlperf/pkg/onnx"
	"github.com/spf13/cobra"
	"google.golang.org/grpc/benchmark/stats"
)

var numBuckets = 100

func runWeightsCmd(cmd *cobra.Command, args []string) error {
	if com.IsDir(modelPath) {
		baseOutputFileName := outputFileName
		if !com.IsDir(baseOutputFileName) {
			os.MkdirAll(baseOutputFileName, os.ModePerm)
		}
		modelPaths, err := zglob.Glob(filepath.Join(modelPath, "**", "*.onnx"))
		if err != nil {
			return errors.Wrapf(err, "unable to glob %s", modelPath)
		}
		for _, path := range modelPaths {
			modelPath = path
			modelName := getModelName(modelPath)
			outputFileName = filepath.Join(baseOutputFileName, modelName+"."+outputFormat)
			pp.Println("processing " + modelName + " from " + modelPath + " to " + outputFileName)
			runWeightsCmd(cmd, args)
		}
		return nil
	}

	if modelPath == "" {
		modelPath = filepath.Join(sourcepath.MustAbsoluteDir(), "..", "assets", "onnx_models", "mnist.onnx")
	} else {
		s, err := filepath.Abs(modelPath)
		if err == nil {
			modelPath = s
		}
	}

	if !com.IsFile(modelPath) {
		return errors.Errorf("file %v does not exist", modelPath)
	}

	net, err := onnx.New(modelPath)
	if err != nil {
		return err
	}
	infos, err := net.Information()
	if err != nil {
		return err
	}

	writer := NewWriter(layerWeights{}, humanFlops)
	defer writer.Close()

	for _, info := range infos {
		if info.OperatorType() == "constant_input" || info.OperatorType() == "constant" {
			continue
		}
		histo := stats.NewHistogram(stats.HistogramOptions{
			// NumBuckets is the number of buckets.
			NumBuckets: numBuckets,
			// GrowthFactor is the growth factor of the buckets. A value of 0.1
			// indicates that bucket N+1 will be 10% larger than bucket N.
			GrowthFactor: float64(1.0) / float64(numBuckets),
			// BaseBucketSize is the size of the first bucket.
			BaseBucketSize: float64(1.0) / float64(numBuckets),
			// MinValue is the lower bound of the first bucket.
			MinValue: 0,
		})

		weigths := info.Weigths()
		if weigths == nil {
			panic("weights are nil")
		}
		for _, w := range weigths {
			histo.Add(int64(w * 100))
		}

		writer.Row(
			layerWeights{
				Name:      info.Name(),
				Type:      info.OperatorType(),
				Weigths:   info.Weigths(),
				Histogram: histo,
			},
		)
	}

	return nil
}

var weightsinfoCmd = &cobra.Command{
	Use:     "weightsinfo",
	Aliases: []string{"weights"},
	Short:   "Get weights information about the model",
	RunE:    runWeightsCmd,
}

func init() {
	rootCmd.AddCommand(weightsinfoCmd)
	weightsinfoCmd.PersistentFlags().IntVarP(&numBuckets, "num_buckets", "n", 100, "number of buckets")
}
