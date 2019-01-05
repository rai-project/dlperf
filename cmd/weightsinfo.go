package cmd

import (
	"os"
	"path/filepath"

	sourcepath "github.com/GeertJohan/go-sourcepath"
	"github.com/Unknwon/com"
	"github.com/k0kubun/pp"
	zglob "github.com/mattn/go-zglob"
	"github.com/montanaflynn/stats"
	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"

	"github.com/rai-project/dlperf/pkg/onnx"
)

var numBuckets = 20

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

	dir, err := filepath.Abs(filepath.Dir(outputFileName))
	if err != nil {
		return err
	}
	dir = filepath.Join(dir, getModelName(modelPath))
	if !com.IsDir(dir) {
		os.MkdirAll(dir, os.ModePerm)
	}

	outputFileName = filepath.Join(dir, filepath.Base(outputFileName))
	writer := NewWriter(layerWeights{}, humanFlops)
	defer writer.Close()

	for _, info := range infos {
		if info.OperatorType() == "ConstantInput" {
			continue
		}
		weigths := info.Weigths()
		if weigths == nil {
			continue
		}

		v := make([]float64, len(weigths))
		for ii := range v {
			v[ii] = float64(weigths[ii])
		}

		max, err := stats.Max(v)
		if err != nil {
			return err
		}
		min, err := stats.Min(v)
		if err != nil {
			return err
		}
		sdev, err := stats.StandardDeviation(v)
		if err != nil {
			return err
		}

		writer.Row(
			layerWeights{
				Name:              info.Name(),
				Type:              info.OperatorType(),
				Length:            len(v),
				Max:               max,
				Min:               min,
				StandardDeviation: sdev,
			},
		)

		p, err := plot.New()
		if err != nil {
			return err
		}
		p.Title.Text = info.Name()

		h, err := plotter.NewHist(plotter.Values(v), numBuckets)
		if err != nil {
			return err
		}
		p.Add(h)

		if err := p.Save(4*vg.Inch, 4*vg.Inch, filepath.Join(dir, info.OperatorType()+"_"+info.Name()+".png")); err != nil {
			return err
		}
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
