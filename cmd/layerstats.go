package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime/debug"

	sourcepath "github.com/GeertJohan/go-sourcepath"
	"github.com/Unknwon/com"
	"github.com/alecthomas/repr"
	"github.com/k0kubun/pp"
	"github.com/pkg/errors"
	"github.com/rai-project/dlperf/pkg/onnx"
	"github.com/rai-project/dlperf/pkg/writer"
	"github.com/spf13/cobra"
	"gonum.org/v1/gonum/graph/encoding/dot"
)

func runLayerStats(cmd *cobra.Command, args []string) error {
	defer func() {
		if r := recover(); r != nil {
			fmt.Println(string(debug.Stack()))
			pp.Println("[PANIC] while computing layer stats " + modelPath + " [error = " + repr.String(r) + "]")
		}
	}()

	if com.IsDir(modelPath) {
		baseOutputFileName := outputFileName
		if !com.IsDir(baseOutputFileName) {
			os.MkdirAll(baseOutputFileName, os.ModePerm)
		}
		modelPaths, err := getModelsIn(modelPath)
		if err != nil {
			return errors.Wrapf(err, "unable to glob %s", modelPath)
		}
		progress := newProgress("> Computing stats models", len(modelPaths))
		defer progress.Finish()
		for _, path := range modelPaths {
			progress.Increment()
			modelPath = path
			modelName := getModelName(modelPath)
			outputFileName = filepath.Join(baseOutputFileName, modelName+"."+outputFormat)
			if true {
				pp.Println("processing " + modelName + " from " + modelPath + " to " + outputFileName)
			}
			if err := runLayerStats(cmd, args); err != nil {
				pp.Println("failed processing "+modelName+" from "+modelPath+" to "+outputFileName, errors.WithStack(err))
			}
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

	net, err := onnx.New(modelPath, batchSize)
	if err != nil {
		return err
	}

	// net.RemoveWeights()

	if outputFormat == "dot" {
		grph := net.ToGraph(onnx.GraphPruneInputs(false), onnx.GraphInputsAsConstantNodes(true))

		dotEnc, err := dot.Marshal(grph, net.GetName(), "", "  ")
		if err != nil {
			return err
		}

		img, err := dotToImage(dotEnc)
		if err != nil {
			return err
		}

		err = com.WriteFile(outputFileName, dotEnc)
		if err != nil {
			return err
		}

		println(img)

		return nil
	}

	infos, err := net.Information()
	if err != nil {
		return err
	}

	writer := NewWriter(
		stat{},
		writer.ShowHumanFlops(humanFlops),
	)
	defer writer.Close()

	for _, info := range infos {
		writer.Row(
			stat{
				Name:             info.Name(),
				Type:             info.OperatorType(),
				InputNames:       info.InputNames(),
				OutputNames:      info.OutputNames(),
				ShapeInformation: info.Shape(),
			},
		)
	}

	return nil
}

// layerstatsCmd represents the layerstats command
var layerstatsCmd = &cobra.Command{
	Use:     "layerstats",
	Aliases: []string{"stats"},
	Short:   "Generates layer statistics for the model",
	RunE:    runLayerStats,
}

func init() {
	rootCmd.AddCommand(layerstatsCmd)
}
