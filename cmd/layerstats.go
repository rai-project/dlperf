package cmd

import (
	"os"
	"path/filepath"
	"strings"

	sourcepath "github.com/GeertJohan/go-sourcepath"
	"github.com/Unknwon/com"
	"github.com/acarl005/stripansi"
	"github.com/k0kubun/pp"
	zglob "github.com/mattn/go-zglob"
	"github.com/pkg/errors"
	"github.com/rai-project/dlperf/pkg/onnx"
	"github.com/spf13/cobra"
	"gonum.org/v1/gonum/graph/encoding/dot"
)

func runLayerStats(cmd *cobra.Command, args []string) error {
	defer func() {
		if r := recover(); r != nil {
			pp.Println("[PANIC] while computing layer stats " + modelPath + " [error = " + stripansi.Strip(pp.Sprint(r)) + "]")
		}
	}()

	if com.IsDir(modelPath) {

		baseOutputFileName := outputFileName
		if !com.IsDir(baseOutputFileName) {
			os.MkdirAll(baseOutputFileName, os.ModePerm)
		}
		modelPaths, err := zglob.Glob(filepath.Join(modelPath, "**", "*.onnx"))
		if err != nil {
			return errors.Wrapf(err, "unable to glob %s", modelPath)
		}
		progress := newProgress("> Computing stats models", len(modelPaths))
		defer progress.Finish()
		for _, path := range modelPaths {
			modelPath = path
			modelName := getModelName(modelPath)
			if strings.HasPrefix(modelName, ".") {
				continue
			}
			outputFileName = filepath.Join(baseOutputFileName, modelName+"."+outputFormat)
			if false {
				pp.Println("processing " + modelName + " from " + modelPath + " to " + outputFileName)
			}
			if err := runLayerStats(cmd, args); err != nil {
				pp.Println("failed processing "+modelName+" from "+modelPath+" to "+outputFileName, errors.WithStack(err))
			}
			progress.Increment()
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

	// net.RemoveWeights()

	if outputFormat == "dot" {
		grph := net.ToGraph(onnx.GraphPruneInputs(false), onnx.GraphInputsAsConstantNodes(true))

		dotEnc, err := dot.Marshal(grph, net.GetName(), "", "  ", true)
		if err != nil {
			return err
		}

		img, err := dotToImage(dotEnc)
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

	writer := NewWriter(stat{}, humanFlops)
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
	RunE:    runLayerStats,
}

func init() {
	rootCmd.AddCommand(layerstatsCmd)
}
