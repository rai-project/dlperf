package cmd

import (
	"path/filepath"
	"strings"

	"github.com/k0kubun/pp"

	sourcepath "github.com/GeertJohan/go-sourcepath"
	"github.com/Unknwon/com"
	"github.com/pkg/errors"
	"github.com/rai-project/dlperf/onnx"
	"github.com/spf13/cobra"
)

var (
	modelPath      string
	fullFlops      bool
	humanFlops     bool
	outputFormat   string
	outputFileName string
	outputPath     string
	noHeader       bool
	appendOutput   bool
	goPath         string
	raiSrcPath     string
)

var FlopsInfoCmd = &cobra.Command{
	Use: "flops",
	Aliases: []string{
		"theoretical_flops",
	},
	Short: "Get flops information about the model",
	PreRunE: func(cmd *cobra.Command, args []string) error {
		if outputFormat == "automatic" {
			outputFormat = ""
		}
		if outputFormat == "" && outputFileName != "" {
			outputFormat = strings.TrimLeft(filepath.Ext(outputFileName), ".")
			if outputFormat == "js" {
				outputFormat = "json"
			}
		} else {
			outputFormat = "table"
		}

		if outputFormat == "json" {
			dir := "theoretical_flops"
			if fullFlops {
				dir += "_full"
			}
			outputFileName = filepath.Join(outputPath, dir)
		}
		return nil
	},
	RunE: func(c *cobra.Command, args []string) error {
		run := func() error {
			modelFile := filepath.Join(sourcepath.MustAbsoluteDir(), "..", "assets", "onnx_models", "mnist.onnx")
			pp.Println(filepath.Abs(modelPath))

			if modelPath != "" {
				s, err := filepath.Abs(modelPath)
				if err == nil {
					modelFile = s
				}
			}

			if !com.IsFile(modelFile) {
				return errors.Errorf("file %v does not exist", modelFile)
			}

			net, err := onnx.NewOnnx(modelFile)
			if err != nil {
				return err
			}

			if fullFlops {
				infos := net.LayerInformations()

				writer := NewWriter(layer{}, humanFlops)
				defer writer.Close()

				for _, info := range infos {
					writer.Row(
						layer{
							Name:             info.Name(),
							Type:             info.OperatorType(),
							FlopsInformation: info.Flops(),
							Total:            info.Flops().Total(),
						},
					)
				}

				return nil
			}

			info := net.FlopsInformation()

			writer := NewWriter(netSummary{}, humanFlops)
			defer writer.Close()

			writer.Row(netSummary{Name: "MultipleAdds", Value: info.MultiplyAdds})
			writer.Row(netSummary{Name: "Additions", Value: info.Additions})
			writer.Row(netSummary{Name: "Divisions", Value: info.Divisions})
			writer.Row(netSummary{Name: "Exponentiations", Value: info.Exponentiations})
			writer.Row(netSummary{Name: "Comparisons", Value: info.Comparisons})
			writer.Row(netSummary{Name: "General", Value: info.General})
			writer.Row(netSummary{Name: "Total", Value: info.Total()})

			return nil
		}

		return run()
	},
}

func init() {
	FlopsInfoCmd.PersistentFlags().StringVar(&modelPath, "model_path", "", "path to the model prototxt file")
	FlopsInfoCmd.PersistentFlags().BoolVar(&humanFlops, "human", false, "print flops in human form")
	FlopsInfoCmd.PersistentFlags().BoolVar(&fullFlops, "full", true, "print all information about flops")

	FlopsInfoCmd.PersistentFlags().BoolVar(&noHeader, "no_header", false, "show header labels for output")
	FlopsInfoCmd.PersistentFlags().StringVarP(&outputFileName, "output", "o", "", "output file name")
	FlopsInfoCmd.PersistentFlags().StringVarP(&outputFormat, "format", "f", "automatic", "print format to use")

	goPath = com.GetGOPATHs()[0]
	raiSrcPath = getSrcPath("github.com/rai-project")
}
