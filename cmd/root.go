package cmd

import (
	"path/filepath"
	"strings"

	"github.com/Unknwon/com"
	"github.com/pkg/errors"
	"github.com/rai-project/dlframework"
	"github.com/rai-project/dlperf/caffe"
	"github.com/spf13/cobra"
)

var (
	modelName           string
	modelVersion        string
	modelPath           string
	fullFlops           bool
	humanFlops          bool
	outputFormat        string
	outputFileName      string
	outputFileExtension string
	noHeader            bool
	appendOutput        bool
	goPath              string
	raiSrcPath          string
	mlArcWebAssetsPath  string
)

func cleanPath(path string) string {
	path = strings.Replace(path, ":", "_", -1)
	path = strings.Replace(path, " ", "_", -1)
	path = strings.Replace(path, "-", "_", -1)
	return strings.ToLower(path)
}

func getGraphPath(model *dlframework.ModelManifest) string {
	graphPath := filepath.Base(model.GetModel().GetGraphPath())
	wd, _ := model.WorkDir()
	return cleanPath(filepath.Join(wd, graphPath))
}

var FlopsInfoCmd = &cobra.Command{
	Use: "flops",
	Aliases: []string{
		"model",
		"theoretical_flops",
	},
	Short: "Get flops information about the model",
	PreRunE: func(cmd *cobra.Command, args []string) error {
		if Framework.Name == "" || Framework.Version == "" {
			return errors.New("Framework is not set.")
		}
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

		if modelName != "all" {
			outputFileExtension = filepath.Ext(outputFileName)
		} else {
			outputFileExtension = outputFormat
		}

		if modelName == "all" && outputFormat == "json" {
			dir := "theoretical_flops"
			if fullFlops {
				dir += "_full"
			}
			outputFileName = filepath.Join(mlArcWebAssetsPath, dir)
		}
		return nil
	},
	RunE: func(c *cobra.Command, args []string) error {
		run := func() error {
			var graphPath string
			if modelPath == "" {
				model, err := Framework.FindModel(modelName + ":" + modelVersion)
				if err != nil {
					return err
				}

				graphPath = getGraphPath(model)
			} else {
				if s, err := filepath.Abs(modelPath); err == nil {
					graphPath = s
				} else {
					graphPath = modelPath
				}
			}

			if !com.IsFile(graphPath) {
				return errors.Errorf("file %v does not exist", graphPath)
			}

			net, err := caffe.NewCaffe(graphPath)
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
							Type:             info.Type(),
							FlopsInformation: info.Flops(),
							Total:            info.Flops().Total(),
						},
					)
				}

				return nil
			}

			// if outputFormat == "json" {
			// 	err := errors.New("json output is not currently supported for full flop information")
			// 	log.WithError(err).Error("unable to output json")
			// 	return err
			// }

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

		return forallmodels(run)
	},
}

func init() {
	FlopsInfoCmd.PersistentFlags().StringVar(&modelName, "model_name", "BVLC-AlexNet", "modelName")
	FlopsInfoCmd.PersistentFlags().StringVar(&modelVersion, "model_version", "1.0", "modelVersion")
	FlopsInfoCmd.PersistentFlags().StringVar(&modelPath, "model_path", "", "path to the model prototxt file")
	FlopsInfoCmd.PersistentFlags().BoolVar(&humanFlops, "human", false, "print flops in human form")
	FlopsInfoCmd.PersistentFlags().BoolVar(&fullFlops, "full", false, "print all information about flops")

	FlopsInfoCmd.PersistentFlags().BoolVar(&noHeader, "no_header", false, "show header labels for output")
	FlopsInfoCmd.PersistentFlags().StringVarP(&outputFileName, "output", "o", "", "output file name")
	FlopsInfoCmd.PersistentFlags().StringVarP(&outputFormat, "format", "f", "automatic", "print format to use")

	goPath = com.GetGOPATHs()[0]
	raiSrcPath = getSrcPath("github.com/rai-project")
	mlArcWebAssetsPath = filepath.Join(raiSrcPath, "ml-arc-web", "src", "assets", "data")
}
