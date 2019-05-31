package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/Unknwon/com"
	"github.com/spf13/cobra"
)

var (
	modelPath      string
	modelDir       string
	batchSize      int64
	fullInfo       bool
	humanFlops     bool
	outputFormat   string
	outputFileName string
	noHeader       bool
	appendOutput   bool
	pruneGraph     bool
	goPath         = com.GetGOPATHs()[0]
	raiSrcPath     = getSrcPath("github.com/rai-project")
)

var rootCmd = &cobra.Command{
	Use:   "dlperf",
	Short: "Compute dlperf",
	PersistentPreRun: func(cmd *cobra.Command, args []string) {
		if outputFormat == "automatic" {
			outputFormat = ""
		}

		if outputFormat == "" {
			if outputFileName == "" {
				outputFormat = "table"
			} else {
				outputFormat = strings.TrimLeft(filepath.Ext(outputFileName), ".")
				if outputFormat == "js" {
					outputFormat = "json"
				} else {
					outputFormat = "table"
				}
			}
		}

	},
}

func init() {
	rootCmd.PersistentFlags().StringVarP(&modelPath, "model_path", "p", "", "path to the model prototxt file")
	rootCmd.PersistentFlags().StringVarP(&modelDir, "model_dir", "d", "", "model directory")
	rootCmd.PersistentFlags().Int64VarP(&batchSize, "batch_size", "b", 1, "batch size")
	rootCmd.PersistentFlags().BoolVar(&humanFlops, "human", false, "print flops in human form")
	rootCmd.PersistentFlags().BoolVar(&fullInfo, "full", false, "print all information about the layers")
	rootCmd.PersistentFlags().BoolVar(&noHeader, "no_header", false, "show header labels for output")
	rootCmd.PersistentFlags().StringVarP(&outputFileName, "output_file", "o", "", "output file name")
	rootCmd.PersistentFlags().StringVarP(&outputFormat, "format", "f", "automatic", "print format to use")

	Init()
}

func Execute() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}
