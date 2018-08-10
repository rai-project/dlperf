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

var rootCmd = &cobra.Command{
	Use:   "dlperf",
	Short: "Compute dlperf",
	PersistentPreRun: func(cmd *cobra.Command, args []string) {
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
	},
}

func init() {
	rootCmd.PersistentFlags().StringVar(&modelPath, "model_path", "", "path to the model prototxt file")
	rootCmd.PersistentFlags().BoolVar(&humanFlops, "human", false, "print flops in human form")
	rootCmd.PersistentFlags().BoolVar(&fullFlops, "full", false, "print all information about flops")

	rootCmd.PersistentFlags().BoolVar(&noHeader, "no_header", false, "show header labels for output")
	rootCmd.PersistentFlags().StringVarP(&outputFileName, "output", "o", "", "output file name")
	rootCmd.PersistentFlags().StringVarP(&outputFormat, "format", "f", "automatic", "print format to use")

	goPath = com.GetGOPATHs()[0]
	raiSrcPath = getSrcPath("github.com/rai-project")
}

func Execute() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}
