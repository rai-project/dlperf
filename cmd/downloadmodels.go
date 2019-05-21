package cmd

import (
	"context"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/Unknwon/com"
	getter "github.com/hashicorp/go-getter"
	"github.com/rai-project/config"
	"github.com/spf13/cobra"
	"golang.org/x/sync/errgroup"
)

func opsetName(url string) string {
	re := regexp.MustCompile(`.*opset_(\d+)\/.*`)
	match := re.FindStringSubmatch(url)
	if len(match) <= 1 {
		return ""
	}
	return "opset_" + match[1]
}

// downloadModelsCmd represents the downloadmodels command
var downloadModelsCmd = &cobra.Command{
	Use:     "downloadmodels",
	Aliases: []string{"download"},
	PreRunE: func(cmd *cobra.Command, args []string) error {
		if outputFileName == "automatic" || outputFileName == "" {
			outputFileName = filepath.Join(config.App.TempDir, "dlperf")
		}

		if !com.IsDir(outputFileName) {
			os.MkdirAll(outputFileName, os.ModePerm)
		}

		return nil
	},
	RunE: func(cmd *cobra.Command, args []string) error {
		concurrencyLimit := 4
		g, _ := errgroup.WithContext(context.Background())

		progress := newProgress("> Downloading models", len(modelURLs))
		defer progress.Finish()

		sem := make(chan bool, concurrencyLimit)
		for ii := range modelURLs {
			url := modelURLs[ii]

			sem <- true

			g.Go(func() error {
				defer progress.Increment()
				defer func() { <-sem }()

				outputDir := outputFileName
				if opsetName(url) != "" {
					outputDir = filepath.Join(outputFileName, opsetName(url))
				}

				targetFilePath := path.Base(url)
				if strings.HasSuffix(url, ".onnx") {
					targetFilePath = path.Base(url)
				}

				targetFilePath = filepath.Join(outputDir, targetFilePath)

				client := &getter.Client{
					Src:  url,
					Dst:  targetFilePath,
					Pwd:  outputDir,
					Mode: getter.ClientModeFile,
				}
				if err := client.Get(); err != nil {
					return err
				}
				return nil
			})
		}

		for i := 0; i < cap(sem); i++ {
			sem <- true
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
