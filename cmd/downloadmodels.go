package cmd

import (
	"context"
	"io"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/Unknwon/com"
	"github.com/rai-project/archive"
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
	Aliases: []string{"download", "download_models"},
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
			model := modelURLs[ii]
			url := model.URL
			modelName := model.Name

			sem <- true

			g.Go(func() error {
				defer progress.Increment()
				defer func() { <-sem }()

				outputDir := outputFileName
				if opsetName(url) != "" && modelName == "" {
					outputDir = filepath.Join(outputFileName, opsetName(url))
				}

				targetFilePath := path.Base(url)
				if strings.HasSuffix(url, ".onnx") {
					targetFilePath = path.Base(url)
				}

				if modelName != "" {
					outputDir = filepath.Join(outputFileName, modelName)
					if !strings.HasSuffix(url, ".onnx") {
						targetFilePath = modelName + ".tar.gz"
					} else {
						targetFilePath = modelName + ".onnx"

					}
				}

				targetFilePath = filepath.Join(outputDir, targetFilePath)

				if !com.IsDir(outputDir) {
					os.MkdirAll(outputDir, os.ModePerm)
				}

				com.WriteFile(filepath.Join(outputDir, "model_name"), []byte(modelName))

				resp, err := http.Get(url)
				if err != nil {
					log.WithError(err).WithField("url", url).Error("failed to download model")
					return nil

				}
				defer resp.Body.Close()
				if strings.HasSuffix(url, ".onnx") {
					f, err := os.OpenFile(targetFilePath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
					if err != nil {
						return err

					}
					defer f.Close()
					_, err = io.Copy(f, resp.Body)
					if err != nil {
						defer os.RemoveAll(targetFilePath)
						log.WithError(err).WithField("url", url).Error("failed to write model")
						return nil

					}
					return nil

				}
				err = archive.Unzip(resp.Body, outputDir, archive.GZipFormat())
				if err != nil {
					defer os.RemoveAll(outputDir)
					log.WithError(err).WithField("url", url).Error("failed to decompress model")
					return nil

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
