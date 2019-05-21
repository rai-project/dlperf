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
	"github.com/spf13/cobra"
	"golang.org/x/sync/errgroup"
)

func opsetName(url string) string {
	re := regexp.MustCompile(`(?m).*opset_(\d+)\/.*`)
	match := re.FindAllString(url, -1)
	if len(match) == 0 {
		return ""
	}
	return "opset_" + match[0]
}

// downloadModelsCmd represents the downloadmodels command
var downloadModelsCmd = &cobra.Command{
	Use:     "downloadmodels",
	Aliases: []string{"download"},
	RunE: func(cmd *cobra.Command, args []string) error {
		g, _ := errgroup.WithContext(context.Background())
		if !com.IsDir(outputFileName) {
			os.MkdirAll(outputFileName, os.ModePerm)
		}

		progress := newProgress("> Downloading models", len(modelURLs))
		defer progress.Finish()

		for ii := range modelURLs {
			url := modelURLs[ii]
			g.Go(func() error {
				defer progress.Increment()
				outputDir := outputFileName
				if opsetName(url) != "" {
					outputDir = filepath.Join(outputFileName, opsetName(url))
				}
				resp, err := http.Get(url)
				if err != nil {
					log.WithError(err).WithField("url", url).Error("failed to download model")
					return nil
				}
				defer resp.Body.Close()
				if strings.HasSuffix(url, ".onnx") {
					filename := filepath.Join(outputDir, path.Base(url))
					f, err := os.OpenFile(filename, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
					if err != nil {
						return err
					}
					defer f.Close()
					_, err = io.Copy(f, resp.Body)
					if err != nil {
						defer os.RemoveAll(filename)
						log.WithError(err).WithField("url", url).Error("failed to write model")
						return nil
					}
					return nil
				}
				err = archive.Unzip(resp.Body, outputDir, archive.GZipFormat())
				if err != nil {
					defer os.RemoveAll(outputFileName)
					log.WithError(err).WithField("url", url).Error("failed to decompress model")
					return nil
				}

				return nil
			})
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
