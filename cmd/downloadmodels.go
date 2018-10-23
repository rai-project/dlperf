package cmd

import (
	"context"
	"io"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/Unknwon/com"
	"github.com/rai-project/archive"
	"github.com/spf13/cobra"
	"golang.org/x/sync/errgroup"
)

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
				resp, err := http.Get(url)
				if err != nil {
					log.WithError(err).WithField("url", url).Error("failed to download model")
					return nil
				}
				defer resp.Body.Close()
				if strings.HasSuffix(url, ".onnx") {
					filename := filepath.Join(outputFileName, path.Base(url))
					f, err := os.OpenFile(filename, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
					if err != nil {
						return err
					}
					defer f.Close()
					_, err = io.Copy(f, resp.Body)
					if err != nil {
						log.WithError(err).WithField("url", url).Error("failed to write model")
						return nil
					}
					return nil
				}
				err = archive.Unzip(resp.Body, outputFileName)
				if err != nil {
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
