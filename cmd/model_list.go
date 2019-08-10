package cmd

import (
	"path/filepath"

	"github.com/pkg/errors"
	"github.com/rai-project/config"
	"github.com/rai-project/dlperf/pkg/writer"
	"github.com/spf13/cobra"
)

var mostListCmd = &cobra.Command{
	Use: "list",
	Aliases: []string{
		"model_list",
		"modellist",
	},
	Short: "list models registered",
	Run: func(cmd *cobra.Command, args []string) {
		writer := NewWriter(modelURLInfo{})
		defer writer.Close()
		for _, model := range modelURLs {
			writer.Row(model)
		}
	},
}

type modelPathInfo struct {
	Name string `json:"name,omitempty"`
	Path string `json:"path,omitempty"`
}

func (modelPathInfo) Header(opts ...writer.Option) []string {
	return []string{"Name", "Path"}
}
func (m modelPathInfo) Row(opts ...writer.Option) []string {
	return []string{m.Name, m.Path}
}

var mostPathsCmd = &cobra.Command{
	Use: "paths",
	Aliases: []string{
		"model_paths",
		"modelpaths",
	},
	Short: "list model paths registered",
	PreRunE: func(cmd *cobra.Command, args []string) error {
		if modelPath == "automatic" || modelPath == "" {
			modelPath = filepath.Join(config.App.TempDir, "dlperf")
		}
		return nil
	},
	RunE: func(cmd *cobra.Command, args []string) error {
		modelPaths, err := getModelsIn(modelPath)
		if err != nil {
			return errors.Wrapf(err, "unable to glob %s", modelPath)
		}
		modelPathInfos := make([]modelPathInfo, len(modelPaths))
		for ii, modelPath := range modelPaths {
			modelPathInfos[ii] = modelPathInfo{
				Name: getModelName(modelPath),
				Path: modelPath,
			}
		}
		writer := NewWriter(modelPathInfo{})
		defer writer.Close()
		for _, elem := range modelPathInfos {
			writer.Row(elem)
		}
		return nil
	},
}

func init() {
	rootCmd.AddCommand(mostListCmd)
	rootCmd.AddCommand(mostPathsCmd)
}
