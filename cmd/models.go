package cmd

import (
	"fmt"

	"github.com/spf13/cobra"
)

var modelURLsCmd = &cobra.Command{
	Use:     "model_urls",
	Aliases: []string{"models"},
	RunE: func(cmd *cobra.Command, args []string) error {
		for _, model := range modelURLs {
			fmt.Printf("(\"%s\", \"%s\"), \n", model.Name, model.URL)
		}
		return nil
	},
}

func init() {
	rootCmd.AddCommand(modelURLsCmd)
}
